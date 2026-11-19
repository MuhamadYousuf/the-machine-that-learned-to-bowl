"""Render the reusable brain visuals into caption-free clip assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aibowler import brain_explorer
from aibowler.viz import render

FPS = 60
HOLD_FRAMES = 2 * FPS
FORMATS = {"9x16": (1080, 1920), "16x9": (1920, 1080)}
ARCHETYPES = ("slogger", "technician", "tailender")


def read_log(run_dir: Path) -> list[dict]:
    path = run_dir / "log.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_model(run_name: str, checkpoint: str) -> dict:
    return brain_explorer.network_payload(ROOT / "runs" / run_name, checkpoint)


def get_cam(run_name: str, checkpoint: str, archetype: str, seed: int) -> dict:
    return brain_explorer.brain_cam(ROOT / "runs" / run_name, checkpoint, archetype, seed)


@lru_cache(maxsize=1600)
def generation_model(run_name: str, generation: int) -> tuple[dict, int]:
    run_dir = ROOT / "runs" / run_name
    checkpoint, saved = brain_explorer.nearest_checkpoint_name(run_dir, generation)
    return get_model(run_name, checkpoint), saved


@lru_cache(maxsize=8)
def saved_generations(run_name: str) -> tuple[int, ...]:
    run_dir = ROOT / "runs" / run_name
    saved = set()
    for folder in ("best_genomes", "checkpoints"):
        for path in (run_dir / folder).glob("gen_*.npz"):
            try: saved.add(int(path.stem.split("_")[1]))
            except ValueError: pass
    if not saved:
        raise FileNotFoundError(f"no saved genomes in {run_dir}")
    return tuple(sorted(saved))


def interpolated_model(run_name: str, generation: int) -> tuple[dict, int]:
    saved = saved_generations(run_name)
    lower = max((g for g in saved if g <= generation), default=min(saved))
    upper = min((g for g in saved if g >= generation), default=max(saved))
    a, _ = generation_model(run_name, lower)
    b, _ = generation_model(run_name, upper)
    t = 0.0 if upper == lower else (generation-lower)/(upper-lower)
    t = max(0.0, min(1.0, t)); t = t*t*(3-2*t)
    return render.interpolate_models(a,b,t), lower if generation-lower <= upper-generation else upper


def common_population_series(run_name: str):
    runs_root = ROOT / "runs"
    genomes, labels, paths_by_run, fingerprint = brain_explorer._saved_populations(runs_root)
    if not len(genomes):
        return []
    mean, components, explained = brain_explorer._pca_basis_cached(fingerprint)
    snapshots = []
    for generation, path in paths_by_run.get(run_name, []):
        with np.load(path, allow_pickle=False) as data:
            population = data["population"].astype(float)
            fitness = data["fitness"].astype(float) if "fitness" in data.files else np.zeros(len(population))
            ids = data["genome_ids"].astype(int) if "genome_ids" in data.files else np.arange(len(population))
        coords = (brain_explorer._expand_population(population) - mean) @ components.T
        snapshots.append({"generation": generation, "points": coords, "fitness": fitness, "ids": ids})
    return sorted(snapshots, key=lambda item: item["generation"]), explained


def lerp_array(a, b, t):
    if a is None:
        return np.asarray(b)
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        rows, cols = min(aa.shape[0], bb.shape[0]), min(aa.shape[1], bb.shape[1])
        out = bb.copy()
        out[:rows, :cols] = aa[:rows, :cols] * (1 - t) + bb[:rows, :cols] * t
        return out
    return aa * (1 - t) + bb * t


def _new_image(size, transparent=False):
    return render.canvas(size, transparent)[0]


def _network_frame(model, size, labels=False, activations=None):
    image = _new_image(size)
    render.draw_network(image, model, activations=activations, labels=labels)
    return image


def _cam_frame(model, delivery, progress, size, labels=False, trajectory_scale=1.0):
    image = _new_image(size)
    w, h = size
    if w > h:
        network_box = (int(w * .62), h)
        network = _new_image(network_box)
        render.draw_network(network, model, activations=(delivery["hidden1"], delivery["hidden2"], delivery["action"]), labels=labels)
        image.paste(network, (0, 0))
        x = network_box[0]; right_w = w - x; upper_h = h // 2
        trajectory_panel = _new_image((right_w, upper_h))
        render.draw_trajectory(trajectory_panel, delivery["trajectory"], progress / trajectory_scale)
        image.paste(trajectory_panel, (x, 0))
        pitch_panel = _new_image((right_w, h-upper_h))
        point = {"length_m": delivery["delivery"]["length_m"], "line_m": delivery["delivery"]["line_m"], "result": delivery["delivery"]["result"]}
        render.draw_pitch(pitch_panel, [point], labels=labels)
        image.paste(pitch_panel, (x, upper_h))
    else:
        network_h = int(h * .55)
        network = _new_image((w, network_h))
        render.draw_network(network, model, activations=(delivery["hidden1"], delivery["hidden2"], delivery["action"]), labels=labels)
        image.paste(network, (0, 0))
        trajectory_h = int(h * .24)
        panel = _new_image((w, trajectory_h))
        render.draw_trajectory(panel, delivery["trajectory"], progress / trajectory_scale)
        image.paste(panel, (0, network_h))
        point = {"length_m": delivery["delivery"]["length_m"], "line_m": delivery["delivery"]["line_m"], "result": delivery["delivery"]["result"]}
        pitch_panel = _new_image((w, h-network_h-trajectory_h))
        render.draw_pitch(pitch_panel, [point], labels=labels)
        image.paste(pitch_panel, (0, network_h+trajectory_h))
    if labels:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((16, 12), f"{delivery['delivery']['result']} · {delivery['release']['speed_kmh']:.1f} km/h", fill=(232,241,251), font=font)
    return image


def network_morph_frames(run_name, size, labels=False, count=FPS*16):
    previous_gen, previous = None, None
    for frame in range(count):
        u = frame / max(1, count - 1)
        eased = u * u * (3 - 2 * u)
        generation = 1 + round(eased * 1499)
        model, saved = interpolated_model(run_name, generation)
        previous_gen, previous = saved, model
        image = _network_frame(model, size, labels)
        if labels:
            ImageDraw.Draw(image).text((18, 14), f"Generation {generation} · saved {previous_gen}", fill=(232,241,251), font=ImageFont.load_default())
        yield image, {"generation": generation, "event": "morph"}


def milestone_frames(run_name, size, labels=False):
    for generation in (1, 100, 500, 1500):
        model, saved = generation_model(run_name, generation)
        image = _network_frame(model, size, labels)
        if labels:
            ImageDraw.Draw(image).text((18, 14), f"Generation {generation} · saved {saved}", fill=(232,241,251), font=ImageFont.load_default())
        for _ in range(2 * FPS):
            yield image, {"generation": generation, "event": "milestone_hold"}


def brain_cam_frames(run_name, checkpoint, archetype, seed, size, labels=False, slow_motion=1.0):
    cam = get_cam(run_name, checkpoint, archetype, seed)
    model = get_model(run_name, checkpoint)
    for delivery in cam["deliveries"]:
        events = delivery["event_times"]
        end = max(float(events.get("stumps_s", 0)), float(events.get("crease_s", 0)), .25)
        duration = end * slow_motion
        nframes = max(2, int(round(duration * FPS)))
        event_frames = {round(float(events.get(key, end)) * slow_motion * FPS): key.replace("_s", "")
                        for key in ("release_s", "bounce_s", "crease_s", "stumps_s")}
        for index in range(nframes):
            elapsed = index / FPS
            image = _cam_frame(model, delivery, elapsed, size, labels, slow_motion)
            event = event_frames.get(index, "flight")
            if index == nframes - 1:
                event = "outcome"
            if labels:
                ImageDraw.Draw(image).text((16, 12), f"{archetype} · ball {delivery['ball']} · {event}", fill=(232,241,251), font=ImageFont.load_default())
            yield image, {"generation": cam.get("checkpoint"), "ball_number": delivery["ball"], "event": event,
                          "simulation_time_s": min(elapsed / slow_motion, end)}


def population_frames(run_name, size, labels=False, count=FPS*20):
    snapshots, explained = common_population_series(run_name)
    if not snapshots:
        raise RuntimeError(f"no saved populations for {run_name}")
    family = brain_explorer.family_tree(ROOT / "runs" / run_name, "champion")
    ancestry_by_generation = {
        int(node["generation"]): int(node["id"])
        for node in family.get("nodes", []) if not node.get("donor")
    }
    for frame in range(count):
        u = frame / max(1, count - 1)
        target = 1 + round(u * 1499)
        lower = max((i for i, item in enumerate(snapshots) if item["generation"] <= target), default=0)
        upper = min((i for i, item in enumerate(snapshots) if item["generation"] >= target), default=len(snapshots)-1)
        a, b = snapshots[lower], snapshots[upper]
        if a["generation"] == b["generation"]:
            t = 0.0
        else:
            t = (target-a["generation"])/(b["generation"]-a["generation"])
            t = max(0., min(1., t)); t=t*t*(3-2*t)
        points=lerp_array(a["points"],b["points"],t);fitness=lerp_array(a["fitness"],b["fitness"],t)
        selected = a if t < .5 else b
        ancestor_id = ancestry_by_generation.get(int(selected["generation"]))
        ancestry_indices = [i for i, genome_id in enumerate(selected["ids"]) if int(genome_id) == ancestor_id]
        champion = points[ancestry_indices[0]] if ancestry_indices else points[int(np.argmax(fitness))]
        image=_new_image(size);render.draw_population(image,points,fitness,champion=champion,ancestry=ancestry_indices,labels=labels)
        if labels:
            ImageDraw.Draw(image).text((16,12),f"Generation {target} · PC variance {sum(explained[:2])*100:.1f}%",fill=(232,241,251),font=ImageFont.load_default())
        yield image,{"generation":target,"event":"population_morph","snapshot_generations":[a["generation"],b["generation"]],"champion_ancestor_id":ancestor_id}


def weight_frames(run_name, size, labels=False, count=FPS*20):
    start,_=generation_model(run_name,1);end,_=generation_model(run_name,1500)
    for frame in range(count):
        u=frame/max(1,count-1);eased=u*u*(3-2*u);model=render.interpolate_models(start,end,eased)
        image=_new_image(size);render.draw_heatmap(image,model,labels=labels)
        if labels:ImageDraw.Draw(image).text((16,12),f"Generation {1+round(u*1499)}",fill=(232,241,251),font=ImageFont.load_default())
        yield image,{"generation":1+round(u*1499),"event":"weight_morph"}


def decision_frames(run_name, size, labels=False, count=FPS*10):
    start=brain_explorer.decision_map(ROOT/"runs"/run_name,"gen_0001","slogger",10421,0,5,7,"speed",resolution=48)
    end=brain_explorer.decision_map(ROOT/"runs"/run_name,"champion","slogger",10421,0,5,7,"speed",resolution=48)
    a,b=np.asarray(start["values"]),np.asarray(end["values"])
    for frame in range(count):
        u=frame/max(1,count-1)
        image=_new_image(size)
        if u<.6:
            q=(u/.6);mix=Image.blend(_decision_panel(a,size,labels),_decision_panel(b,size,labels),q)
            image.paste(mix,(0,0))
        else:
            left=_decision_panel(a,(size[0]//2,size[1]),labels);right=_decision_panel(b,(size[0]-size[0]//2,size[1]),labels)
            image.paste(left,(0,0));image.paste(right,(size[0]//2,0))
        if labels:ImageDraw.Draw(image).text((14,12),"Generation 1 → champion",fill=(232,241,251),font=ImageFont.load_default())
        yield image,{"generation":1 if u<.6 else 1500,"event":"crossfade" if u<.6 else "side_by_side"}


def _decision_panel(values, size, labels):
    image=_new_image(size);render.draw_decision(image,values,labels=labels,center=132.5);return image


def family_frames(run_name,size,labels=False,count=FPS*10):
    family=brain_explorer.family_tree(ROOT/"runs"/run_name,"champion")
    maxgen=int(family.get("champion_generation",1500))
    for frame in range(count):
        gen=round(frame/max(1,count-1)*maxgen)
        image=_new_image(size);render.draw_family(image,family,through_generation=gen,labels=labels)
        yield image,{"generation":gen,"event":"lineage_reveal"}


def learning_frames(size, labels=False, count=FPS*12):
    rows={name:read_log(ROOT/"runs"/name) for name in ("ga1500","ga2")}
    configs={}
    for name in rows:
        try:configs[name]=json.loads((ROOT/"runs"/name/"run_config.json").read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):configs[name]={}
    events={name:list(configs[name].get("events",[])) for name in rows}
    for frame in range(count):
        u=frame/max(1,count-1)
        if u<.45:
            gen=max(1,int(u/.45*1500));cur={"ga1500":[r for r in rows["ga1500"] if int(r["gen"])<=gen]}
            cursor=gen
        else:
            gen=max(1,int((u-.45)/.55*1500));cur={"ga1500":rows["ga1500"],"ga2":[r for r in rows["ga2"] if int(r["gen"])<=gen]};cursor=gen
        image=_new_image(size);render.draw_learning_curve(image,cur,cursor,events,labels=labels)
        yield image,{"generation":cursor,"event":"ga1500_line" if u<.45 else "ga2_line"}


def comparison_frames(size, labels=False, count=FPS*10, seed=10421, archetype="slogger"):
    model1=get_model("ga1500","champion");model2=get_model("ga2","champion")
    cam1=get_cam("ga1500","champion",archetype,seed);cam2=get_cam("ga2","champion",archetype,seed)
    deliveries1,deliveries2=cam1["deliveries"],cam2["deliveries"]
    for ball_index in range(max(len(deliveries1),len(deliveries2))):
        d1=deliveries1[min(ball_index,len(deliveries1)-1)]
        d2=deliveries2[min(ball_index,len(deliveries2)-1)]
        events1,events2=d1["event_times"],d2["event_times"]
        end=max(float(events1.get("stumps_s",0)),float(events1.get("crease_s",0)),
                float(events2.get("stumps_s",0)),float(events2.get("crease_s",0)),.3)
        frame_count=max(2,int(round(end*FPS)))
        boundaries={}
        for event_times in (events1,events2):
            for key in ("release_s","bounce_s","crease_s","stumps_s"):
                boundaries[round(float(event_times.get(key,end))*FPS)]=key.replace("_s","")
        for frame in range(frame_count):
            t=frame/FPS
            image=_new_image(size)
            if size[0]>size[1]:
                half=size[0]//2;left=_cam_frame(model1,d1,t,(half,size[1]),labels);right=_cam_frame(model2,d2,t,(size[0]-half,size[1]),labels)
                image.paste(left,(0,0));image.paste(right,(half,0))
            else:
                half=size[1]//2;top=_cam_frame(model1,d1,t,(size[0],half),labels);bottom=_cam_frame(model2,d2,t,(size[0],size[1]-half),labels)
                image.paste(top,(0,0));image.paste(bottom,(0,half))
            event=boundaries.get(frame,"flight")
            if frame==frame_count-1:event="outcome"
            if labels:
                draw=ImageDraw.Draw(image);font=ImageFont.load_default();draw.text((10,10),f"ga1500 v1 · ball {ball_index+1}",fill=(232,241,251),font=font);draw.text((size[0]//2+10,10),f"ga2 v2 · ball {ball_index+1}",fill=(232,241,251),font=font)
            yield image,{"generation":{"ga1500":cam1.get("checkpoint"),"ga2":cam2.get("checkpoint")},"ball_number":ball_index+1,"event":event,"simulation_time_s":t}


def write_contact_sheet(samples: list[Image.Image], path: Path):
    tile=(400,225);sheet=Image.new("RGB",(tile[0]*3,tile[1]*3),(8,13,24))
    if not samples:
        samples=[Image.new("RGB",tile,BG)]
    for i in range(9):
        image=samples[min(i,len(samples)-1)]
        im=render.resize_fit(image,tile);sheet.paste(im,((i%3)*tile[0],(i//3)*tile[1]))
    sheet.save(path)


def write_variant(name: str, frames, output_root: Path, labels=False, png_sequence=False, transparent=False,
                  source_data: str="") -> dict:
    clip_dir=output_root/name;clip_dir.mkdir(parents=True,exist_ok=True)
    manifest={"name":name,"source_data":source_data,"fps":FPS,"formats":{},"contact_sheet":str((clip_dir/"contact_sheet.png").relative_to(output_root))}
    contact=[];durations={}
    for fmt,size in FORMATS.items():
        format_dir=clip_dir/fmt;format_dir.mkdir(parents=True,exist_ok=True)
        def make_frames():return iter(frames(size,labels))
        iterator=make_frames()
        try:first,first_info=next(iterator)
        except StopIteration:raise RuntimeError(f"clip {name} has no frames")
        first=first.convert("RGBA" if transparent else "RGB").resize(size,Image.Resampling.LANCZOS)
        if fmt=="16x9":contact.append(first.copy())
        frame_index=0;timing=[];samples=[]
        if png_sequence:
            frame_dir=format_dir/"frames";frame_dir.mkdir(exist_ok=True)
            def put(image,info):
                nonlocal frame_index
                image=image.convert("RGBA" if transparent else "RGB").resize(size,Image.Resampling.LANCZOS)
                if transparent:
                    rgba=np.asarray(image).copy();bg=np.asarray(BG,dtype=np.uint8)
                    rgba[np.all(rgba[:,:,:3]==bg,axis=2),3]=0;image=Image.fromarray(rgba,"RGBA")
                image.save(frame_dir/f"{frame_index:06d}.png")
                timing.append({"frame":frame_index,**info});frame_index+=1
                if fmt=="16x9" and (frame_index-1)%(max(1,1))==0 and (frame_index-1) in {0,120,240,480,720,960,1200,1440,1680}:
                    samples.append(image.convert("RGB").copy())
            for _ in range(HOLD_FRAMES):put(first,{**first_info,"event":"start_hold"})
            put(first,first_info)
            last=first;last_info=first_info
            for image,info in iterator:
                put(image,info);last=image;last_info=info
            for _ in range(HOLD_FRAMES):put(last,{**last_info,"event":"end_hold"})
        else:
            ffmpeg=shutil.which("ffmpeg")
            if not ffmpeg:raise RuntimeError("ffmpeg was not found on PATH")
            out_file=format_dir/f"{name}.mp4"
            cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{size[0]}x{size[1]}","-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","medium","-profile:v","high","-crf","16","-pix_fmt","yuv420p",str(out_file)]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            def put(image,info):
                nonlocal frame_index
                image=image.convert("RGB").resize(size,Image.Resampling.LANCZOS)
                proc.stdin.write(image.tobytes())
                timing.append({"frame":frame_index,**info});frame_index+=1
                if fmt=="16x9" and len(samples)<9 and (frame_index-1)%max(1,(FPS*30)//9)==0:samples.append(image.copy())
            try:
                for _ in range(HOLD_FRAMES):put(first,{**first_info,"event":"start_hold"})
                put(first,first_info)
                last=first;last_info=first_info
                for image,info in iterator:
                    put(image,info);last=image;last_info=info
                for _ in range(HOLD_FRAMES):put(last,{**last_info,"event":"end_hold"})
                proc.stdin.close();error=proc.stderr.read().decode("utf-8",errors="replace");code=proc.wait()
                if code:raise RuntimeError(f"ffmpeg exited {code}: {error[-1000:]}")
            finally:
                if proc.poll() is None:proc.kill()
        duration=frame_index/FPS;durations[fmt]=duration
        if fmt=="16x9":contact.extend(samples)
        sidecar={"clip":name,"format":fmt,"fps":FPS,"frame_count":frame_index,"duration_seconds":duration,
                 "resolution":list(size),"frame_timing":timing}
        (format_dir/"timing.json").write_text(json.dumps(sidecar,separators=(",",":")),encoding="utf-8")
        manifest["formats"][fmt]={"duration_seconds":duration,"file":str((format_dir/("frames" if png_sequence else f"{name}.mp4")).relative_to(output_root)),"resolution":list(size)}
    # Contact sheets are landscape for quick review and shareable across both aspect ratios.
    write_contact_sheet(contact,sheet_path:=clip_dir/"contact_sheet.png")
    manifest["duration_seconds"]={k:round(v,3) for k,v in durations.items()}
    return manifest


def build_clips(args):
    output=ROOT/"outputs"/"clips";output.mkdir(parents=True,exist_ok=True)
    labels=args.labels;seed=args.seed;items=[]
    items.append(write_variant("C1_network_morph",lambda size,labels:network_morph_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 per-generation best_genomes"))
    items.append(write_variant("C1_network_milestones",lambda size,labels:milestone_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 best genomes at generations 1, 100, 500, 1500"))
    for arch in ARCHETYPES:
        items.append(write_variant(f"C2_cam_{arch}",lambda size,labels,a=arch:brain_cam_frames("ga2","champion",a,seed,size,labels,args.slow_motion),output,labels,args.png_sequence,args.transparent,f"ga2 champion replay · {arch} · seed {seed} · {args.slow_motion:g}x slow motion"))
    items.append(write_variant("C3_population_swarm",lambda size,labels:population_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 saved populations · common PCA basis"))
    items.append(write_variant("C4_weight_morph",lambda size,labels:weight_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 generation 1 to champion weight matrices and biases"))
    items.append(write_variant("C5_decision_map",lambda size,labels:decision_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 speed decision map · fixed Slogger state and seed"))
    items.append(write_variant("C6_family_tree",lambda size,labels:family_frames("ga2",size,labels),output,labels,args.png_sequence,args.transparent,"ga2 champion lineage.csv.gz"))
    items.append(write_variant("C7_learning_curve",lambda size,labels:learning_frames(size,labels),output,labels,args.png_sequence,args.transparent,"ga1500 v1 and ga2 v2 validation logs/events"))
    items.append(write_variant("C8_v1_vs_v2",lambda size,labels:comparison_frames(size,labels,seed=seed,archetype="slogger"),output,labels,args.png_sequence,args.transparent,"ga1500 v1 and ga2 v2 champions · same Slogger seed"))
    index={"schema_version":1,"fps":FPS,"codec":"H.264 high profile · CRF 16 · yuv420p","clips":items}
    (output/"index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    print(json.dumps({"clips":[{"name":x["name"],"duration_seconds":x["duration_seconds"],"contact_sheet":x["contact_sheet"]} for x in items],"index":str(output/"index.json")},indent=2))


def main():
    parser=argparse.ArgumentParser(description="Render offline Brain Explorer clip frames and encode them with ffmpeg")
    parser.add_argument("--run",default="ga2",choices=("ga2",),help="evolution run for the v2 clip suite")
    parser.add_argument("--seed",type=int,default=10421)
    parser.add_argument("--labels",action="store_true",help="add small reference labels to visuals")
    parser.add_argument("--png-sequence",action="store_true",help="save every rendered frame as PNG instead of encoding MP4")
    parser.add_argument("--transparent",action="store_true",help="use alpha in PNG sequence frames")
    parser.add_argument("--slow-motion",type=float,default=1.0,help="slow-motion factor for Brain Cam clips")
    args=parser.parse_args()
    if args.slow_motion<=0:parser.error("--slow-motion must be positive")
    if args.transparent and not args.png_sequence:parser.error("--transparent requires --png-sequence")
    build_clips(args)


if __name__=="__main__":main()
