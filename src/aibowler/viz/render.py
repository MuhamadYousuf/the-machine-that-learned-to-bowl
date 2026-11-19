"""Shared Pillow drawing primitives used by the offline clip exporter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (8, 13, 24)
PANEL = (14, 23, 37)
GRID = (42, 58, 78)
INK = (232, 241, 251)
MUTED = (147, 167, 189)
TEAL = (73, 222, 190)
PINK = (251, 113, 133)
AMBER = (251, 191, 36)
VIOLET = (167, 139, 250)
GREEN = (94, 224, 160)


def _font(size: int):
    for name in ("C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas(size: tuple[int, int], transparent: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA" if transparent else "RGB", size, (0, 0, 0, 0) if transparent else BG)
    return image, ImageDraw.Draw(image, "RGBA" if transparent else None)


def _mix(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def diverging(value: float, maximum: float) -> tuple[int, int, int]:
    t = max(-1.0, min(1.0, float(value) / max(abs(float(maximum)), 1e-9)))
    return _mix((16, 24, 38), PINK if t < 0 else TEAL, abs(t) ** 0.72)


def interpolate_models(a: dict, b: dict, amount: float) -> dict:
    if a is None:
        return b
    t = float(max(0.0, min(1.0, amount)))
    out = dict(b)
    out["weights"] = [
        [[float(a["weights"][l][i][j]) * (1 - t) + float(y) * t
          if l < len(a.get("weights", [])) and i < len(a["weights"][l]) and j < len(a["weights"][l][i])
          else float(y)
          for j, y in enumerate(row)] for i, row in enumerate(matrix)]
        for l, matrix in enumerate(b["weights"])
    ]
    out["biases"] = [
        [float(a["biases"][l][i]) * (1 - t) + float(y) * t
         if l < len(a.get("biases", [])) and i < len(a["biases"][l])
         else float(y)
         for i, y in enumerate(vector)]
        for l, vector in enumerate(b["biases"])
    ]
    return out


def draw_network(image: Image.Image, model: dict, activations=None, labels: bool = False, top_k: int = 360):
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    layers = [len(model["inputs"]), *model["hidden_sizes"], len(model["outputs"])]
    side = max(46, int(w * 0.13))
    xs = [side, side + (w - 2 * side) / 3, side + 2 * (w - 2 * side) / 3, w - side]
    margin_y = max(30, int(h * 0.06))
    nodes = [[(int(xs[layer]), int(margin_y + i * (h - 2 * margin_y) / max(1, layers[layer] - 1)))
              for i in range(layers[layer])] for layer in range(4)]
    edges = []
    for layer, matrix in enumerate(model["weights"]):
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                edges.append((abs(float(weight)), layer, i, j, float(weight)))
    edges.sort(reverse=True)
    selected = edges[:top_k]
    scale = max((e[0] for e in selected), default=1e-9)
    for magnitude, layer, i, j, weight in selected:
        a, b = nodes[layer][i], nodes[layer + 1][j]
        color = TEAL if weight >= 0 else PINK
        alpha = int(28 + 165 * magnitude / scale)
        draw.line((a, b), fill=(*color, alpha), width=max(1, round(1 + 3 * magnitude / scale)))
    acts = activations or []
    for layer, points in enumerate(nodes):
        for i, point in enumerate(points):
            value = None
            if layer == 1 and len(acts) > 0 and i < len(acts[0]): value = float(acts[0][i])
            if layer == 2 and len(acts) > 1 and i < len(acts[1]): value = float(acts[1][i])
            if layer == 3 and len(acts) > 2 and i < len(acts[2]): value = float(acts[2][i])
            color = (210, 225, 239) if value is None else diverging(value, 1)
            radius = max(4, min(9, int(w * (0.005 if layer in (0, 3) else 0.006))))
            x, y = point
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(*color, 245), outline=(235, 246, 255, 190), width=1)
            if labels and layer in (0, 3):
                text = (model["inputs"] if layer == 0 else model["outputs"])[i]
                anchor = "rm" if layer == 0 else "lm"
                pos = (x-radius-5, y) if layer == 0 else (x+radius+5, y)
                draw.text(pos, text, fill=MUTED, font=_font(max(11, int(w * .009))), anchor=anchor)
    if labels:
        for x, title in zip(xs, ("Inputs", "Hidden 1", "Hidden 2", "Actions")):
            draw.text((int(x), max(4, margin_y // 3)), title, fill=INK, font=_font(max(12, int(w * .011))), anchor="mm")
    return image


def draw_pitch(image: Image.Image, points: Iterable[dict], highlighted: int | None = None, labels: bool = False):
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    # Reserve the upper third for a comfortable map even in portrait formats.
    left, right, top, bottom = int(w * .12), int(w * .06), int(h * .12), int(h * .15)
    x0, x1, y0, y1 = left, w-right, top, h-bottom
    X = lambda x: x0 + max(0, min(14, float(x))) / 14 * (x1-x0)
    Y = lambda y: y1 - max(-1, min(1, float(y))) / 2 * (y1-y0) - (y1-y0)/2
    draw.rectangle((X(5), y0, X(8), y1), fill=(38, 94, 113, 65))
    for tick in range(0, 15, 2):
        x = X(tick); draw.line((x, y0, x, y1), fill=GRID, width=max(1, w//1200))
        if labels: draw.text((x, y1+5), str(tick), fill=MUTED, font=_font(max(10,w//110)), anchor="mt")
    for tick in (-1, -.5, 0, .5, 1):
        y = Y(tick); draw.line((x0, y, x1, y), fill=GRID, width=max(1, w//1200))
        if labels: draw.text((x0-7, y), f"{tick:g}", fill=MUTED, font=_font(max(10,w//110)), anchor="rm")
    draw.line((X(0), Y(-.32), X(0), Y(.32)), fill=INK, width=max(3,w//350))
    draw.line((X(.45), y0, X(.45), y1), fill=(225,237,248,180), width=max(1,w//900))
    pts = list(points)
    result_colors = {"runs": GREEN, "dot": (128,148,170), "wide": AMBER, "no_ball": (255,146,80),
                     "bowled": PINK, "lbw": PINK, "caught_behind": VIOLET, "caught_field": PINK}
    radius = max(4, w//230)
    for i, point in enumerate(pts):
        x, y = X(point.get("length_m", 0) or 0), Y(point.get("line_m", 0) or 0)
        result = str(point.get("result", "dot"))
        color = result_colors.get(result, TEAL)
        r = radius + (3 if i == highlighted else 0)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(*color,255), outline=INK if i == highlighted else None, width=max(1,w//900))
        if labels: draw.text((x,y-r-3),str(i+1),fill=INK,font=_font(max(10,w//115)),anchor="ms")
    if labels:
        draw.text(((x0+x1)//2, h-10), "Bounce distance from stumps (m) · good length shaded", fill=MUTED, font=_font(max(11,w//95)), anchor="ms")
        draw.text((12,(y0+y1)//2), "Line (m)", fill=MUTED, font=_font(max(10,w//110)))
    return image


def draw_trajectory(image: Image.Image, trajectory: list[dict], time_s: float):
    draw = ImageDraw.Draw(image, "RGBA")
    w,h=image.size
    if not trajectory: return image
    left,right,top,bottom=int(w*.12),int(w*.08),int(h*.1),int(h*.12)
    xmin,xmax=-2.4,2.4; zmax=3.2
    X=lambda x:left+(float(x)-xmin)/(xmax-xmin)*(w-left-right)
    Y=lambda z:h-bottom-max(0.0,min(zmax,float(z)))/zmax*(h-top-bottom)
    draw.line((left,h-bottom,w-right,h-bottom),fill=GRID,width=max(1,w//1000))
    chosen=[p for p in trajectory if float(p.get("t_s",0))<=time_s]
    coords=[(X(p.get("x_m",0)),Y(p.get("z_m",0))) for p in chosen]
    if len(coords)>1: draw.line(coords,fill=(*TEAL,230),width=max(3,w//500),joint="curve")
    if coords:
        x,y=coords[-1];r=max(5,w//230);draw.ellipse((x-r,y-r,x+r,y+r),fill=INK,outline=TEAL,width=max(2,w//700))
    return image


def draw_heatmap(image: Image.Image, model: dict, labels: bool = False):
    draw=ImageDraw.Draw(image,"RGBA");w,h=image.size
    weights=model["weights"];biases=model["biases"]
    top=int(h*.1);gap=int(h*.07);available=h-top-gap*2;cell_h=available/3
    for layer,matrix in enumerate(weights):
        arr=np.asarray(matrix,dtype=float);rows,cols=arr.shape
        x0=int(w*.11);y0=int(top+layer*(cell_h+gap));plot_w=int(w*.78);plot_h=int(cell_h*.78)
        dx=plot_w/max(1,cols);dy=plot_h/max(1,rows);scale=max(np.max(np.abs(arr)),1e-8)
        for i in range(rows):
            for j in range(cols):
                x=x0+j*dx;y=y0+i*dy;draw.rectangle((x,y,x+max(1,dx),y+max(1,dy)),fill=diverging(arr[i,j],scale))
        bar_y=y0+plot_h+max(4,int(cell_h*.08));bias=np.asarray(biases[layer]);bscale=max(np.max(np.abs(bias)),1e-8);bw=plot_w/max(1,len(bias))
        for j,v in enumerate(bias):draw.rectangle((x0+j*bw,bar_y,x0+(j+1)*bw,bar_y+max(3,int(cell_h*.06))),fill=diverging(v,bscale))
        if labels:draw.text((int(w*.04),y0),f"Layer {layer+1} weights / bias",fill=MUTED,font=_font(max(11,w//100)))
    return image


def draw_population(image: Image.Image, points, fitness, champion=None, ancestry=None, labels=False):
    draw=ImageDraw.Draw(image,"RGBA");w,h=image.size
    p=np.asarray(points,dtype=float)
    if not len(p):return image
    lo=p.min(axis=0);hi=p.max(axis=0);margin=int(min(w,h)*.08)
    X=lambda x:margin+(float(x)-lo[0])/(hi[0]-lo[0] or 1)*(w-2*margin)
    Y=lambda y:h-margin-(float(y)-lo[1])/(hi[1]-lo[1] or 1)*(h-2*margin)
    fit=np.asarray(fitness if fitness is not None else np.zeros(len(p)),dtype=float);f0,f1=float(fit.min()),float(fit.max())
    ancestry=set(ancestry or []);rad=max(3,w//330)
    for i,(x,y) in enumerate(p):
        f=(fit[i]-f0)/(f1-f0 or 1);color=_mix((40,72,142),(73,222,190),f)
        draw.ellipse((X(x)-rad,Y(y)-rad,X(x)+rad,Y(y)+rad),fill=(*color,205))
        if i in ancestry:draw.ellipse((X(x)-rad*2,Y(y)-rad*2,X(x)+rad*2,Y(y)+rad*2),outline=AMBER,width=max(2,w//800))
    if champion is not None:
        x,y=champion;r=max(7,w//180);draw.regular_polygon((X(x),Y(y),r),5,rotation=-90,fill=AMBER,outline=INK)
    if labels:draw.text((margin,margin//2),"Shared PCA population projection",fill=MUTED,font=_font(max(11,w//100)))
    return image


def draw_learning_curve(image: Image.Image, series: dict[str, list[dict]], generation: int | None = None,
                        events: dict[str, list[dict]] | None = None, labels: bool = False):
    draw=ImageDraw.Draw(image,"RGBA");w,h=image.size
    left,right,top,bottom=int(w*.08),int(w*.04),int(h*.08),int(h*.12);all_values=[]
    for rows in series.values():
        for row in rows:
            try:
                val=float(row.get("champion_validation_score") or row.get("validation_score") or "nan")
                if math.isfinite(val):all_values.append(val)
            except (TypeError,ValueError):pass
    lo=min([0.0,*all_values]);hi=max([1.1,*all_values]);
    X=lambda g:left+(int(g)-1)/1499*(w-left-right)
    Y=lambda v:h-bottom-(float(v)-lo)/(hi-lo or 1)*(h-top-bottom)
    for i in range(5):
        y=top+i*(h-top-bottom)/4;draw.line((left,y,w-right,y),fill=GRID,width=max(1,w//1300))
    for name,rows in series.items():
        color=VIOLET if name=="ga1500" else TEAL;prev=None
        for row in rows:
            try:g=int(row["gen"]);v=float(row.get("champion_validation_score") or row.get("validation_score") or "nan")
            except (KeyError,TypeError,ValueError):continue
            if not math.isfinite(v):continue
            point=(X(g),Y(v))
            if prev:draw.line((prev,point),fill=(*color,220),width=max(2,w//700))
            prev=point
    for run,items in (events or {}).items():
        for event in items:
            if event.get("type") not in ("power_cut","resume","gate_override"):continue
            g=event.get("generation",event.get("interrupted_after_generation",1));x=X(g);c=PINK if event["type"]=="power_cut" else AMBER if event["type"]=="gate_override" else TEAL
            draw.line((x,top,x,h-bottom),fill=(*c,120),width=max(1,w//1100))
    if generation is not None:
        x=X(generation);draw.line((x,top,x,h-bottom),fill=INK,width=max(2,w//700))
    if labels:
        draw.text((left,top//2),"Held-out normalized champion score",fill=MUTED,font=_font(max(11,w//100)))
        draw.text((left,h-8),"Generation 1",fill=MUTED,font=_font(max(10,w//110)),anchor="ls")
        draw.text((w-right,h-8),"Generation 1500",fill=MUTED,font=_font(max(10,w//110)),anchor="rs")
    return image


def draw_family(image: Image.Image, family: dict, through_generation: int | None = None, labels=False):
    draw=ImageDraw.Draw(image,"RGBA");w,h=image.size
    nodes=family.get("nodes",[])
    if not family.get("available") or not nodes:return image
    maxgen=int(family.get("champion_generation",max(n["generation"] for n in nodes)));mingen=min(n["generation"] for n in nodes)
    stop=maxgen if through_generation is None else int(through_generation)
    visible=[n for n in nodes if n["generation"]<=stop];byid={n["id"]:n for n in visible}
    left,right,top,bottom=int(w*.06),int(w*.05),int(h*.18),int(h*.18)
    X=lambda g:left+(g-mingen)/max(1,maxgen-mingen)*(w-left-right)
    main_y=int(h*.48);pos={n["id"]:(X(n["generation"]),main_y+(int(h*.13) if n.get("donor") else 0)) for n in visible}
    for edge in family.get("edges",[]):
        if edge["from"] not in pos or edge["to"] not in pos:continue
        color=VIOLET if edge["kind"]=="crossover" else (77,100,128)
        draw.line((*pos[edge["from"]],*pos[edge["to"]]),fill=(*color,185),width=max(1,w//900))
    for node in visible:
        x,y=pos[node["id"]];r=max(3,w//420);color=AMBER if node.get("was_elite") else _mix(PINK,TEAL,max(0,min(1,(node["fitness"]+1)/2)))
        draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
        if node["id"]==family.get("champion_id"):draw.ellipse((x-r*2,y-r*2,x+r*2,y+r*2),outline=INK,width=2)
    if labels:draw.text((left,top//2),family.get("message","Champion ancestry"),fill=MUTED,font=_font(max(11,w//100)))
    return image


def draw_decision(image: Image.Image, values, labels=False, center=0.0):
    draw=ImageDraw.Draw(image,"RGBA");w,h=image.size;arr=np.asarray(values,dtype=float)
    top,left,bottom,right=int(h*.1),int(w*.1),int(h*.1),int(w*.08);rows,cols=arr.shape
    shifted=arr-float(center);maxv=float(np.max(np.abs(shifted))) or 1.0
    for i in range(rows):
        for j in range(cols):
            x=left+j*(w-left-right)/cols;y=top+(rows-1-i)*(h-top-bottom)/rows
            draw.rectangle((x,y,x+(w-left-right)/cols+1,y+(h-top-bottom)/rows+1),fill=diverging(shifted[i,j],maxv))
    if labels:draw.text((left,top//2),"Noise-free decoded policy response",fill=MUTED,font=_font(max(11,w//100)))
    return image


def resize_fit(image: Image.Image, size: tuple[int,int], background=BG) -> Image.Image:
    image=image.convert("RGB");image.thumbnail(size,Image.Resampling.LANCZOS)
    out=Image.new("RGB",size,background);out.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2));return out
