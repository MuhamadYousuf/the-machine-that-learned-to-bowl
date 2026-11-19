"""Render one baked 3D shot (from tools/bake_shot.mts) in the Blender stadium.

blender -b -P blender/render_shot.py -- <shot_dir> <cameras comma list> <W> <H> <out_dir> [frames e.g. 0-120 or 10,50,90]
"""

import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cricketer  # noqa: E402
import stadium  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
shot_dir, cam_list, W, H, out_dir = os.path.abspath(args[0]), args[1].split(","), int(args[2]), int(args[3]), os.path.abspath(args[4])
frame_sel = args[5] if len(args) > 5 else None
VIDEO = os.path.abspath(os.path.join(HERE, ".."))
cricketer.TEX_DIR = os.path.join(VIDEO, "public", "models", "bowler")
shot = json.load(open(os.path.join(shot_dir, "shot.json")))
fps = shot["fps"]
per = shot["perFrame"]

t0 = time.time()
stadium.clear()
stadium.render_settings(W, H, fps=fps, samples=48)
stadium.build(os.path.join(HERE, "assets", "moonless_golf_2k.hdr"))
try:
    stadium.compositor()
except Exception as exc:  # compositor API differs between versions; the render still works without it
    print("COMPOSITOR", exc)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 0, len(per) - 1

# ---------------------------------------------------------------- bowler (FBX, NLA strip time per frame)
bowler, _ = cricketer.import_cricketer(os.path.join(cricketer.TEX_DIR, "CricketBowlingAnimations.fbx"), "Bowler",
                                       {"top": None, "bottom": (0.0, 0.32, 0.02, 1)})
fast = next(a for a in bpy.data.actions if a.name.endswith("RightArmFastBowler"))
clip_fps = (fast.frame_range[1] - fast.frame_range[0]) / 7.58
rel, _scale = cricketer.place_bowler(bowler, fast, clip_fps)
f0 = fast.frame_range[0]
bowler.animation_data.action = None
track = bowler.animation_data.nla_tracks.new()
strip = track.strips.new("bowl", int(f0), fast)
strip.use_animated_time = True
for i, row in enumerate(per):
    strip.strip_time = f0 + (cricketer.RELEASE_S + row["bowler"]) * clip_fps
    strip.keyframe_insert("strip_time", frame=i)
for fc in (bowler.animation_data.nla_tracks[0].strips[0].fcurves if hasattr(strip, "fcurves") else []):
    for k in fc.keyframe_points:
        k.interpolation = "LINEAR"

# ---------------------------------------------------------------- batter (baked glTF)
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=os.path.join(shot_dir, "batter.glb"))
new = [o for o in bpy.data.objects if o not in before]
roots = [o for o in new if o.parent is None]
pivot = bpy.data.objects.new("BatterPivot", None)
sc.collection.objects.link(pivot)
for r in roots:
    r.parent = pivot
pivot.rotation_euler = (0, 0, math.pi)  # three/glTF (+X off side, -Y along) → our Blender frame (-X off side, +Y along)
kit = json.loads(os.environ.get("BATTER_KIT", '{"top": [0.62, 0.06, 0.09, 1], "bottom": [0.42, 0.04, 0.06, 1]}'))
mats = cricketer.kit_materials({"top": tuple(kit["top"]), "bottom": tuple(kit["bottom"])})
for o in new:
    if o.type == "MESH":
        key = next((k for k in mats if o.name.startswith(k)), None)
        if key:
            o.data.materials.clear()
            o.data.materials.append(mats[key])
        else:  # kit pieces keep their glTF colours; give them better surfaces
            for m in o.data.materials:
                if m and m.use_nodes and "Principled BSDF" in m.node_tree.nodes:
                    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.55
arm_b = next((o for o in new if o.type == "ARMATURE"), None)
if arm_b and arm_b.animation_data and arm_b.animation_data.action is None and bpy.data.actions.get("Bake"):
    cricketer.set_action(arm_b, bpy.data.actions["Bake"])
sc.render.fps = fps

# ---------------------------------------------------------------- ball
leather = stadium.mat("Ball", (0.86, 0.86, 0.83, 1), 0.42)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.036, segments=24, ring_count=16)
ball = bpy.context.active_object
ball.name = "Ball"
ball.data.materials.append(leather)
# Where the bowler's hand really is at release; the flying ball starts there and blends into the sim path.
hand_rel = cricketer.bone_world(bowler, "RightHand", f0 + cricketer.RELEASE_S * clip_fps) if False else None
BLEND = 0.15
hand_track = {}
for i, row in enumerate(per):
    if row["tau"] < 0.0 or (row["tau"] < BLEND and row["ball"] is not None):
        sc.frame_set(i)
        hb = bowler.pose.bones["RightHand"]
        # ball sits in the palm: a little along the hand bone from the wrist
        hand_track[i] = bowler.matrix_world @ (hb.head + (hb.tail - hb.head) * 0.9)
# hand position exactly at release (first frame with tau >= 0)
rel_frame = next(i for i, r in enumerate(per) if r["tau"] >= 0.0)
sc.frame_set(rel_frame)
hb = bowler.pose.bones["RightHand"]
release_hand = bowler.matrix_world @ (hb.head + (hb.tail - hb.head) * 0.9)
offset = release_hand - cricketer.SIM_RELEASE_B
print("RELEASE hand", tuple(round(x, 2) for x in release_hand), "offset to sim", tuple(round(x, 2) for x in offset))
for i, row in enumerate(per):
    if row["tau"] < 0.0:
        ball.location = hand_track[i]  # in the bowler's hand during the run-up
        vis = True
    elif row["ball"] is not None:
        u = min(1.0, row["tau"] / BLEND)
        w = 1.0 - u * u * (3 - 2 * u)
        ball.location = Vector(row["ball"]) + offset * w
        vis = True
    else:
        vis = False
    if vis:
        ball.keyframe_insert("location", frame=i)
    ball.hide_render = not vis
    ball.keyframe_insert("hide_render", frame=i)

# ---------------------------------------------------------------- stumps break on bowled
coll = bpy.data.collections.get("BatterStumps")
if coll and any(r["tilt"] > 0 for r in per):
    stumps = [o for o in coll.objects if "_s" in o.name]
    bails = [o for o in coll.objects if "_b" in o.name]
    for o in stumps + bails:
        o["base"] = list(o.location)
        o["base_rot"] = list(o.rotation_euler)
    for i, row in enumerate(per):
        for k, o in enumerate(stumps):
            # knocked backwards (+Y, away from the bowler), middle stump hardest, outer ones splayed sideways
            amount = row["tilt"] / 0.35
            back = (1.25 if k == 1 else 0.7) * amount
            o.rotation_euler = (-back, (k - 1) * 0.45 * amount, 0)
            o.keyframe_insert("rotation_euler", frame=i)
        for k, o in enumerate(bails):
            s = row["bail"]
            bx, by, bz = o["base"]
            vx, vy, vz = (-0.6 + 1.2 * k) * 1.4, 3.2, 3.4
            o.location = (bx + vx * s, by + vy * s, max(0.02, bz + vz * s - 4.9 * s * s))
            o.rotation_euler = (o["base_rot"][0] + 9 * s, o["base_rot"][1], o["base_rot"][2] + 6 * s * (1 if k else -1))
            o.keyframe_insert("location", frame=i)
            o.keyframe_insert("rotation_euler", frame=i)

# ---------------------------------------------------------------- cameras
BAT = Vector((0.42, 19.2, 1.0))
CAMS = {
    "broadcast": ((0.7, -17.0, 5.2), (0.0, 14.0, 0.9), 75, "batter"),
    "tele": ((0.5, -24.0, 3.0), (0.35, 19.0, 1.05), 230, "batter"),
    "batterSide": ((-8.5, 19.6, 1.35), (0.4, 19.1, 1.0), 55, "batter"),
    "stumps": ((0.2, 23.2, 0.8), (0.1, 12.0, 1.0), 40, "batter"),
    "release": ((1.0, 9.0, 1.7), (-0.35, -1.2, 1.7), 60, "bowler"),
    "wide": ((26.0, -28.0, 15.0), (0.0, 10.0, 0.0), 32, "pitch"),
    "square": ((-18.0, 9.5, 2.6), (0.0, 9.5, 1.0), 38, "pitch"),
    "stumpsLow": ((0.35, 22.6, 0.32), (0.0, 9.0, 0.75), 30, "pitch"),
    "runSide": (None, None, 40, "bowler"),  # tracks the bowler's run-up from the off side
}
bpy.ops.object.camera_add()
cam = bpy.context.active_object
sc.camera = cam
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = 2.0
focus = bpy.data.objects.new("Focus", None)
sc.collection.objects.link(focus)
cam.data.dof.focus_object = focus
for i, row in enumerate(per):
    name = cam_list[row["seg"] % len(cam_list)]
    loc, tgt, lens, foc = CAMS[name]
    # Evaluate the frame FIRST: frame_set re-applies existing camera keys, so anything set before it is lost.
    sc.frame_set(i)
    if name == "runSide":
        hp = bowler.matrix_world @ bowler.pose.bones["Hips"].head
        y = min(hp.y, 2.0)
        loc, tgt = (-9.5, y + 1.2, 1.6), (-0.4, y + 0.6, 1.3)
    if foc == "batter":
        focus.location = BAT
    elif foc == "bowler":
        focus.location = bowler.matrix_world @ bowler.pose.bones["Spine2"].head
    else:
        focus.location = (0, 10, 0.5)
    cam.location = loc
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens * (0.62 if H > W else 1.0)  # portrait: wider lens keeps the same vertical framing
    for path, owner in (("location", cam), ("rotation_euler", cam), ("location", focus)):
        owner.keyframe_insert(path, frame=i)
    cam.data.keyframe_insert("lens", frame=i)
for a in bpy.data.actions:
    for fc in getattr(a, "fcurves", []):
        if fc.data_path in ("location", "rotation_euler", "lens") and a.users and a.name.startswith(("CameraAction", "FocusAction", "Camera", "Focus")):
            for k in fc.keyframe_points:
                k.interpolation = "CONSTANT"

print(f"BUILD {time.time() - t0:.1f}s, {len(per)} frames")
if os.environ.get("DIAG"):
    for i in range(84, 108, 2):
        sc.frame_set(i)
        hb = bowler.pose.bones["RightHand"]
        hw = bowler.matrix_world @ hb.head
        print("DIAG f", i, "tau", per[i]["tau"], "strip", round(strip.strip_time, 1), "hand", tuple(round(x, 2) for x in hw), "ball", tuple(round(x, 2) for x in ball.matrix_world.translation), "hidden", ball.hide_render)
    sys.exit(0)

# ---------------------------------------------------------------- render
os.makedirs(out_dir, exist_ok=True)
if frame_sel and "-" in frame_sel:
    a, b = map(int, frame_sel.split("-"))
    frames = list(range(a, b + 1))
elif frame_sel:
    frames = [int(x) for x in frame_sel.split(",")]
else:
    frames = list(range(len(per)))
t1 = time.time()
for f in frames:
    target = os.path.join(out_dir, f"f_{f:05d}.png")
    if os.path.exists(target) and os.path.getsize(target) > 0:
        continue  # resume: frame already rendered
    sc.frame_set(f)
    sc.render.filepath = target
    bpy.ops.render.render(write_still=True)
print(f"RENDER {len(frames)} frames in {time.time() - t1:.1f}s")
