"""Quality/speed probe: stadium + bowler at release, rendered from behind the bowler.

blender -b -P blender/test_still.py -- <out.png> [frame_offset_seconds] [camera]
"""

import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stadium  # noqa: E402
import cricketer  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = args[0] if args else os.path.join(HERE, "..", "out", "edit_review", "blender", "still.png")
offset = float(args[1]) if len(args) > 1 else 0.0
cam_name = args[2] if len(args) > 2 else "broadcast"

VIDEO = os.path.abspath(os.path.join(HERE, ".."))
cricketer.TEX_DIR = os.path.join(VIDEO, "public", "models", "bowler")

t0 = time.time()
stadium.clear()
stadium.render_settings(1920, 1080, fps=30, samples=48)
stadium.build(os.path.join(HERE, "assets", "moonless_golf_2k.hdr"))
try:
    stadium.compositor()
except Exception as exc:
    print('COMPOSITOR', exc)
arm, _ = cricketer.import_cricketer(os.path.join(cricketer.TEX_DIR, "CricketBowlingAnimations.fbx"), "Bowler", {"top": None, "bottom": (0.0, 0.32, 0.02, 1)})
acts = {a.name: a for a in bpy.data.actions}
fast = next(a for n, a in acts.items() if n.endswith("RightArmFastBowler"))
span = fast.frame_range[1] - fast.frame_range[0]
fps = span / 7.58
rel, scale = cricketer.place_bowler(arm, fast, fps)
sc0 = bpy.context.scene
sc0.frame_set(int(rel))
hand = cricketer.bone_world(arm, "RightHand", rel)
meshes = [o for o in bpy.data.objects if o.name.startswith("Bowler_")]
print("DIAG arm", tuple(round(v, 2) for v in arm.matrix_world.translation), "hand", tuple(round(v, 2) for v in hand),
      "meshes", len(meshes), [(o.name, o.parent.name if o.parent else None, [m.type for m in o.modifiers]) for o in meshes[:3]])
import mathutils
bb = [o.matrix_world @ mathutils.Vector(c) for o in meshes[:1] for c in o.bound_box]
print("DIAG mesh bbox", [tuple(round(x, 1) for x in v) for v in bb[:2]])
print(f"BUILD {time.time() - t0:.1f}s  clip fps {fps:.1f}  release frame {rel:.1f}  scale {scale:.3f}")

# Sun as a moon/key fill for crisp shadows
bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
sun = bpy.context.active_object
sun.data.energy = 0.6
sun.rotation_euler = (0.9, 0.2, 0.6)

cams = {
    "broadcast": ((0.9, -14.0, 4.2), (0.0, 8.0, 1.1), 85),
    "front": ((-1.2, 9.0, 1.7), (-0.3, -1.2, 1.6), 70),
    "side": ((-9.0, -1.0, 1.6), (0.0, -1.0, 1.3), 50),
    "wide": ((24.0, -30.0, 14.0), (0.0, 10.0, 0.0), 35),
}
loc, target, lens = cams[cam_name]
bpy.ops.object.camera_add(location=loc)
cam = bpy.context.active_object
cam.data.lens = lens
cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
cam.data.dof.use_dof = True
cam.data.dof.focus_distance = (Vector(target) - Vector(loc)).length
cam.data.dof.aperture_fstop = 1.8
bpy.context.scene.camera = cam

frame = rel + offset * fps
sc = bpy.context.scene
sc.frame_set(int(frame), subframe=frame - int(frame))
sc.render.filepath = os.path.abspath(out)
os.makedirs(os.path.dirname(sc.render.filepath), exist_ok=True)
t1 = time.time()
bpy.ops.render.render(write_still=True)
print(f"RENDER {time.time() - t1:.1f}s -> {sc.render.filepath}")
