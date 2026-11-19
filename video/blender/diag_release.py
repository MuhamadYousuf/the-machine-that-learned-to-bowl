import os, sys, bpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cricketer
cricketer.TEX_DIR = os.path.join(HERE, "..", "public", "models", "bowler")
bpy.ops.wm.read_factory_settings(use_empty=True)
arm, _ = cricketer.import_cricketer(os.path.join(cricketer.TEX_DIR, "CricketBowlingAnimations.fbx"), "Bowler", {"top": None, "bottom": (0, 0.3, 0, 1)})
fast = next(a for a in bpy.data.actions if a.name.endswith("RightArmFastBowler"))
cricketer.set_action(arm, fast)
f0 = fast.frame_range[0]; fps = (fast.frame_range[1] - f0) / 7.58
print("RANGE", fast.frame_range[:], "fps", fps)
sc = bpy.context.scene
for k in range(0, 40):
    t = 6.25 + k * 0.01
    f = f0 + t * fps
    sc.frame_set(int(f), subframe=f - int(f))
    h = arm.matrix_world @ arm.pose.bones["RightHand"].head
    s = arm.matrix_world @ arm.pose.bones["RightArm"].head
    hips = arm.matrix_world @ arm.pose.bones["Hips"].head
    run = (arm.matrix_world @ arm.pose.bones["Hips"].head)
    print(f"T {t:.2f} hand_z {h.z:.2f} hand-shoulder (fwd along run) {(h - s).dot(fwd) if (fwd:=None) else 0:.2f}" if False else f"T {t:.2f} hand {tuple(round(x,2) for x in h)} shoulder {tuple(round(x,2) for x in s)}")
