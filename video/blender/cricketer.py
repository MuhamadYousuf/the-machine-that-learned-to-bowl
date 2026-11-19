"""Import the cricketer FBX into Blender, rebuild his materials, and align the bowler with the simulator."""

import math
import os

import bpy
from mathutils import Matrix, Vector

TEX_DIR = None  # set by caller
SIM_RELEASE_B = Vector((-0.30, -1.22, 2.20))
BOWLER_SCALE = 1.054  # sim release (x=-1.22, y=0.30 off side, z=2.20) in Blender coords
RELEASE_S = 6.37  # seconds into RightArmFastBowler (as imported in Blender): arm ~40° past vertical, hand still ~2 m high at peak speed


def _img(name):
    p = os.path.join(TEX_DIR, name)
    img = bpy.data.images.load(p, check_existing=True)
    if "Normal" in name:
        img.colorspace_settings.name = "Non-Color"
    return img


def _material(name, base_tex=None, color=None, normal_tex=None, rough=0.65, sheen=0.0, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    b = n["Principled BSDF"]
    if base_tex:
        t = n.new("ShaderNodeTexImage"); t.image = _img(base_tex)
        l.new(t.outputs["Color"], b.inputs["Base Color"])
    if color:
        b.inputs["Base Color"].default_value = color
    if normal_tex:
        t = n.new("ShaderNodeTexImage"); t.image = _img(normal_tex)
        nm = n.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.8
        l.new(t.outputs["Color"], nm.inputs["Color"]); l.new(nm.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = rough
    if sheen and "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    if sss and "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (0.9, 0.35, 0.2)
        if "Subsurface Scale" in b.inputs:
            b.inputs["Subsurface Scale"].default_value = 0.01
    return m


def kit_materials(kit):
    """kit: dict(top=(r,g,b,a) or None for the textured green shirt, bottom=(r,g,b,a))."""
    skin = _material("Skin", "SkinDiffuse.png", rough=0.5, sss=0.25)
    body = _material("BodySkin", "SkinDiffuse.png", normal_tex="BodyNormal.png", rough=0.5, sss=0.25)
    top = _material("Top", "OutfitTopDifffuse.png", normal_tex="OutfitTopNormal.png", rough=0.75, sheen=0.4) if kit.get("top") is None \
        else _material("Top", color=kit["top"], normal_tex="OutfitTopNormal.png", rough=0.75, sheen=0.4)
    bottom = _material("Bottom", color=kit["bottom"], normal_tex="OutfitBottomNormal.png", rough=0.75, sheen=0.4)
    shoes = _material("Shoes", "FootWearDiffuse.png", normal_tex="FootWearNormal.png", rough=0.55)
    hair = _material("Hair", color=(0.012, 0.01, 0.009, 1), normal_tex="HairNormal.png", rough=0.7)
    eye = _material("Eye", "EyeDiffuse.png", rough=0.1)
    teeth = _material("Teeth", "TeethDiffuse.png", rough=0.3)
    return {"Body": body, "Head": skin, "OutfitTop": top, "OufitTop": top, "OutfitBottom": bottom, "OufitBottom": bottom,
            "FootWear": shoes, "Hair": hair, "Beard": hair, "Eye": eye, "Teeth": teeth}


def import_cricketer(fbx_path, name, kit):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=fbx_path, use_anim=True, automatic_bone_orientation=False)
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    arm.name = name
    mats = kit_materials(kit)
    for o in new:
        if o.type == "MESH":
            key = next((k for k in mats if o.name.startswith(k)), None)
            o.data.materials.clear()
            o.data.materials.append(mats[key] if key else mats["Body"])
            o.name = f"{name}_{o.name}"
    return arm, new


def actions_for(arm):
    return {a.name.split("|")[-1]: a for a in bpy.data.actions if a.name.startswith("Armature|") or "|" in a.name}


def bone_world(arm, bone, frame):
    bpy.context.scene.frame_set(int(frame), subframe=frame - int(frame))
    return arm.matrix_world @ arm.pose.bones[bone].head


def set_action(arm, action):
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action
    if hasattr(arm.animation_data, "action_slot") and action.slots:
        arm.animation_data.action_slot = action.slots[0]


def rig_root(arm):
    """An Empty above the (animated) armature object, so placement isn't overwritten by the clip's root motion."""
    name = f"{arm.name}_Root"
    root = bpy.data.objects.get(name)
    if root is None:
        root = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(root)
        arm.parent = root
    return root


def place_bowler(arm, action, fps):
    """Rotate/scale/translate the bowler so the clip's release hand lands on the sim release point, running along +Y."""
    set_action(arm, action)
    root = rig_root(arm)
    f0 = action.frame_range[0]
    rel = f0 + RELEASE_S * fps
    root.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    h0 = bone_world(arm, "Hips", f0 + 1.0 * fps)
    h1 = bone_world(arm, "Hips", rel - 0.5 * fps)
    run = h1 - h0
    run.z = 0
    yaw = math.atan2(run.x, run.y)  # rotation about Z that turns the run-up onto +Y
    scale = BOWLER_SCALE  # same 1.07x as the browser preview (a 1.98 m fast bowler); height is not forced to the sim's 2.2 m
    root.matrix_world = Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(scale, 4)
    bpy.context.view_layer.update()
    hand = bone_world(arm, "RightHand", rel)
    shift = SIM_RELEASE_B - hand
    shift.z = 0.0  # align along the ground only; the ball blends from the real hand into the sim trajectory
    root.matrix_world = Matrix.Translation(shift) @ root.matrix_world
    bpy.context.view_layer.update()
    return rel, scale
