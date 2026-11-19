"""Blender stadium for the documentary's 3D player shots (run inside Blender).

Coordinates: Blender X = -sim y (off side is -X), Blender Y = sim x (along the pitch, bowler's stumps at 0,
batter's at 20.12), Blender Z = sim z (up). Units are metres.

Everything is procedural except the night HDRI (Poly Haven, CC0) and the cricketer (Unity Asset Store pack,
not redistributable, kept in video/public/models/ which is git-ignored).
"""

import math
import random

import bpy
from mathutils import Matrix, Vector

L = 20.12
ROOT = bpy.path.abspath("//") if bpy.data.filepath else None


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, base=(0.8, 0.8, 0.8, 1), rough=0.6, metal=0.0, emit=None, emit_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = base
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = emit
        b.inputs["Emission Strength"].default_value = emit_strength
    return m


def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes["Principled BSDF"]


def obj(mesh_op, name, material=None, **kw):
    mesh_op(**kw)
    o = bpy.context.active_object
    o.name = name
    if material:
        o.data.materials.append(material)
    return o


# ---------------------------------------------------------------- world, render

def world(hdri_path, strength=0.35):
    w = bpy.data.worlds.new("Night")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    mapping = nt.nodes.new("ShaderNodeMapping")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # Floodlight haze
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Density"].default_value = 0.0004
    vol.inputs["Anisotropy"].default_value = 0.6
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])


def render_settings(width, height, fps=30, engine="BLENDER_EEVEE", samples=64):
    s = bpy.context.scene
    s.render.engine = engine
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.fps = fps
    s.render.use_motion_blur = True
    s.render.motion_blur_shutter = 0.5
    s.view_settings.view_transform = "AgX"
    try:
        s.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    e = s.eevee
    for attr, val in (("taa_render_samples", samples), ("use_raytracing", True), ("use_shadows", True),
                      ("volumetric_tile_size", "4"), ("volumetric_end", 140.0), ("use_volumetric_shadows", True)):
        if hasattr(e, attr):
            try:
                setattr(e, attr, val)
            except Exception:
                pass
    s.render.film_transparent = False
    s.render.image_settings.file_format = "PNG"


# ---------------------------------------------------------------- field

def field():
    m, n, l, b = node_mat("Grass")
    tex = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    l.new(tex.outputs["Object"], sep.inputs["Vector"])
    # mowing stripes along the pitch: bands 5 m wide across Y
    wave = n.new("ShaderNodeMath"); wave.operation = "SINE"
    scale = n.new("ShaderNodeMath"); scale.operation = "MULTIPLY"; scale.inputs[1].default_value = math.pi / 5.0
    l.new(sep.outputs["Y"], scale.inputs[0]); l.new(scale.outputs[0], wave.inputs[0])
    step = n.new("ShaderNodeMath"); step.operation = "GREATER_THAN"; step.inputs[1].default_value = 0.0
    l.new(wave.outputs[0], step.inputs[0])
    noise = n.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 0.6; noise.inputs["Detail"].default_value = 6
    l.new(tex.outputs["Object"], noise.inputs["Vector"])
    fine = n.new("ShaderNodeTexNoise"); fine.inputs["Scale"].default_value = 90; fine.inputs["Detail"].default_value = 2
    l.new(tex.outputs["Object"], fine.inputs["Vector"])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.11, 0.035, 1)
    ramp.color_ramp.elements[1].color = (0.075, 0.2, 0.06, 1)
    mix = n.new("ShaderNodeMath"); mix.operation = "MULTIPLY_ADD"
    l.new(step.outputs[0], mix.inputs[0]); mix.inputs[1].default_value = 0.45
    l.new(noise.outputs["Fac"], mix.inputs[2])
    l.new(mix.outputs[0], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.95
    bump = n.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    l.new(fine.outputs["Fac"], bump.inputs["Height"]); l.new(bump.outputs["Normal"], b.inputs["Normal"])
    obj(bpy.ops.mesh.primitive_circle_add, "Field", m, vertices=128, radius=78, fill_type="NGON", location=(0, L / 2, 0))


def pitch(texture_path=None):
    m, n, l, b = node_mat("Pitch")
    tc = n.new("ShaderNodeTexCoord")
    noise = n.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 3.0; noise.inputs["Detail"].default_value = 10
    l.new(tc.outputs["Object"], noise.inputs["Vector"])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.36, 0.29, 0.18, 1)
    ramp.color_ramp.elements[1].color = (0.62, 0.53, 0.37, 1)
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.92
    crack = n.new("ShaderNodeTexVoronoi"); crack.feature = "DISTANCE_TO_EDGE"; crack.inputs["Scale"].default_value = 14
    l.new(tc.outputs["Object"], crack.inputs["Vector"])
    bump = n.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
    l.new(crack.outputs["Distance"], bump.inputs["Height"]); l.new(bump.outputs["Normal"], b.inputs["Normal"])
    p = obj(bpy.ops.mesh.primitive_plane_add, "Pitch", m, size=1, location=(0, L / 2, 0.004))
    p.scale = (3.05, L + 4.0, 1)
    paint = mat("Paint", (0.95, 0.95, 0.93, 1), 0.5)
    def line(name, x, y, w, d):
        o = obj(bpy.ops.mesh.primitive_plane_add, name, paint, size=1, location=(x, y, 0.008))
        o.scale = (w, d, 1)
    for end, sgn in ((0.0, -1), (L, 1)):
        line(f"Bowling{end}", 0, end, 2.64, 0.05)
        line(f"Popping{end}", 0, end + sgn * 1.22, 3.66, 0.05)
        for x in (-1.32, 1.32):
            line(f"Return{end}{x}", x, end + sgn * 0.4, 0.05, 2.44)
    stumps(0.0, "BowlerStumps")
    stumps(L, "BatterStumps")


def stumps(y, name):
    wood = mat("Stump", (0.85, 0.78, 0.62, 1), 0.45)
    group = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(group)
    parts = []
    for i, x in enumerate((-0.114, 0.0, 0.114)):
        o = obj(bpy.ops.mesh.primitive_cylinder_add, f"{name}_s{i}", wood, radius=0.019, depth=0.711, vertices=24, location=(x, y, 0.711 / 2))
        # pivot at the base (where a stump is anchored in the ground), so a knocked stump tips over instead of spinning
        o.data.transform(Matrix.Translation((0, 0, 0.711 / 2)))
        o.location.z = 0.0
        parts.append(o)
    for i, x in enumerate((-0.057, 0.057)):
        o = obj(bpy.ops.mesh.primitive_cylinder_add, f"{name}_b{i}", wood, radius=0.008, depth=0.11, vertices=12, location=(x, y, 0.725), rotation=(0, math.pi / 2, 0))
        parts.append(o)
    for o in parts:
        for c in o.users_collection:
            c.objects.unlink(o)
        group.objects.link(o)
    return group


# ---------------------------------------------------------------- stadium

def boundary():
    rope = mat("Rope", (0.9, 0.9, 0.95, 1), 0.4)
    obj(bpy.ops.mesh.primitive_torus_add, "Rope", rope, major_radius=66, minor_radius=0.07, location=(0, L / 2, 0.07), major_segments=256)
    # LED advertising boards: a ring of emissive panels with alternating colours
    colours = [(0.0, 0.45, 0.32, 1), (0.02, 0.12, 0.6, 1), (0.75, 0.42, 0.0, 1), (0.6, 0.02, 0.08, 1)]
    for i in range(72):
        a = i / 72 * 2 * math.pi
        x, y = math.sin(a) * 70, L / 2 + math.cos(a) * 70
        c = colours[i % len(colours)]
        m = mat(f"LED{i % 4}", (0.02, 0.02, 0.02, 1), 0.3, emit=c, emit_strength=0.9)
        o = obj(bpy.ops.mesh.primitive_cube_add, f"Board{i}", m, size=1, location=(x, y, 0.55))
        o.scale = (6.0, 0.15, 1.0)
        o.rotation_euler = (0, 0, -a)


def ring_mesh(name, rows, segments=192):
    """Stepped seating bowl: for each row an annulus (tread) at its height plus the riser below it."""
    import bmesh
    bm = bmesh.new()
    for (r_in, r_out, z, z_prev) in rows:
        loops = []
        for (r, h) in ((r_in, z_prev), (r_in, z), (r_out, z)):
            loop = [bm.verts.new((math.sin(2 * math.pi * i / segments) * r, L / 2 + math.cos(2 * math.pi * i / segments) * r, h)) for i in range(segments)]
            loops.append(loop)
        for a, b in ((loops[0], loops[1]), (loops[1], loops[2])):
            for i in range(segments):
                j = (i + 1) % segments
                bm.faces.new((a[i], a[j], b[j], b[i]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def stands():
    concrete = mat("Concrete", (0.11, 0.115, 0.13, 1), 0.85)
    seat_rows = 34
    r0, rise, depth = 74.0, 0.55, 0.9
    rows = []
    for k in range(seat_rows):
        r = r0 + k * depth
        z = rise * (k + 1) + 0.4
        rows.append((r, r + depth, z, z - rise if k else 0.0))
    bowl = ring_mesh("Stands", rows)
    bowl.data.materials.append(concrete)
    # Roof canopy: a sloped ring over the top rows
    roof = mat("Roof", (0.06, 0.065, 0.075, 1), 0.5, metal=0.6)
    top = r0 + seat_rows * depth
    canopy = ring_mesh("Roof", [(top - 14, top + 2, seat_rows * rise + 12, seat_rows * rise + 11)])
    canopy.data.materials.append(roof)
    crowd(r0, rise, depth, seat_rows)


def crowd(r0, rise, depth, rows, density=0.55, seed=4):
    """Thousands of spectators: a few person meshes instanced on every step with random shirt colours."""
    random.seed(seed)
    m, n, l, b = node_mat("Crowd")
    info = n.new("ShaderNodeObjectInfo")
    ramp = n.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    palette = [(0.02, 0.35, 0.12), (0.85, 0.85, 0.85), (0.08, 0.2, 0.55), (0.75, 0.6, 0.1), (0.6, 0.08, 0.1), (0.05, 0.05, 0.06), (0.12, 0.5, 0.18)]
    els[0].color = (*palette[0], 1)
    els[1].color = (*palette[1], 1)
    for i, c in enumerate(palette[2:], start=2):
        e = els.new(i / (len(palette) - 1))
        e.color = (*c, 1)
    ramp.color_ramp.interpolation = "CONSTANT"
    l.new(info.outputs["Random"], ramp.inputs["Fac"])
    l.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    # One person: torso + head, merged.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.55, vertices=8, location=(0, 0, 0.28))
    torso = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, segments=8, ring_count=6, location=(0, 0, 0.67))
    head = bpy.context.active_object
    head.select_set(True)
    torso.select_set(True)
    bpy.context.view_layer.objects.active = torso
    bpy.ops.object.join()
    person = torso
    person.name = "Person"
    person.data.materials.append(m)
    coll = bpy.data.collections.new("Crowd")
    bpy.context.scene.collection.children.link(coll)
    for c in person.users_collection:
        c.objects.unlink(person)
    coll.objects.link(person)
    count = 0
    for k in range(rows):
        r = r0 + (k + 0.5) * depth
        z = rise * (k + 1) + 0.4
        steps = int(2 * math.pi * r / 0.6)
        for i in range(steps):
            if random.random() > density:
                continue
            a = (i + random.random() * 0.3) / steps * 2 * math.pi
            inst = person.copy()
            inst.location = (math.sin(a) * r, L / 2 + math.cos(a) * r, z)
            inst.rotation_euler = (0, 0, -a + math.pi + random.uniform(-0.3, 0.3))
            s = random.uniform(0.9, 1.1)
            inst.scale = (s, s, s)
            coll.objects.link(inst)
            count += 1
    person.hide_render = True
    person.hide_viewport = True
    return count


def floodlights():
    steel = mat("Steel", (0.3, 0.32, 0.35, 1), 0.4, metal=0.8)
    lamp = mat("Lamp", (1, 1, 1, 1), 0.2, emit=(1.0, 0.95, 0.85, 1), emit_strength=60.0)
    for i, a in enumerate((math.radians(40), math.radians(140), math.radians(220), math.radians(320))):
        x, y = math.sin(a) * 96, L / 2 + math.cos(a) * 96
        obj(bpy.ops.mesh.primitive_cylinder_add, f"Tower{i}", steel, radius=0.7, depth=48, vertices=12, location=(x, y, 24))
        head = obj(bpy.ops.mesh.primitive_cube_add, f"Lamps{i}", lamp, size=1, location=(x * 0.985, L / 2 + (y - L / 2) * 0.985, 48))
        head.scale = (9, 0.6, 5)
        head.rotation_euler = (math.radians(-20), 0, -a)
        # The light the lamps actually throw onto the field
        bpy.ops.object.light_add(type="SPOT", location=(x * 0.97, L / 2 + (y - L / 2) * 0.97, 47))
        spot = bpy.context.active_object
        spot.data.energy = 120000
        spot.data.spot_size = math.radians(55)
        spot.data.spot_blend = 0.6
        spot.data.shadow_soft_size = 3.0
        spot.data.color = (1.0, 0.96, 0.9)
        direction = Vector((0, L / 2, 0)) - spot.location
        spot.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def sight_screens():
    white = mat("SightScreen", (0.012, 0.012, 0.014, 1), 0.9)
    for y in (-66.0, L + 66.0):
        o = obj(bpy.ops.mesh.primitive_cube_add, f"Screen{y}", white, size=1, location=(0, y, 5))
        o.scale = (22, 0.4, 10)


def build(hdri):
    world(hdri)
    field()
    pitch()
    boundary()
    stands()
    floodlights()
    sight_screens()


def compositor(glare=True):
    """Floodlight bloom + a soft vignette."""
    sc = bpy.context.scene
    sc.use_nodes = True
    tree = sc.compositing_node_group if hasattr(sc, "compositing_node_group") else sc.node_tree
    if tree is None:
        tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
        sc.compositing_node_group = tree
    n, l = tree.nodes, tree.links
    n.clear()
    rl = n.new("CompositorNodeRLayers")
    out = n.new("NodeGroupOutput") if hasattr(sc, "compositing_node_group") else n.new("CompositorNodeComposite")
    if hasattr(sc, "compositing_node_group") and not tree.interface.items_tree:
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    last = rl.outputs["Image"]
    if glare:
        g = n.new("CompositorNodeGlare")
        try:
            g.glare_type = "FOG_GLOW"
        except Exception:
            pass
        for key, val in (("threshold", 1.2), ("size", 7), ("mix", -0.75), ("quality", "HIGH")):
            if hasattr(g, key):
                try:
                    setattr(g, key, val)
                except Exception:
                    pass
        for name, val in (("Threshold", 1.2), ("Strength", 0.35), ("Size", 0.6)):
            if name in g.inputs:
                g.inputs[name].default_value = val
        l.new(last, g.inputs[0])
        last = g.outputs[0]
    l.new(last, out.inputs[0])
