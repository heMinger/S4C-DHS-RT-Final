"""
Render SSC voxel PLY files from a front-facing KITTI-style camera.

This script is intended for pairing a 3D SSC visualization with its monocular
input image in an overview figure.  It keeps the same vertex colors, matte
material, and lighting as render_blender_final.py, but uses a lower,
front-facing camera and a KITTI-like wide canvas.

Single frame (recommended architecture view):
  ~/blender-4.2.0-linux-x64/blender --background \
      --python scripts/blender/render_blender_front.py -- \
      --ply out/voxel_outputs_vis/dhs_rt_seq0006_frame0146/51/000146.ply \
      --out out/voxel_outputs_vis/dhs_rt_seq0006_frame0146/rendered/000146_front.png \
      --view ego_oblique --transparent

Directory:
  ~/blender-4.2.0-linux-x64/blender --background \
      --python scripts/blender/render_blender_front.py -- \
      --ply_dir out/voxel_outputs_vis/example/51 \
      --out_dir out/voxel_outputs_vis/example/rendered_front
"""

import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# The exported SSC PLY uses x as the lateral axis, y as the forward/depth
# axis, and z as height.  The volume is approximately:
#   x in [-25.6, 25.6], y in [0, 51.2], z in [0, 6.4].
# Hence the input camera is close to y=0 and looks towards +y.  Keeping the
# render camera at y<0 preserves the input image's near/far ordering.
#
# ego_oblique is recommended for the architecture figure: it stays on the
# input side but adds a modest lateral/elevation offset so the 3D completion
# remains readable.  ego_front is closer to the actual monocular viewpoint.
CAMERA_PRESETS = {
    "ego_oblique": {
        "eye": (10.0, -12.0, 9.0),
        "target": (0.0, 18.0, 1.8),
        "lens": 27.0,
    },
    "ego_front": {
        "eye": (0.0, -8.0, 3.2),
        "target": (0.0, 22.0, 1.7),
        "lens": 20.0,
    },
}
CAMERA_UP = (0.0, 0.0, 1.0)

# A normal landscape canvas is easier to crop and place in an architecture
# diagram.  Transparent margins can be removed in the figure editor.
RENDER_WIDTH = 2400
RENDER_HEIGHT = 1350


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {
        "ply": None,
        "out": None,
        "ply_dir": None,
        "out_dir": None,
        "view": "ego_oblique",
        "transparent": False,
    }

    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key == "transparent":
            args[key] = True
            i += 1
        elif key in args and i + 1 < len(argv):
            args[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)


def import_ply(path):
    bpy.ops.wm.ply_import(filepath=str(path))
    selected = bpy.context.selected_objects
    if not selected:
        raise RuntimeError(f"Blender imported no object from {path}")
    return selected[0]


def setup_voxel_material(obj):
    """Use PLY vertex colors with a matte material and visible cube shading."""
    material = bpy.data.materials.new("VoxelMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = "Col"

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.9
    principled.inputs["Metallic"].default_value = 0.0

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(material)


def look_at_matrix(eye, target, up):
    """Return a Blender camera matrix; Blender cameras look along local -Z."""
    eye = Vector(eye)
    target = Vector(target)
    up = Vector(up)

    forward = (target - eye).normalized()
    right = forward.cross(up).normalized()
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    true_up = right.cross(forward).normalized()

    return Matrix(
        (
            (right.x, true_up.x, -forward.x, eye.x),
            (right.y, true_up.y, -forward.y, eye.y),
            (right.z, true_up.z, -forward.z, eye.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def setup_camera(view):
    if view not in CAMERA_PRESETS:
        choices = ", ".join(sorted(CAMERA_PRESETS))
        raise ValueError(f"Unknown view '{view}'. Choose one of: {choices}")

    preset = CAMERA_PRESETS[view]
    eye = preset["eye"]
    target = preset["target"]

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = preset["lens"]
    camera_data.clip_start = 0.1
    camera_data.clip_end = 200.0

    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.matrix_world = look_at_matrix(
        eye,
        target,
        CAMERA_UP,
    )
    bpy.context.scene.camera = camera

    print(
        f"  camera view={view}, eye={eye}, target={target}, "
        f"lens={preset['lens']}mm"
    )
    return camera


def setup_lighting():
    """Directional light makes individual voxel faces visually separable."""
    import math

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(3.0)

    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (
        math.radians(50.0),
        math.radians(15.0),
        math.radians(30.0),
    )


def setup_render(output_path, transparent=False):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = 1.0

    # Standard view transform preserves a genuinely white paper background.
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass

    scene.render.film_transparent = transparent
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.filepath = str(output_path)

    try:
        scene.eevee.taa_render_samples = 128
    except Exception:
        pass


def render_one(ply_path, output_path, view="ego_oblique", transparent=False):
    ply_path = Path(ply_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render] {ply_path}")
    clear_scene()
    obj = import_ply(ply_path)
    print(f"  vertices={len(obj.data.vertices)}")

    setup_voxel_material(obj)
    setup_camera(view)
    setup_render(output_path, transparent=transparent)
    setup_lighting()

    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    print(f"[saved] {output_path}")


def main():
    args = parse_args()

    if args["ply"] and args["out"]:
        render_one(
            args["ply"],
            args["out"],
            view=args["view"],
            transparent=args["transparent"],
        )
        return

    if args["ply_dir"] and args["out_dir"]:
        ply_dir = Path(args["ply_dir"]).expanduser()
        output_dir = Path(args["out_dir"]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        ply_files = sorted(ply_dir.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {ply_dir}")

        for ply_path in ply_files:
            render_one(
                ply_path,
                output_dir / f"{ply_path.stem}.png",
                view=args["view"],
                transparent=args["transparent"],
            )
        return

    raise ValueError(
        "Usage: --ply input.ply --out output.png "
        "[--view ego_oblique|ego_front] [--transparent], or "
        "--ply_dir input_dir --out_dir output_dir "
        "[--view ego_oblique|ego_front] [--transparent]"
    )


if __name__ == "__main__":
    main()
