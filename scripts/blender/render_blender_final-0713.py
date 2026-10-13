"""
render_blender_final.py -- SSC 体素渲染 (Blender headless, MonoScene 规范)

依据来源 (全部为顶会官方代码):
  - 相机外参: S4C (3DV 2024) point_utils.py read_calib() cam2velo
  - z_flip:   S4C export_camera_trj.py
  - 渲染规范: MonoScene (CVPR 2022) kitti_vis_pred.py
      白底 bgcolor=(1,1,1) / cube / scale=voxel*0.95 / 无光照(自发光)
  - 配色:     你的 gen_voxelgrid_npy.py classes_to_colors (已验证100%命中)

用法:
  # 单帧
  ~/blender-4.2.0-linux-x64/blender --background --python render_blender_final.py -- \
      --ply out/voxel_outputs/51/000200.ply --out out/rendered/000200.png

  # 批量
  ~/blender-4.2.0-linux-x64/blender --background --python render_blender_final.py -- \
      --ply_dir out/voxel_outputs/51/ --out_dir out/rendered/
"""
import bpy
import sys
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector

# ============ 无需外参: 直接用 PLY 坐标系设相机 (见下方 setup_camera) ============



def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {"ply": None, "out": None, "ply_dir": None, "out_dir": None}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k in args and i + 1 < len(argv):
            args[k] = argv[i + 1]; i += 2
        else:
            i += 1
    return args


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in bpy.data.meshes: bpy.data.meshes.remove(m)
    for m in bpy.data.materials: bpy.data.materials.remove(m)
    for l in bpy.data.lights: bpy.data.lights.remove(l)


def import_ply(p):
    bpy.ops.wm.ply_import(filepath=str(p))
    return bpy.context.selected_objects[0]


def setup_emission_material(obj):
    """带光照的哑光材质: 顶点色 + Diffuse, 使体素有立体积木感 (对标S4C)"""
    mat = bpy.data.materials.new("VoxLit")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    vcol = nt.nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "Col"
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9   # 哑光
    bsdf.inputs["Metallic"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _look_at_matrix(eye, target, up=Vector((0, 0, 1))):
    """构造 look-at 相机矩阵 (Blender 相机光轴 -Z)"""
    eye = Vector(eye); target = Vector(target)
    forward = (target - eye).normalized()
    right = forward.cross(up).normalized()
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    true_up = right.cross(forward).normalized()
    return Matrix((
        (right.x, true_up.x, -forward.x, eye.x),
        (right.y, true_up.y, -forward.y, eye.y),
        (right.z, true_up.z, -forward.z, eye.z),
        (0, 0, 0, 1),
    ))


# ============ 用户在 Blender GUI 里亲手调好的相机视角 (固化) ============
# 通过 Ctrl+Alt+0 对齐视口后, 从 matrix_world 导出
USER_CAM_MATRIX = [
    [-9.71782804e-02, -1.71126246e-01,  9.80444908e-01,  3.93874969e+01],
    [ 9.95211959e-01, -6.34857826e-03,  9.75338593e-02,  2.96825085e+01],
    [-1.04661733e-02,  9.85228658e-01,  1.70923829e-01,  8.28792286e+00],
    [ 0.0,             0.0,             0.0,             1.0],
]

# 沿视线方向拉近，不改变朝向；正值=拉近，单位米
DOLLY = 12.0
import numpy as np
_M = np.array(USER_CAM_MATRIX, dtype=float)
_M[:3, 3] += -_M[:3, 2] * DOLLY
USER_CAM_MATRIX = _M.tolist()

def setup_camera():
    """使用用户在 Blender 里亲手调好的视角 (固化的 matrix_world)"""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 552.554261 / 1408 * cam_data.sensor_width
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.matrix_world = Matrix(USER_CAM_MATRIX)
    bpy.context.scene.camera = cam
    print(f"  相机(用户视角) 位置={[round(USER_CAM_MATRIX[i][3],1) for i in range(3)]}")
    return cam


def setup_lighting():
    """方向光: 让体素立方体三个面明暗不同 (立体感, 对标S4C)"""
    import math
    # 主方向光 (太阳), 从斜上方打
    sun_data = bpy.data.lights.new("Sun", type='SUN')
    sun_data.energy = 2.5
    sun_data.angle = math.radians(3)   # 硬一点的阴影, 边界清晰
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun)
    # 从右上前方照射 (让顶面最亮, 一个侧面中等, 另一侧暗)
    sun.rotation_euler = (math.radians(50), math.radians(15), math.radians(30))
    # 环境光: 提亮暗部, 避免死黑 (但保留明暗差)
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Strength"].default_value = 0.5  # 环境光适中


def setup_render(out_path, w=2400, h=1350):
    sc = bpy.context.scene
    try:
        sc.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        sc.render.engine = 'BLENDER_EEVEE'
    # 白色背景
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 0.5
    sc.render.film_transparent = False
    sc.render.resolution_x = w        # 提高分辨率 2400x1350
    sc.render.resolution_y = h
    sc.render.image_settings.file_format = 'PNG'
    sc.render.filepath = str(out_path)
    try:
        sc.eevee.taa_render_samples = 128   # 提高采样, 更清晰
    except Exception:
        pass


def render_one(ply, out):
    print(f"[渲染] {ply}")
    clear_scene()
    obj = import_ply(ply)
    print(f"  {len(obj.data.vertices)} 顶点")
    setup_emission_material(obj)
    setup_camera()
    setup_render(out)
    setup_lighting()   # 材质和世界设好后再加光
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    print(f"[✓] {out}")


def main():
    a = parse_args()
    if a["ply"] and a["out"]:
        render_one(a["ply"], a["out"])
    elif a["ply_dir"] and a["out_dir"]:
        pd = Path(a["ply_dir"]); od = Path(a["out_dir"])
        od.mkdir(parents=True, exist_ok=True)
        for ply in sorted(pd.glob("*.ply")):
            render_one(ply, od / (ply.stem + ".png"))
    else:
        print("用法: --ply x.ply --out x.png  或  --ply_dir d/ --out_dir d/")


if __name__ == "__main__":
    main()
