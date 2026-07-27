"""
check_ply_semantic.py -- 验证 SSC PLY 是否为正确的带语义场景表示

检查三个维度:
  1. 几何: 坐标范围是否符合 SSCBench (X/Y/Z 跨度)
  2. 颜色: PLY 顶点色是否命中 S4C classes_to_colors 官方表
  3. 语义分布: 各类别体素占比是否合理 (road/building/vegetation 应为主要类别)

用法:
    python check_ply_semantic.py /path/to/000000.ply
    # 或用 Blender 自带 python:
    ~/blender-4.2.0-linux-x64/blender --background --python check_ply_semantic.py -- /path/to/000000.ply
"""
import sys
import numpy as np

# S4C 官方 classes_to_colors (index = 语义标签, 来自 gen_voxelgrid_npy.py)
CLASSES_TO_COLORS = {
    0:  ([255, 255, 255], "unlabeled/free"),
    1:  ([100, 150, 245], "car"),
    2:  ([255, 0, 0],     "bicycle"),
    3:  ([255, 0, 255],   "motorcycle"),
    4:  ([255, 150, 255], "truck"),
    5:  ([75, 0, 75],     "other-vehicle"),
    6:  ([175, 0, 75],    "person"),
    7:  ([255, 200, 0],   "road"),
    8:  ([150, 150, 150], "sidewalk"),
    9:  ([30, 60, 150],   "building"),
    10: ([80, 30, 180],   "fence"),
    11: ([8, 97, 0],      "vegetation"),
    12: ([184, 56, 2],    "terrain"),
    13: ([255, 143, 46],  "pole"),
    14: ([112, 255, 50],  "traffic-sign"),
    15: ([194, 0, 0],     "other-object"),
}


def parse_ply_path():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1]
    return argv[-1]


def read_ply_manual(path):
    """轻量读取 PLY 顶点坐标和颜色 (不依赖 plyfile)"""
    with open(path, "rb") as f:
        # 读 header
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        # 解析顶点数和属性
        n_verts = 0
        props = []
        in_vertex = False
        fmt = "ascii"
        for h in header:
            if h.startswith("format"):
                fmt = h.split()[1]
            if h.startswith("element vertex"):
                n_verts = int(h.split()[-1])
                in_vertex = True
            elif h.startswith("element"):
                in_vertex = False
            elif h.startswith("property") and in_vertex:
                props.append(h.split()[-1])

        has_color = "red" in props
        if fmt == "ascii":
            verts = []
            colors = []
            for _ in range(n_verts):
                parts = f.readline().split()
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_color:
                    ci = props.index("red")
                    colors.append([int(parts[ci]), int(parts[ci+1]), int(parts[ci+2])])
            return np.array(verts), (np.array(colors) if has_color else None)
        else:
            # binary_little_endian: x,y,z float32 + red,green,blue uint8
            import struct
            verts, colors = [], []
            for _ in range(n_verts):
                x = struct.unpack("<f", f.read(4))[0]
                y = struct.unpack("<f", f.read(4))[0]
                z = struct.unpack("<f", f.read(4))[0]
                verts.append([x, y, z])
                if has_color:
                    r = struct.unpack("<B", f.read(1))[0]
                    g = struct.unpack("<B", f.read(1))[0]
                    b = struct.unpack("<B", f.read(1))[0]
                    colors.append([r, g, b])
            return np.array(verts), (np.array(colors) if has_color else None)


def main():
    path = parse_ply_path()
    print(f"检查文件: {path}\n")

    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"]
        verts = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1)
        if "red" in v.data.dtype.names:
            colors = np.stack([np.array(v["red"]), np.array(v["green"]), np.array(v["blue"])], axis=1)
        else:
            colors = None
    except Exception:
        verts, colors = read_ply_manual(path)

    # ========== 检查1: 几何 ==========
    print("=" * 55)
    print("检查1: 几何范围 (SSCBench-KITTI-360 标准)")
    print("=" * 55)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    print(f"  X: [{x.min():.1f}, {x.max():.1f}]  跨度 {x.max()-x.min():.1f}  (标准≈51.2 左右)")
    print(f"  Y: [{y.min():.1f}, {y.max():.1f}]  跨度 {y.max()-y.min():.1f}  (标准≈51.2 深度)")
    print(f"  Z: [{z.min():.1f}, {z.max():.1f}]  跨度 {z.max()-z.min():.1f}  (标准≈6.4 高度)")
    print(f"  顶点总数: {len(verts)}")
    geo_ok = abs((z.max()-z.min()) - 6.4) < 2.0
    print(f"  几何判断: {'✓ 高度跨度符合' if geo_ok else '✗ 高度异常, 检查坐标系'}")

    # ========== 检查2: 颜色命中官方表 ==========
    print("\n" + "=" * 55)
    print("检查2: 顶点色是否命中 S4C 官方 classes_to_colors")
    print("=" * 55)
    if colors is None:
        print("  ✗ PLY 无颜色信息! 不是语义 PLY (可能是纯几何/高度着色)")
        return

    # 统计唯一颜色
    uniq, counts = np.unique(colors, axis=0, return_counts=True)
    known = {tuple(c): (lbl, name) for lbl, (c, name) in CLASSES_TO_COLORS.items()}
    total = len(colors)
    matched = 0
    print(f"  发现 {len(uniq)} 种唯一颜色:")
    # 按占比排序
    order = np.argsort(-counts)
    class_hist = {}
    for i in order:
        c = tuple(int(v) for v in uniq[i])
        pct = 100.0 * counts[i] / total
        if c in known:
            lbl, name = known[c]
            matched += counts[i]
            class_hist[name] = class_hist.get(name, 0) + pct
            print(f"    RGB{c}  {pct:5.1f}%  -> [{lbl:2d}] {name}  ✓")
        else:
            print(f"    RGB{c}  {pct:5.1f}%  -> 未知颜色 ✗ (不在官方表)")
    match_pct = 100.0 * matched / total
    print(f"\n  颜色命中率: {match_pct:.1f}%")
    print(f"  判断: {'✓ 配色正确 (命中官方表)' if match_pct > 95 else '✗ 存在非标准颜色, 配色可能错误'}")

    # ========== 检查3: 语义分布合理性 ==========
    print("\n" + "=" * 55)
    print("检查3: 语义类别分布合理性")
    print("=" * 55)
    print("  (户外驾驶场景: road/building/vegetation 应为主要类别)")
    for name in sorted(class_hist, key=lambda k: -class_hist[k]):
        # 顶点占比 (每体素8顶点, 占比近似体素占比)
        bar = "#" * int(class_hist[name] / 2)
        print(f"    {name:15s} {class_hist[name]:5.1f}%  {bar}")

    major = {"road", "building", "vegetation", "sidewalk", "terrain"}
    major_pct = sum(class_hist.get(m, 0) for m in major)
    print(f"\n  主要类别(road/building/veg/sidewalk/terrain)合计: {major_pct:.1f}%")
    print(f"  判断: {'✓ 分布合理' if major_pct > 60 else '⚠ 主要类别偏少, 需人工核对'}")

    print("\n" + "=" * 55)
    print("总结")
    print("=" * 55)
    print(f"  几何: {'✓' if geo_ok else '✗'}   颜色命中: {'✓' if match_pct>95 else '✗'} ({match_pct:.0f}%)   "
          f"语义分布: {'✓' if major_pct>60 else '⚠'}")


if __name__ == "__main__":
    main()
