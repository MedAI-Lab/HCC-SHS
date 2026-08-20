import json
import os
import glob
import numpy as np
import csv
from scipy.spatial import cKDTree
from scipy.stats import entropy

try:
    import openslide
except ImportError:
    openslide = None

def get_mpp_from_svs(svs_path):
    if openslide is None:
        raise ImportError("缺少 openslide-python。请先安装 openslide-python / openslide-bin。")
    
    slide = openslide.OpenSlide(svs_path)
    
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    
    if mpp_x is None or mpp_y is None:
        x_res = slide.properties.get('tiff.XResolution')
        y_res = slide.properties.get('tiff.YResolution')
        res_unit = slide.properties.get('tiff.ResolutionUnit')
        
        if x_res and y_res:
            mpp_x = 10000 / float(x_res)
            mpp_y = 10000 / float(y_res)
        else:
            mpp_x = 0.25
            mpp_y = 0.25
    
    slide.close()
    
    return (float(mpp_x) + float(mpp_y)) / 2  # 新增

# 输入输出路径和SVS路径
path_pairs = [
    (r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\乏氧json\filtered', r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\乏氧json\features', r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\CAIX_R'),
    #(r'M:\HCC\Cellpoint\patchwithratio\zhuhai_tumor-stroma_json', r'M:\HCC\Cellpoint\patchwithratio\zhuhai_tumor-stroma_json\features', r'M:\HCC\Cellpoint\patchwithratio\zhuhai_tumor-stroma_svs'),
    #(r'M:\HCC\Cellpoint\hunan\HE_Tcell_json', r'M:\HCC\Cellpoint\hunan\HE_Tcell_json\neigh_out', r'M:\HCC\Cellpoint\hunan\HE_Tcell_svs')
]

type_map = {
    "1": "neoplasm",
    "2": "inflam",
    "3": "stroma",
    "4": "necrosis",
    "5": "CD3",
    "6": "CD8",
    "7": "FOXP3",
    "8": "CD20",
    "9": "CD68",
    "10": "CD163",
    "11": "CD66B",
    "12": "CD57",
    "13": "PD-L1",
    "14": "CAIX"
}
radius_list = [10, 50]

# 扩展免疫细胞类型
immune_types = ["CD3", "CD8", "FOXP3", "CD20", "CD68", "CD163", "CD66B", "CD57", "PD-L1", "CAIX"]
immune_type_ids = {
    "CD3": "5", "CD8": "6", "FOXP3": "7", "CD20": "8",
    "CD68": "9", "CD163": "10", "CD66B": "11", "CD57": "12",
    "PD-L1": "13", "CAIX": "14"
}

def calc_entropy(arr, bins=10):
    if arr.size == 0:
        return 0
    hist, _ = np.histogram(arr, bins=bins)
    hist = hist.astype(float)
    if hist.sum() == 0:
        return 0
    return entropy(hist, base=2)

# 结果表头
header = ["filename"]
for r in radius_list:
    for t in ["stroma", "necrosis"]:
        header += [
            f"{t}_within_{r}_mean",
            f"{t}_within_{r}_std",
            f"{t}_within_{r}_median",
            f"{t}_within_{r}_q10",
            f"{t}_within_{r}_q25",
            f"{t}_within_{r}_q75",
            f"{t}_within_{r}_q90",
            f"{t}_within_{r}_iqr",         # 新增IQR
            f"{t}_within_{r}_entropy"      # 新增熵
        ]

# 结果表头增加最近邻距离统计
for t in ["stroma", "necrosis"]:
    header += [
        f"nearest_{t}_mean",
        f"nearest_{t}_std",
        f"nearest_{t}_median",
        f"nearest_{t}_q10",
        f"nearest_{t}_q25",
        f"nearest_{t}_q75",
        f"nearest_{t}_q90",
        f"nearest_{t}_iqr",         # 新增IQR
        f"nearest_{t}_entropy"      # 新增熵
    ]

# 结果表头扩展: 所有免疫细胞与肿瘤相关细胞的交互
for main_type in ["neoplasm", "stroma", "necrosis"]:
    for immune in immune_types:
        for r in radius_list:
            header += [
                f"{main_type}_within_{r}_{immune}_mean",
                f"{main_type}_within_{r}_{immune}_std",
                f"{main_type}_within_{r}_{immune}_median",
                f"{main_type}_within_{r}_{immune}_q10",
                f"{main_type}_within_{r}_{immune}_q25",
                f"{main_type}_within_{r}_{immune}_q75",
                f"{main_type}_within_{r}_{immune}_q90",
                f"{main_type}_within_{r}_{immune}_iqr",         # 新增IQR
                f"{main_type}_within_{r}_{immune}_entropy"      # 新增熵
            ]
        header += [
            f"{main_type}_nearest_{immune}_mean",
            f"{main_type}_nearest_{immune}_std",
            f"{main_type}_nearest_{immune}_median",
            f"{main_type}_nearest_{immune}_q10",
            f"{main_type}_nearest_{immune}_q25",
            f"{main_type}_nearest_{immune}_q75",
            f"{main_type}_nearest_{immune}_q90",
            f"{main_type}_nearest_{immune}_iqr",         # 新增IQR
            f"{main_type}_nearest_{immune}_entropy"      # 新增熵
        ]

# 原有免疫细胞间空间关系表头
immune_pairs = [
    ("CD8", "FOXP3"), ("CD8", "CD20"),
    ("FOXP3", "CD8"), ("FOXP3", "CD20"),
    ("CD20", "CD8"), ("CD20", "FOXP3"),
    ("CD3", "CD20"), ("CD20", "CD3")
]

# 新增巨噬细胞内部交互
macrophage_pairs = [
    ("CD68", "CD163"),
    ("CD163", "CD68")
]
immune_pairs.extend(macrophage_pairs)

for main, target in immune_pairs:
    for r in radius_list:
        header += [
            f"{main}_within_{r}_{target}_mean",
            f"{main}_within_{r}_{target}_std",
            f"{main}_within_{r}_{target}_median",
            f"{main}_within_{r}_{target}_q10",
            f"{main}_within_{r}_{target}_q25",
            f"{main}_within_{r}_{target}_q75",
            f"{main}_within_{r}_{target}_q90",
            f"{main}_within_{r}_{target}_iqr",         # 新增IQR
            f"{main}_within_{r}_{target}_entropy"      # 新增熵
        ]
    header += [
        f"{main}_nearest_{target}_mean",
        f"{main}_nearest_{target}_std",
        f"{main}_nearest_{target}_median",
        f"{main}_nearest_{target}_q10",
        f"{main}_nearest_{target}_q25",
        f"{main}_nearest_{target}_q75",
        f"{main}_nearest_{target}_q90",
        f"{main}_nearest_{target}_iqr",         # 新增IQR
        f"{main}_nearest_{target}_entropy"      # 新增熵
    ]

for json_dir, output_dir, svs_dir in path_pairs:
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    intermediate_csv = os.path.join(output_dir, 'intermediate.csv')
    processed_txt = os.path.join(output_dir, 'processed.txt')

    # 读取已处理文件名
    if os.path.exists(processed_txt):
        with open(processed_txt, 'r') as f:
            processed_files = set(line.strip() for line in f)
    else:
        processed_files = set()

    # 如果中间csv不存在,写入表头
    if not os.path.exists(intermediate_csv):
        with open(intermediate_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for json_path in json_files:
        fname = os.path.basename(json_path)
        if fname in processed_files:
            continue

        svs_path = os.path.join(svs_dir, os.path.splitext(fname)[0] + '.svs')
        if os.path.exists(svs_path):
            mpp = get_mpp_from_svs(svs_path)
        else:
            print(f"警告: 未找到对应的SVS文件 {svs_path}，使用默认值 0.25 微米/像素")
            mpp = 0.25

        with open(json_path, 'r') as f:
            data = json.load(f)

        coords_neoplasm = []
        coords_stroma = []
        coords_necrosis = []
        for cell in data['nuc'].values():
            t = str(cell['type'])
            c = cell['centroid']
            if t == "1":
                coords_neoplasm.append(c)
            elif t == "3":
                coords_stroma.append(c)
            elif t == "4":
                coords_necrosis.append(c)

        coords_neoplasm = np.array(coords_neoplasm)
        coords_stroma = np.array(coords_stroma)
        coords_necrosis = np.array(coords_necrosis)

        tree_stroma = cKDTree(coords_stroma) if len(coords_stroma) > 0 else None
        tree_necrosis = cKDTree(coords_necrosis) if len(coords_necrosis) > 0 else None

        # 提取所有免疫细胞坐标
        coords_immune = {k: [] for k in immune_types}
        for cell in data['nuc'].values():
            t = str(cell['type'])
            c = cell['centroid']
            for immune in immune_types:
                if t == immune_type_ids[immune]:
                    coords_immune[immune].append(c)

        for immune in immune_types:
            coords_immune[immune] = np.array(coords_immune[immune])

        trees_immune = {k: cKDTree(v) if len(v) > 0 else None for k, v in coords_immune.items()}

        # 最紧邻距离分析
        nearest_stroma = []
        nearest_necrosis = []
        if len(coords_neoplasm) > 0:
            if tree_stroma and len(coords_stroma) > 0:
                dists_stroma, _ = tree_stroma.query(coords_neoplasm, k=1)
                nearest_stroma = dists_stroma * mpp
            else:
                nearest_stroma = np.array([])
            if tree_necrosis and len(coords_necrosis) > 0:
                dists_necrosis, _ = tree_necrosis.query(coords_neoplasm, k=1)
                nearest_necrosis = dists_necrosis * mpp
            else:
                nearest_necrosis = np.array([])
        else:
            nearest_stroma = np.array([])
            nearest_necrosis = np.array([])

        # 统计每个neoplasm细胞的邻域计数
        stats = {}
        for r in radius_list:
            stats[f"stroma_{r}"] = []
            stats[f"necrosis_{r}"] = []

        for center in coords_neoplasm:
            for r in radius_list:
                r_pixel = r / mpp
                if tree_stroma:
                    count_stroma = len(tree_stroma.query_ball_point(center, r_pixel))
                else:
                    count_stroma = 0
                if tree_necrosis:
                    count_necrosis = len(tree_necrosis.query_ball_point(center, r_pixel))
                else:
                    count_necrosis = 0
                stats[f"stroma_{r}"].append(count_stroma)
                stats[f"necrosis_{r}"].append(count_necrosis)

        row = [os.path.basename(json_path)]
        for r in radius_list:
            for t in ["stroma", "necrosis"]:
                arr = np.array(stats[f"{t}_{r}"])
                if arr.size > 0:
                    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                    ent = calc_entropy(arr)
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90),
                        iqr,
                        ent
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 增加最近邻距离统计
        for arr in [nearest_stroma, nearest_necrosis]:
            if len(arr) > 0:
                iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                ent = calc_entropy(arr)
                row += [
                    np.mean(arr),
                    np.std(arr),
                    np.median(arr),
                    np.percentile(arr, 10),
                    np.percentile(arr, 25),
                    np.percentile(arr, 75),
                    np.percentile(arr, 90),
                    iqr,
                    ent
                ]
            else:
                row += [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 针对每类主细胞,分别做邻域计数和最近邻距离
        for main_type, coords_main in zip(
            ["neoplasm", "stroma", "necrosis"],
            [coords_neoplasm, coords_stroma, coords_necrosis]
        ):
            for immune in immune_types:
                tree_immune = trees_immune[immune]
                # 邻域计数
                stats = {r: [] for r in radius_list}
                for center in coords_main:
                    for r in radius_list:
                        r_pixel = r / mpp
                        if tree_immune:
                            count = len(tree_immune.query_ball_point(center, r_pixel))
                        else:
                            count = 0
                        stats[r].append(count)
                for r in radius_list:
                    arr = np.array(stats[r])
                    if arr.size > 0:
                        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                        ent = calc_entropy(arr)
                        row += [
                            np.mean(arr),
                            np.std(arr),
                            np.median(arr),
                            np.percentile(arr, 10),
                            np.percentile(arr, 25),
                            np.percentile(arr, 75),
                            np.percentile(arr, 90),
                            iqr,
                            ent
                        ]
                    else:
                        row += [0, 0, 0, 0, 0, 0, 0, 0, 0]
                # 最近邻距离
                if tree_immune and len(coords_main) > 0:
                    dists, _ = tree_immune.query(coords_main, k=1)
                    arr = dists * mpp
                else:
                    arr = np.array([])
                if len(arr) > 0:
                    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                    ent = calc_entropy(arr)
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90),
                        iqr,
                        ent
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 免疫细胞间空间关系统计(包括巨噬细胞内部交互)
        for main, target in immune_pairs:
            coords_main = coords_immune[main]
            coords_target = coords_immune[target]
            tree_target = trees_immune[target]
            # 邻域计数
            stats = {r: [] for r in radius_list}
            for center in coords_main:
                for r in radius_list:
                    r_pixel = r / mpp
                    if tree_target:
                        count = len(tree_target.query_ball_point(center, r_pixel))
                    else:
                        count = 0
                    stats[r].append(count)
            for r in radius_list:
                arr = np.array(stats[r])
                if arr.size > 0:
                    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                    ent = calc_entropy(arr)
                    row += [
                        np.mean(arr),
                        np.std(arr),
                        np.median(arr),
                        np.percentile(arr, 10),
                        np.percentile(arr, 25),
                        np.percentile(arr, 75),
                        np.percentile(arr, 90),
                        iqr,
                        ent
                    ]
                else:
                    row += [0, 0, 0, 0, 0, 0, 0, 0, 0]
            # 最近邻距离
            if tree_target and len(coords_main) > 0:
                dists, _ = tree_target.query(coords_main, k=1)
                arr = dists * mpp
            else:
                arr = np.array([])
            if len(arr) > 0:
                iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                ent = calc_entropy(arr)
                row += [
                    np.mean(arr),
                    np.std(arr),
                    np.median(arr),
                    np.percentile(arr, 10),
                    np.percentile(arr, 25),
                    np.percentile(arr, 75),
                    np.percentile(arr, 90),
                    iqr,
                    ent
                ]
            else:
                row += [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 追加写入中间csv
        with open(intermediate_csv, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 记录已处理文件
        with open(processed_txt, 'a') as f:
            f.write(fname + '\n')

    # 最终整理
    csv_path = os.path.join(output_dir, 'neoplasm_neigh_stats.csv')
    with open(intermediate_csv, 'r', encoding='utf-8-sig') as fin, \
         open(csv_path, 'w', newline='', encoding='utf-8-sig') as fout:
        lines = fin.readlines()
        fout.writelines(lines)