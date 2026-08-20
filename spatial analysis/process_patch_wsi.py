import os
import h5py
import numpy as np
import click
from PIL import Image
import json
from tqdm import tqdm
import openslide

# 导入 DeepLIIF 的核心推理函数和后处理函数
from deepliif.models import get_opt, inference
from deepliif.postprocessing import compute_cell_results, DEFAULT_SEG_THRESH

def save_results_to_json(output_path, all_cells_info):
    """
    将一个WSI的所有细胞核信息保存为指定的JSON格式。
    """
    # 细胞类型映射：阳性->1, 阴性->2
    type_mapping = {True: 1, False: 2}

    nuc_dict = {}
    for i, cell in enumerate(all_cells_info):
        # 确保bbox, centroid, contour中的坐标是整数
        bbox = [[int(p[0]), int(p[1])] for p in cell['bbox']]
        centroid = [int(cell['centroid'][0]), int(cell['centroid'][1])]
        contour = [[int(p[0]), int(p[1])] for p in cell['boundary']]
        
        nuc_dict[str(i + 1)] = {
            "bbox": bbox,
            "centroid": centroid,
            "contour": contour,
            "type": type_mapping.get(cell['positive'], 0) # 0作为未知类型
        }

    final_json = {
        "mag": 10.0,
        "nuc": nuc_dict
    }

    with open(output_path, 'w') as f:
        json.dump(final_json, f, separators=(',', ':'))


@click.command()
@click.option('--h5-dir', required=True, type=click.Path(exists=True), help='包含 .h5 坐标文件的目录。')
@click.option('--svs-dir', required=True, type=click.Path(exists=True), help='包含 .svs 图像文件的目录。')
@click.option('--output-dir', required=True, type=click.Path(), help='保存新 .json 结果文件的目录。')
@click.option('--model-dir', default=r'D:\HCC_wsi\DeepLIIF\model-server\DeepLIIF_Latest_Model', type=click.Path(exists=True), help='DeepLIIF 模型文件所在的目录。')
@click.option('--patch-size', default=224, type=int, help='从SVS文件中提取的图像块大小。')
@click.option('--patch-level', default=0, type=int, help='从SVS文件的哪个层级提取图像块。')
def main(h5_dir, svs_dir, output_dir, model_dir, patch_size, patch_level):
    """
    处理预先分块的WSI数据。
    
    从 .h5 文件读取坐标，从对应的 .svs 文件提取图像块，运行DeepLIIF推理，
    并将所有细胞的分割结果整合到输出目录中的一个新 .json 文件中。
    """
    os.makedirs(output_dir, exist_ok=True)

    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    if not h5_files:
        print(f"在目录 {h5_dir} 中未找到 .h5 文件。")
        return

    print(f"找到 {len(h5_files)} 个 .h5 文件进行处理。")

    print("正在加载模型配置...")
    opt = get_opt(model_dir)
    opt.use_dp = True
    opt.gpu_ids = [0]
    overlap_size = 0
    print("模型配置加载完成。")

    # 在主循环中添加检查逻辑
    for h5_filename in tqdm(h5_files, desc="处理WSI"):
        base_name = os.path.splitext(h5_filename)[0]
        output_json_path = os.path.join(output_dir, f"{base_name}.json")
        
        # 新增存在性检查
        if os.path.exists(output_json_path):
            print(f"检测到已存在结果文件: {output_json_path}")
            continue

        h5_path = os.path.join(h5_dir, h5_filename)
        svs_path = os.path.join(svs_dir, base_name + '.svs')

        if not os.path.exists(svs_path):
            print(f"警告: 在目录 {svs_dir} 中未找到与 {h5_filename} 匹配的 .svs 文件，跳过。")
            continue
        
        print(f"\n正在处理: {h5_filename} 和 {os.path.basename(svs_path)}")

        try:
            slide = openslide.OpenSlide(svs_path)
            with h5py.File(h5_path, 'r') as f:
                if 'coords' not in f:
                    print(f"警告: {h5_filename} 中缺少 'coords' 数据集，跳过。")
                    continue
                
                coords = f['coords'][:]
                wsi_all_cells = [] # 用于收集当前WSI的所有细胞信息

                for patch_coord in tqdm(coords, desc=f"处理 {base_name} 的图像块", leave=False):
                    patch_image = slide.read_region(
                        location=(patch_coord[0], patch_coord[1]), 
                        level=patch_level, 
                        size=(patch_size, patch_size)
                    ).convert('RGB')

                    # 1. 运行推理，获取分割图和标记图
                    images = inference(
                        img=patch_image,
                        tile_size=patch_size,
                        overlap_size=overlap_size,
                        model_path=model_dir,
                        opt=opt,
                        seg_only=True
                    )

                    # 2. 对单个图像块运行后处理，获取详细细胞信息
                    if 'Seg' in images:
                        resolution = '40x' if patch_size > 384 else ('20x' if patch_size > 224 else '10x')
                        cell_data = compute_cell_results(
                            seg=images['Seg'],
                            marker=images.get('Marker'),
                            resolution=resolution,
                            seg_thresh=DEFAULT_SEG_THRESH
                        )
                        
                        # 3. 坐标转换并聚合
                        patch_x_offset, patch_y_offset = patch_coord[0], patch_coord[1]
                        for cell in cell_data['cells']:
                            # 将细胞的相对坐标转换为WSI的绝对坐标
                            cell['bbox'] = [[p[0] + patch_x_offset, p[1] + patch_y_offset] for p in cell['bbox']]
                            cell['centroid'] = (cell['centroid'][0] + patch_x_offset, cell['centroid'][1] + patch_y_offset)
                            cell['boundary'] = [[p[0] + patch_x_offset, p[1] + patch_y_offset] for p in cell['boundary']]
                            wsi_all_cells.append(cell)

            slide.close()

            if wsi_all_cells:
                output_json_path = os.path.join(output_dir, f"{base_name}.json")
                print(f"正在将结果保存到: {output_json_path}")
                save_results_to_json(output_json_path, wsi_all_cells)

        except Exception as e:
            print(f"处理 {h5_filename} 时发生错误: {e}")
            if 'slide' in locals() and slide:
                slide.close()

    print("\n所有文件处理完毕。")

if __name__ == '__main__':
    main()