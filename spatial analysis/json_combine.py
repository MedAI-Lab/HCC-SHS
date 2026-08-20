import json
import os
# import argparse # 不再需要
from collections import defaultdict

def parse_src_args(src_args):
    # src_args: ['path1:type1', 'path2:type2', ...]
    src_list = []
    for item in src_args:
        path, type_str = item.rsplit(':', 1)  # 只分割最后一个冒号
        src_list.append((path, int(type_str)))
    return src_list

def merge_jsons(filename, src_list):
    merged_nuc = []
    for src_path, type_val in src_list:
        json_path = os.path.join(src_path, filename)
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nuc = data.get('nuc', {})
        for obj in nuc.values():
            if type_val != -1:
                obj['type'] = type_val
            merged_nuc.append(obj)
    # 重新编号
    merged_nuc_dict = {str(i+1): obj for i, obj in enumerate(merged_nuc)}
    # 以第一个文件的 mag 为主
    mag = None
    for src_path, _ in src_list:
        json_path = os.path.join(src_path, filename)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                mag = json.load(f).get('mag', 10.0)
            break
    return {"mag": mag, "nuc": merged_nuc_dict}

def main():
    # --- 在此处直接修改参数 ---
    # 源文件夹列表，格式: ['路径1:类型1', '路径2:类型2', ...]
    src_args = [
        r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\HE_hover\edited_jsons:-1',
        #r'I:\chenzhou\CD3_seg\edited_jsons:5',
        #r'I:\chenzhou\CD8_seg\edited_jsons:6',
        #r'I:\chenzhou\FOXp3_seg\edited_jsons:7',
        #r'I:\chenzhou\CD20_seg\edited_jsons:8',
        #r'I:\chenzhou\CD68_seg\edited_jsons:9',
        #r'I:\chenzhou\CD163_seg\edited_jsons:10',
        #r'I:\chenzhou\CD66B_seg\edited_jsons:11',
        #r'I:\chenzhou\CD57_seg\edited_jsons:12',
        #r'I:\chenzhou\PD-L1_seg\edited_jsons:13',
        r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\CAIX_R_seg\edited_jsons:14'
    ]
    # 输出文件夹
    out_folder = r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\乏氧json'
    # 如果输出文件已存在，是否跳过
    skip_exist = True
    # --- 参数设置结束 ---

    src_list = parse_src_args(src_args)
    # 只用第一个路径下的文件名
    base_path = src_list[0][0]
    filenames = set()
    if os.path.exists(base_path):
        filenames.update([f for f in os.listdir(base_path) if f.endswith('.json')])

    os.makedirs(out_folder, exist_ok=True)

    for filename in filenames:
        out_path = os.path.join(out_folder, filename)
        if skip_exist and os.path.exists(out_path):
            print(f"Skip existing file: {out_path}")
            continue
        merged_data = merge_jsons(filename, src_list)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, separators=(',', ':'))
        print(f"Generated: {out_path}")

if __name__ == '__main__':
    main()

#    "1": "neopla", "2": "inflam", "3": "connec", "4": "necros"
#  5:CD3,6:CD8,7:FOXp3,8:CD20,9:CD68,10:CD163,11:CD66B，12: CD57,
# 13：PD-L1，14:CAIX