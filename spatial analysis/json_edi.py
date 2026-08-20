import json
import os
import glob

def process_json_file(filepath, output_dir, exclude_types, overwrite=True):
    try:
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        if not overwrite and os.path.exists(output_path):
            print(f"跳过已存在文件: {output_path}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nuc = data.get('nuc', {})
        nuc_filtered = [v for v in nuc.values() if v.get('type') not in exclude_types]
        nuc_renumbered = {str(i+1): v for i, v in enumerate(nuc_filtered)}
        data['nuc'] = nuc_renumbered

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        print(f"处理文件出错: {filepath}\n错误信息: {e}")

def batch_process_jsons(input_dirs, output_dirs, exclude_types, overwrite=True):
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_files = glob.glob(os.path.join(input_dir, '*.json'))
        for filepath in json_files:
            process_json_file(filepath, output_dir, exclude_types, overwrite)

if __name__ == '__main__':
    input_dirs = [
        r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\HE_hover',
        #r"M:\HCC\Cellpoint\chenzhou\CD66B_seg",
        #r"M:\HCC\Cellpoint\chenzhou\CD163_seg",
        #r"M:\HCC\Cellpoint\chenzhou\CD57_seg",
        #r"M:\HCC\Cellpoint\chenzhou\PD-L1_seg",
    ]
    output_dirs = [
        r'J:\乏氧研究\第三批\新建文件夹\肝切辅助靶免SVS\HE_hover\edited_jsons',
        #r"M:\HCC\Cellpoint\chenzhou\CD66B_seg\edited_jsons",
        #r"M:\HCC\Cellpoint\chenzhou\CD163_seg\edited_jsons",
        #r"M:\HCC\Cellpoint\chenzhou\CD57_seg\edited_jsons",
        #r"M:\HCC\Cellpoint\chenzhou\PD-L1_seg\edited_jsons",
    ]
    exclude_types = [0, 5]
    #IHC中1为阳性，2为阴性，0为不确定；HE中"0""nolabe","1""neopla","2""inflam","3""connec","4""necros","5""no-neo"
    batch_process_jsons(input_dirs, output_dirs, exclude_types, overwrite=False)