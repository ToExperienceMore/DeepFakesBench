import os
import json
import sys

def generate_json(path, input_json):
    output_json = {
        "FMFCC-V": {
            "FMFCC-V_Real": {"train": {}, "test": {}},
            "FMFCC-V_Fake": {"train": {}, "test": {}}
        }
    }
    
    # 遍历路径下的每个文件夹
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            # 获取对应的标签
            label_key = f"{folder}.mp4"  # 生成对应的键
            label = input_json.get(label_key, {}).get("label", None)
            print("label_key:",label_key)
            print("label:",label)
            
            if label is not None:  # 只有在标签存在时才处理
                if label == "real":
                    target_key = "FMFCC-V_Real"
                else:
                    target_key = "FMFCC-V_Fake"
                
                # 初始化frames列表
                frames = []
                
                # 遍历文件夹中的每个图片
                print("#### folder_path:",folder_path)
                for img in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img)
                    if img.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
                        # 将图片路径添加到frames列表中
                        frames.append(img_path)
                
                # 只有在frames不为空时才添加到输出JSON中
                if frames:
                    print("frames:",frames)
                    output_json["FMFCC-V"][target_key]["test"][folder] = {"label": label, "frames": frames}
    
    return output_json

if __name__ == "__main__":
    # 从命令行参数获取路径和JSON文件
    if len(sys.argv) != 4:
        print("用法: python script.py <path> <input_json> <output_json>")
        sys.exit(1)

    path = sys.argv[1]
    input_json_file = sys.argv[2]
    output_json_file = sys.argv[3]

    # 读取输入的JSON文件
    with open(input_json_file, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    # 生成输出的JSON
    output_json = generate_json(path, input_json)

    # 将输出的JSON写入文件
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    print(f"输出已写入 {output_json_file}")