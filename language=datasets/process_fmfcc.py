import os

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
            
            if label is not None:  # 只有在标签存在时才处理
                if label == "real":
                    target_key = "FMFCC-V_Real"
                else:
                    target_key = "FMFCC-V_Fake"
                
                # 初始化frames列表
                frames = []
                
                # 遍历文件夹中的每个图片
                for img in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img)
                    if img.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
                        # 将图片路径添加到frames列表中
                        frames.append(img_path)
                
                # 只有在frames不为空时才添加到输出JSON中
                if frames:
                    output_json["FMFCC-V"][target_key]["test"][folder] = {"label": label, "frames": frames}
    
    return output_json 