写一个python:
需要将数据集meta data,划分为train, test
输入:一个file,每一行格式如下
val_perturb_release/6/be4f0b1cf4cdb741f86f538960a65dc6/47f4c19dd89e1e5aceb98ad52d0e0eb2/frame00049.jpg 1 2 4
relative file_path, <binary_cls_label> <triple_cls_label> <16cls_label>
file_path中/6/是第二个目录，表示

输出:train_list.txt, test_list.txt, 格式
FaceForensics++/original_sequences/youtube/c23/frames/000/000.png 0
relative file_path, binary_cls_label

划分规则：
file_path中/6/是第二级目录，取值为1-19.安装该目录划分
将1-12，16-18划分到train
13-15,19划分到test

2.
写python, 需要用input txt生成output json,json是用来做训练的，包含图片路径，label.
input txt:
/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ssl_vits_df/Py_data/ForgeryNet_test_list.txt
每一行格式如下
val_perturb_release/6/be4f0b1cf4cdb741f86f538960a65dc6/47f4c19dd89e1e5aceb98ad52d0e0eb2/frame00049.jpg 1 2 4
relative file_path, <binary_cls_label> <triple_cls_label> <16cls_label>
file_path中/6/是第二个目录

output json, 跟Py_data/UADFV_simplified.json几乎一样
json里面的内容需要用数据集名称填充，输出文件名也是，我的数据集名称是 ForgeryNet

划分规则：
file_path中/6/是第二级目录，取值为1-19.按照该目录划分
将1-12，16-18划分到train
13-15,19划分到test

做一个check, 保证转换正确：统计输入文件中的图片个数，输出文件个数。也就是行数，打印到终端。要代码检查train+test个数和input_file相等，如果不是，需要报错提示出来。另外，1-19的子文件夹的图片数据打印到终端
先说一下需求理解，然后才继续 



写一个python, 将input_json文件简化到output json
每个frames的key 下面的内容只保持2项





ForgeryNet img list.txt


#
py 要实现json to list
要把UADFC.json 生成 2个list
UADFV_train.list
UADFV_test.list
list格式如下,<image_path> <label>, 1表示fake
DFDC/test/frames/aqtrruhcat/019.png 0
DFDC/test/frames/aqtrruhcat/019.png 1

{
    "UADFV": {
        "UADFV_Real": {
            "test": {
                "video_id": {
                    "label": "UADFV_Real",
                    "frames": ["path/to/frame1", "path/to/frame2", ...]
                }
            }
        }
    }
}