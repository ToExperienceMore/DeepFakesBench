#config=./training/config/detector/timesformer.yaml
#config=./training/config/detector/efficientnet.yaml
#config=./training/config/detector/efficientnetv2.yaml
#config=./training/config/detector/ftcn.yaml
#config=./training/config/detector/uia_vit.yaml
#config=./training/config/detector/iid.yaml
#config=./training/config/detector/xception.yaml
#config=./training/config/detector/sbi.yaml
#config=./training/config/detector/altfreezing.yaml
#config=./training/config/detector/tall.yaml
#config=./training/config/detector/videomae.yaml
#config=./training/config/detector/tall_video.yaml
config=./training/config/detector/clip_enhanced.yaml
#config=./training/config/detector/clip_stan.yaml

# 设置检查点路径（如果有的话）
#checkpoint="./logs/training/clip_stan_2025-05-30-23-08-08/test/avg/ckpt_best.pth"
#checkpoint="/path/to/your/checkpoint.pth"

# 设置任务名称
#task_name="clip_stan_debug"
task_name="clip_enhanced_prompt"

#nohup python training/train.py \
#--detector_path $config \
#--train_dataset "FaceForensics++" \
#--test_dataset  "Celeb-DF-v2" > train_output_efficientnet_B0.log 2>&1 &

#--checkpoint /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-02-18-16-15-32/test/avg/ckpt_best.pth \

#nohup python training/train.py \
#--detector_path $config \
#--train_dataset "FMFCC-V" \
#--test_dataset  "FMFCC-V" > train_FMFCC-V_efficientnet_B0.log 2>&1 &

#nohup python training/train.py \
#--detector_path $config \
#--train_dataset "DFDC" "FaceForensics++"  \
#--test_dataset  "DFDC" "FaceForensics++" > train_DFDC_FaceForensics++_efficientnet_B0.log 2>&1 &

#--amp \
set -x

#--max_train_images 100 \
#--max_test_images 100 \
#--max_train_images 100 \
#--max_test_images 100 \
#${checkpoint:+--checkpoint $checkpoint} \
nohup python training/train.py \
--detector_path $config \
--train_dataset "FaceForensics++" \
--test_dataset "DFDC" \
--task_target $task_name \
> train_FF++_${task_name}-0531-debug.log 2>&1 &

#python training/train.py \
#--detector_path $config \
#--train_dataset "ForgeryNet"  \
#--test_dataset  "ForgeryNet" > train_ForgeryNet_xception1.log 2>&1 &