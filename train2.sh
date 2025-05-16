#config=./training/config/detector/timesformer.yaml
#config=./training/config/detector/efficientnet.yaml
config=./training/config/detector/efficientnetv2.yaml
#config=./training/config/detector/iid.yaml
#config=./training/config/detector/xception.yaml
#config=./training/config/detector/sbi.yaml
#config=./training/config/detector/altfreezing.yaml
#config=./training/config/detector/tall.yaml

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

set -x

nohup python training/train.py \
--detector_path $config \
--train_dataset "FaceForensics++"  \
--test_dataset  "DFDC" "Celeb-DF-v2" > train_FaceForensics++_efficientnetv2.log 2>&1 &