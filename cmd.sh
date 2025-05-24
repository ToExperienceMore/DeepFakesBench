weight=./training/weights/xception_best.pth
#test_list=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ssl_vits_df/val_list-n.txt
#test_list=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ssl_vits_df/test_lists/FaceForensics++_test_list.txt
#test_list=/root/autodl-tmp/benchmark_deepfakes/ssl_vits_df/Py_data/ForgeryNet_test_list.txt
#weight=./logs/training/xception_2025-05-15-05-13-08/test/avg/ckpt_best.pth
#weight=./logs/training/xception_2025-05-15-05-13-08/test/avg/ckpt_best.pth

#weight=./logs/training/efficientnetb0_2025-05-15-07-10-51/test/avg/ckpt_best.pth
#timesformer
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-02-18-16-15-32/test/avg/ckpt_best.pth

#timesformer+efficientnet
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-05-08-22-33-06/test/avg/ckpt_best.pth
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-05-08-23-01-57/test/avg/ckpt_best.pth

#efficientnetb0 FMFCC-V
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/efficientnetb0_2025-05-13-20-36-42/test/avg/ckpt_best.pth

#efficientnetb0 FF++
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/efficientnetb0_2025-05-12-00-18-17/test/avg/ckpt_best.pth

#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-05-13-22-54-49/test/avg/ckpt_best.pth


#weight=./training/weights/altfreezing_best.pth
#weight=./training/weights/tall_trainFF_testCDF.pth
#config=training/config/detector/altfreezing.yaml
#config=training/config/detector/tall.yaml
config=./training/config/detector/xception.yaml
#config=./training/config/detector/timesformer.yaml
#config=./training/config/detector/efficientnet.yaml
#config=./training/config/detector/clip_enhanced.yaml
# 使用deepfake-detection的权重
#weight=../deepfake-detection/weights/model.ckpt
set -x

#python3 training/test.py --detector_path $config --test_dataset   "FF-DF"  "FF-F2F"  "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight

#python3 training/test.py --detector_path $config --test_dataset  DFDC  --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset   "DFDC" "FF-F2F" "FF-DF" "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset  "FaceForensics++" "UADFV" "Celeb-DF-v2" --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset  "FaceForensics++" "DFDC" "UADFV" --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset   "DFDC"  --weights_path $weight

#FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, Celeb-DF-v1, Celeb-DF-v2, DFDCP, DFDC, DeeperForensics-1.0, UADFV
#DFDC, FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, Celeb-DF-v2, , UADFV
#python3 training/test.py --detector_path $config --test_dataset  "FaceForensics++" --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset  "FMFCC-V" "UADFV" "Celeb-DF-v2" "DeepFakeDetection" "FF-F2F" "FF-DF" "FF-FS" "FF-NT" "FaceShifter" --weights_path $weight

python3 training/test.py --detector_path $config --test_dataset  "FaceForensics++" --weights_path $weight

# 示例命令

#python3 training/test2.py \
#    --detector_path $config \
#    --test_list $test_list \
#    --weights_path $weight

#--data_dir /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/FaceForensics++