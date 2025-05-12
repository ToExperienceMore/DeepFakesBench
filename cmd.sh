#weight=./training/weights/xception_best.pth
#timesformer
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-02-18-16-15-32/test/avg/ckpt_best.pth

#timesformer+efficientnet
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-05-08-22-33-06/test/avg/ckpt_best.pth
weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-05-08-23-01-57/test/avg/ckpt_best.pth

#weight=./training/weights/altfreezing_best.pth
#weight=./training/weights/tall_trainFF_testCDF.pth
#config=training/config/detector/altfreezing.yaml
#config=training/config/detector/tall.yaml
#config=./training/config/detector/xception.yaml
config=./training/config/detector/timesformer.yaml
set -x

#python3 training/test.py --detector_path $config --test_dataset   "FF-DF"  "FF-F2F"  "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight

#python3 training/test.py --detector_path $config --test_dataset  DFDC  --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset   "DFDC" "FF-F2F" "FF-DF" "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset  "FaceForensics++" "UADFV" "Celeb-DF-v2" --weights_path $weight
python3 training/test.py --detector_path $config --test_dataset  "UADFV" --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset   "DFDC"  --weights_path $weight

#FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, Celeb-DF-v1, Celeb-DF-v2, DFDCP, DFDC, DeeperForensics-1.0, UADFV
#DFDC, FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, Celeb-DF-v2, , UADFV