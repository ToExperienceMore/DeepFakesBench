weight=./training/weights/xception_best.pth
#weight=/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-02-18-16-15-32/test/avg/ckpt_best.pth

#weight=./training/weights/altfreezing_best.pth
#weight=./training/weights/tall_trainFF_testCDF.pth
#config=training/config/detector/altfreezing.yaml
#config=training/config/detector/tall.yaml
config=./training/config/detector/xception.yaml
#config=./training/config/detector/timesformer.yaml
set -x

#python3 training/test.py --detector_path $config --test_dataset   "FF-DF"  "FF-F2F"  "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight

#python3 training/test.py --detector_path $config --test_dataset  DFDC  --weights_path $weight
python3 training/test.py --detector_path $config --test_dataset   "FF-DF" "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight
