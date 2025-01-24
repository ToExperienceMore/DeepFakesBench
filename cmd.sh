#weight=./training/weights/xception_best.pth
#weight=./training/weights/altfreezing_best.pth
weight=./training/weights/tall_trainFF_testCDF.pth
#config=training/config/detector/altfreezing.yaml
config=training/config/detector/tall.yaml
#./training/config/detector/xception.yaml
set -x

python3 training/test.py --detector_path $config --test_dataset   "FF-DF"  "FF-F2F"  "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path $weight

python3 training/test.py --detector_path $config --test_dataset  DFDC  --weights_path $weight
#python3 training/test.py --detector_path $config --test_dataset   "FF-DF" "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path ./training/weights/xception_best.pth
