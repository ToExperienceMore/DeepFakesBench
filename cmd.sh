set -x
python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset  DFDC  --weights_path ./training/weights/xception_best.pth

#python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset   "FF-DF"  "FF-F2F"  "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path ./training/weights/xception_best.pth
#python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset   "FF-DF" "FF-FS"  "FF-NT" "DeepFakeDetection"  "FaceShifter"  --weights_path ./training/weights/xception_best.pth
