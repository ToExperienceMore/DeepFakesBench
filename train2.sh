nohup python training/train.py \
--detector_path ./training/config/detector/timesformer.yaml \
--train_dataset "FaceForensics++" \
--test_dataset  "Celeb-DF-v2" > train_output2.log 2>&1 &