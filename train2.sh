nohup python training/train.py \
--detector_path ./training/config/detector/timesformer.yaml \
--train_dataset "FaceForensics++" \
--checkpoint /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/logs/training/timesformer_2025-02-18-16-15-32/test/avg/ckpt_best.pth \
--test_dataset  "Celeb-DF-v2" > train_output3.log 2>&1 &