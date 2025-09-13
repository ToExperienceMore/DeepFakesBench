set -x
python training/predict.py \
    --detector_path ./training/config/detector/clip_enhanced.yaml \
    --weights_path ../deepfake-detection/weights/model.ckpt \
    --image_path /root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/datasets/rgb/DFDCP/original_videos/frames/1152039_A_001/000.png