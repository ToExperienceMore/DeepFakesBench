(DeepfakeBench) root@autodl-container-f74b419777-65470359:DeepfakeBench# sh cmd.sh 
+ python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset DFDC --weights_path ./training/weights/xception_best.pth
['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
spatial_count=0 keep_stride_count=0
===> Load checkpoint done!
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4129/4129 [04:44<00:00, 14.51it/s]
dataset: DFDC
acc: 0.6415574192376396
auc: 0.7129731023873718
eer: 0.35436655338654865
ap: 0.7367475462394519
pred: [0.2541699  0.00319419 0.17357123 ... 0.4890571  0.31631312 0.34502238]
video_auc: 0.7398307758652645
label: [0 0 0 ... 1 1 0]
===> Test Done!
