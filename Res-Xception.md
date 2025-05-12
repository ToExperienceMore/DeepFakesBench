(DeepfakeBench) root@autodl-container-f74b419777-65470359:DeepfakeBench# sh cmd.sh 
+ python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset FF-DF FF-FS FF-NT DeepFakeDetection FaceShifter --weights_path ./training/weights/xception_best.pth
['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
spatial_count=0 keep_stride_count=0
===> Load checkpoint done!
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:22<00:00, 12.35it/s]
dataset: FF-DF
acc: 0.8766756032171582
auc: 0.9876292559754349
eer: 0.05626255860683188
ap: 0.9893586750640395
pred: [0.99540114 0.09636033 0.508014   ... 0.82483906 0.9996791  0.08062071]
video_auc: 0.9973979591836735
label: [1 0 0 ... 1 1 0]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:20<00:00, 13.98it/s]
dataset: FF-FS
acc: 0.8711478338543993
auc: 0.9800314504692511
eer: 0.06764902880107167
ap: 0.983609148276284
pred: [0.26119277 0.65116924 0.00549823 ... 0.9999927  0.9939803  0.96219426]
video_auc: 0.9939285714285715
label: [0 1 0 ... 1 1 1]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:20<00:00, 13.88it/s]
dataset: FF-NT
acc: 0.8519758874748827
auc: 0.9486545358331937
eer: 0.11944630497878991
ap: 0.9555878198890766
pred: [0.04861896 0.00482206 0.9998247  ... 0.5840302  0.9980186  0.13759327]
video_auc: 0.9704591836734694
label: [0 0 1 ... 0 1 1]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3205/3205 [03:40<00:00, 14.53it/s]
dataset: DeepFakeDetection
acc: 0.8034834846549185
auc: 0.8467472463646415
eer: 0.24169071781025975
ap: 0.9783750435490077
pred: [0.6400272  0.98308116 0.9886657  ... 0.99874747 0.78666323 0.75933474]
video_auc: 0.8961245617534533
label: [1 1 1 ... 1 1 1]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 280/280 [00:19<00:00, 14.22it/s]
dataset: FaceShifter
acc: 0.5367269479794597
auc: 0.5908603225461223
eer: 0.4233087742799732
ap: 0.5484003299258837
pred: [0.6849107  0.3431891  0.25549412 ... 0.47702727 0.7538128  0.8280899 ]
video_auc: 0.6056122448979592
label: [1 0 1 ... 0 0 1]
===> Test Done!

(DeepfakeBench) root@autodl-container-f74b419777-65470359:DeepfakeBench# sh cmd.sh 
+ python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset DFDC --weights_path ./training/weights/xception_best.pth
['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
spatial_count=0 keep_stride_count=0
===> Load checkpoint done!
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4129/4129 [04:49<00:00, 14.28it/s]
dataset: DFDC
acc: 0.6415574192376396
auc: 0.7129731023873718
eer: 0.35436655338654865
ap: 0.7367475462394519
pred: [0.2541699  0.00319419 0.17357123 ... 0.4890571  0.31631312 0.34502238]
video_auc: 0.7398307758652645
label: [0 0 0 ... 1 1 0]
===> Test Done!

(DeepfakeBench) root@autodl-container-f74b419777-65470359:DeepfakeBench# sh cmd.sh 
+ python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset Celeb-DF-v2 --weights_path ./training/weights/xception_best.pth
['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype']
spatial_count=0 keep_stride_count=0
===> Load checkpoint done!
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 514/514 [00:37<00:00, 13.82it/s]
dataset: Celeb-DF-v2
acc: 0.695006090133983
auc: 0.7402563430868591
eer: 0.3286476868327402
ap: 0.8371910697293352
pred: [0.9719672  0.33310133 0.9071739  ... 0.9769001  0.76295537 0.3971603 ]
video_auc: 0.8164573694646398
label: [0 0 1 ... 1 1 0]
===> Test Done!