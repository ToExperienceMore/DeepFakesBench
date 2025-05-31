from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        """
        计算视频级别的AUC和EER指标
        Args:
            image: 图像路径列表
            pred: 预测值列表（模型输出的概率值）
            label: 真实标签列表（0或1）
        Returns:
            v_auc: 视频级别的AUC值
            v_eer: 视频级别的EER值
        """
        # 初始化存储结果的字典和列表
        result_dict = {}  # 用于按视频分组存储数据
        new_label = []    # 存储视频级别的标签
        new_pred = []     # 存储视频级别的预测值

        # 将三个列表堆叠并转置，使得每个元素包含[图像路径, 预测值, 标签]
        # np.stack将三个列表垂直堆叠，np.transpose转置后，每行包含一个样本的完整信息
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
            # item[0]是图像路径
            s = item[0]
            # 处理不同操作系统的路径分隔符
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            # 获取视频ID（路径中的倒数第二个部分）
            a = parts[-2]  # 视频ID
            b = parts[-1]  # 帧ID

            # 如果这个视频ID还没有在字典中，创建一个新的列表
            if a not in result_dict:
                result_dict[a] = []

            # 将当前帧的信息添加到对应视频的列表中
            result_dict[a].append(item)

        # 获取所有视频的帧列表
        image_arr = list(result_dict.values())

        # 对每个视频进行处理
        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            # video是一个列表，包含该视频的所有帧
            # frame是一个包含[图像路径, 预测值, 标签]的数组
            for frame in video:
                # frame[1]是预测值（概率）
                pred_sum += float(frame[1])
                # frame[2]是标签（0或1）
                label_sum += int(frame[2])
                leng += 1
            # 计算该视频的平均预测值和平均标签
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))

        # 打印视频级别的预测值和标签数量
        print("video_len(new_pred):", len(new_pred))
        print("video_len(new_label):", len(new_label))

        # 计算ROC曲线的关键点
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        # 计算AUC值
        v_auc = metrics.auc(fpr, tpr)
        # 计算FNR（False Negative Rate）
        fnr = 1 - tpr
        # 计算EER（Equal Error Rate）
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    #print("img_names:", img_names)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc
    print("len(y_pred):",len(y_pred))
    print("len(y_true):",len(y_true))

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
