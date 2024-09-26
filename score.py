import numpy as np
from PIL import Image
import os


def calculate_iou(pred_mask_path, true_mask_path):
    pred_mask = np.array(Image.open(pred_mask_path))  # 读取预测掩码
    true_mask = np.array(Image.open(true_mask_path))  # 读取真实掩码
    
    intersection = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    union = np.logical_or(pred_mask == 1, true_mask == 1).sum()
    
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_f1(pred_mask_path, true_mask_path):
    pred_mask = np.array(Image.open(pred_mask_path))
    true_mask = np.array(Image.open(true_mask_path))
    
    true_positives = np.logical_and(pred_mask == 1, true_mask == 1).sum()
    false_positives = np.logical_and(pred_mask == 1, true_mask == 0).sum()
    false_negatives = np.logical_and(pred_mask == 0, true_mask == 1).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    
    if precision + recall == 0:
        return 0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_score(pred_mask_path, true_mask_path):
    iou = calculate_iou(pred_mask_path, true_mask_path)
    f1 = calculate_f1(pred_mask_path, true_mask_path)
    score = (0.5 * iou + 0.5 * f1) * 100
    return score

def main():

    label_dir = './dataset/val/labels/'
    result_dir = './submits/dataset/val/NL_LinkNet_EGaussian/'
    log_file = 'evaluation_log_NL_LinkNet_EGaussian.txt'

    total_score = 0
    image_count = 0

    # 打开日志文件写入
    with open(log_file, 'w') as log:
        print("Filename\tIOU\tF1\tScore\n")
        log.write("Filename\tIOU\tF1\tScore\n")  # 写入标题行

        # 遍历所有label文件
        for file_name in os.listdir(label_dir):
            label_path = os.path.join(label_dir, file_name)
            result_path = os.path.join(result_dir, file_name)
            
            if os.path.exists(result_path):
                # 计算IOU和F1分数
                iou = calculate_iou(result_path, label_path)
                f1 = calculate_f1(result_path, label_path)
                score = (0.5 * iou + 0.5 * f1) * 100

                # 写入每个文件的文件名及其分数
                print(f"{file_name}\t{iou:.4f}\t{f1:.4f}\t{score:.2f}\n")
                log.write(f"{file_name}\t{iou:.4f}\t{f1:.4f}\t{score:.2f}\n")
                
                # 计算总分
                total_score += score
                image_count += 1

        # 计算平均分数
        average_score = total_score / image_count if image_count != 0 else 0

        # 写入平均分数
        log.write(f"\nAverage Score: {average_score:.2f}\n")

    print(f"Evaluation completed. Results saved to {log_file}")


if __name__ == "__main__":
    main()