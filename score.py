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

    label_dir = 'path/to/label/folder'
    result_dir = 'path/to/result/folder'

    total_score = 0
    image_count = 0

    for file_name in os.listdir(label_dir):
        label_path = os.path.join(label_dir, file_name)
        result_path = os.path.join(result_dir, file_name)
        
        if os.path.exists(result_path):
            score = calculate_score(result_path, label_path)
            total_score += score
            image_count += 1

    average_score = total_score / image_count if image_count != 0 else 0
    print(f"Average Score: {average_score:.2f}")

