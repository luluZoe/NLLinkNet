import cv2
import numpy as np
import os
import glob

def modify_mask(image_path, output_path):
    # 读取图像
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查图像是否读取成功
    if mask is None:
        print(f"Error reading image: {image_path}")
        return
    
    # 修改像素值
    mask[mask>=128] = 1
    mask[mask<128] = 0
    
    # 确保 mask 的数据类型为 uint8
    mask = mask.astype(np.uint8)
    
    # 保存修改后的图像
    cv2.imwrite(output_path, mask)
    print(f"Processed and saved: {output_path}")

def batch_modify_masks(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入目录中的所有图像文件
    image_files = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))
    
    for image_file in image_files:
        # 构建输出文件路径
        base_name = os.path.basename(image_file)
        output_file = os.path.join(output_dir, base_name)
        
        # 修改并保存图像
        modify_mask(image_file, output_file)

if __name__ == "__main__":
    input_dir = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/deep_train/labels_old'  # 输入目录
    output_dir = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/deep_train/labels'  # 输出目录
    

    # 调用批量处理函数
    batch_modify_masks(input_dir, output_dir)