import cv2
import numpy as np

def print_image_pixels(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查图像是否读取成功
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    
    # 打印图像的基本信息
    print(f"Image shape: {image.shape}")
    print(f"Image data type: {image.dtype}")
    
    # 打印前几行像素值
    print("First few rows of pixel values:")
    print(image[:100, :10])  # 打印前30行的像素值

if __name__ == "__main__":
    image_path = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset_test/train/labels/999667.png'  # 替换为你的图像路径
    print_image_pixels(image_path)