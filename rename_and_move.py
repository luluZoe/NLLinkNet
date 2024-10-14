import os
import re
import shutil

def rename_and_move_images(source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            # 检查文件名是否为纯数字
            if re.match(r'^\d+\.png$', filename):
                # 提取数字部分
                number = filename[:-4]
                # 构建新的文件名
                new_filename = f'666{number}888.png'
                # 构建完整的文件路径
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, new_filename)
                # 移动并重命名文件
                shutil.move(source_path, target_path)
                print(f'Moved and renamed: {filename} -> {new_filename}')

# 指定源目录和目标目录
source_directory = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/deep_train/labels'
target_directory = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/train/labels'

# 调用函数
rename_and_move_images(source_directory, target_directory)