import os
import re

def delete_files_matching_pattern(directory, pattern):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否符合模式
        if re.match(pattern, filename):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            except OSError as e:
                print(f'Error deleting {file_path}: {e}')

# 指定目录和文件名模式
directory = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/train/labels'
pattern = r'^666\d+888\.png$'

# 调用函数
delete_files_matching_pattern(directory, pattern)