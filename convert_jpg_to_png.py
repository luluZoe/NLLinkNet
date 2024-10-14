import os
from PIL import Image

def convert_jpg_to_png(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            base_filename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_filename}.png")
            img.save(output_path, 'PNG')
            print(f"Converted {img_path} to {output_path}")

if __name__ == "__main__":
    input_dir = '/public/home/zzutaopw/workspace/NL-LinkNet/new/images/wh_6'
    output_dir = '/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset/train/images'
    convert_jpg_to_png(input_dir, output_dir)