import zipfile

try:
    with zipfile.ZipFile('/public/share/sugonhpctest02/share/sugonhpctest01/tao/images.zip', 'r') as zip_ref:
        zip_ref.extractall('extracted_images')
        print("Extracted successfully")
except zipfile.BadZipFile:
    print("The ZIP file is corrupt")