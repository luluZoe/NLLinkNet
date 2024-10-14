"""
批处理文件夹train下的images和labels文件夹中的图片，进行重新修改尺寸操作，并替换保存。
注意是**批处理**，会把train中所有的文件夹下的所有图片都修改分辨率。
敲黑板，划重点！！！会直接把 原分辨率 的图片替换为 指定分辨率的图片！！！
文件夹结构：
    PATH：'/home/stu/zy/MySwin-Unet/data/train'
    childPATH：PATH下的'/labels'
    childPATH：PATH下的'images'


"""

import cv2
import os
import sys


# PATH = r'/root/autodl-tmp/MySwin-Unet/data/' # 这个路径只需写到train和val文件夹即可。文件夹下的图片程序会自动帮你打开
PATH = r'/public/home/zzutaopw/workspace/NL-LinkNet/NLLinkNet/dataset_test/train'
# 我这里是相对路径,亲测中文路径也可以

def resizeImage(file, NoResize):
    image = cv2.imread(file, cv2.IMREAD_COLOR)

    # 如果type(image) == 'NoneType',会报错,导致程序中断,所以这里先跳过这些图片,
    # 并记录下来,结束程序后手动修改(删除)

    if image is None:
        NoResize += [str(file)]
    else:
        resizeImg = cv2.resize(image, (512, 512)) # 这里改为自己想要的分辨率
        cv2.imwrite(file, resizeImg)
        cv2.waitKey(100)


def resizeAll(root):
    # 待修改文件夹
    fileList = os.listdir(root)
    # 输出文件夹中包含的文件
    # print("修改前："+str(fileList))
    # 得到进程当前工作目录
    currentpath = os.getcwd()
    # 将当前工作目录修改为待修改文件夹的位置
    os.chdir(root)

    NoResize = []  # 记录没被修改的图片

    for file in fileList:  # 遍历文件夹中所有文件
        file = str(file)
        resizeImage(file, NoResize)

    print("---------------------------------------------------")

    os.chdir(currentpath)  # 改回程序运行前的工作目录

    sys.stdin.flush()  # 刷新

    print('没别修改的图片: ', NoResize)


if __name__ == "__main__":
    # 子文件夹
    for childPATH in os.listdir(PATH):
        # 子文件夹路径
        childPATH = PATH + '/' + str(childPATH)
        # print(childPATH)
        resizeAll(childPATH)
    print('------修改图片大小全部完成❥(^_-)')


