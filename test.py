import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import random

from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline
from networks.nllinknet_pairwise_func import NL_LinkNet_DotProduct, NL_LinkNet_Gaussian, NL_LinkNet_EGaussian
from networks.unet import Unet
from test_framework import TTAFramework


def test_models(model, name, source='./dataset/val', scales=(1.0,), target='./dataset/val/', num_samples=None):
    if type(scales) == tuple:
        scales = list(scales)
    print(model, name, source, scales, target)

    solver = TTAFramework(model)
    solver.load('weights/' + name + '.th')

    if target == '':
        target = 'submits/' + name + '/'
    else:
        target = 'submits/' + target + '/' + name+ '/'

    # source = '../dataset/Road/valid/'
    # val = os.listdir(source)

    img_source = os.path.join(source, 'images/')
    val = os.listdir(img_source)
    # 如果指定了 num_samples，则随机选择 num_samples 个样本
    if num_samples is not None:
        val = random.sample(val, num_samples)

    if not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError as e:
            import errno
            if e.errno != errno.EEXIST:
                raise
    len_scales = int(len(scales))
    if len_scales > 1:
        print('multi-scaled test : ', scales)

    for i, name in tqdm(enumerate(val), ncols=10, desc="Testing "):
        # mask = solver.test_one_img_from_path(img_source + name, scales)
        # print(img_source + name)
        # mask[mask > 4.0 * len_scales] = 255  # 4.0
        # mask[mask <= 4.0 * len_scales] = 0
        # mask = mask[:, :, None]
        # mask = np.concatenate([mask, mask, mask], axis=2)
        # print(2)
        # cv2.imwrite(target + name, mask.astype(np.uint8))
        # print(target + name)

        mask = solver.test_one_img_from_path(img_source + name, scales)
        # print(f"Mask data type: {mask.dtype}")
        # print(f"Mask type: {type(mask)}")
        # print(mask,"\n")
        # 二值化：将掩膜值大于某个阈值的设置为 1，其余为 0
        # print(f"len_scales:{len_scales}")
        mask[mask > 4.0 * len_scales] = 255
        mask[mask <= 4.0 * len_scales] = 0

        # # mask[mask >= 1.0] = 255
        # # mask[mask < 1.0] = 0
        # print(f"mask值{mask}")
        # # 保存为单通道的 0-1 二值图像
        # cv2.imwrite(target + name, mask.astype(np.uint8))  # 不需要转换为 RGB

        # print(mask.shape)       # 查看掩码的形状
        # print(np.max(mask))     # 查看掩码的最大值
        # print(np.min(mask))     # 查看掩码的最小值
        # print(np.unique(mask))  # 检查唯一值

        mask = mask.astype(np.uint8)  # 确保掩码图为 uint8 类型

        # cv2.imwrite(target + name[:-4] + "_255.png", mask)

        # print(mask.shape)       # 查看掩码的形状
        # print(np.max(mask))     # 查看掩码的最大值
        # print(np.min(mask))     # 查看掩码的最小值
        # print(np.unique(mask))  # 检查唯一值

        mask[mask == 255] = 1
        mask[mask == 0] = 0

        # print(mask.shape)       # 查看掩码的形状
        # print(np.max(mask))     # 查看掩码的最大值
        # print(np.min(mask))     # 查看掩码的最小值
        # print(np.unique(mask))  # 检查唯一值

        cv2.imwrite(target + name, mask)
        # img = cv2.imread(target + name, cv2.IMREAD_UNCHANGED)
        # print(np.unique(img))  # 输出图像中的唯一像素值，确保只有 0 和 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="set model name")
    parser.add_argument("--name", help="set path of weights")
    parser.add_argument("--source", help="path of test datasets", default='./dataset/val')
    parser.add_argument("--scales", help="set scales for MST", default=[1.0], type=float, nargs='*')
    parser.add_argument("--target", help="path of submit files", default='./dataset/val/')
    parser.add_argument("--num_samples", help="sample quantity during testing", type=int, default=None)

    args = parser.parse_args()

    models = {'NL3_LinkNet': NL3_LinkNet, 'NL4_LinkNet': NL4_LinkNet, 'NL34_LinkNet': NL34_LinkNet,
              'Baseline': Baseline,
              'NL_LinkNet_DotProduct': NL_LinkNet_DotProduct, 'NL_LinkNet_Gaussian': NL_LinkNet_Gaussian,
              'NL_LinkNet_EGaussian': NL_LinkNet_EGaussian,
              'UNet': Unet, 'LinkNet': LinkNet34, 'DLinkNet': DinkNet34}

    model = models[args.model]
    name = args.name
    scales = args.scales
    target = args.target
    source = args.source
    num_samples = args.num_samples

    test_models(model=model, name=name, source=source, scales=scales, target=target, num_samples=num_samples)


if __name__ == "__main__":
    main()
