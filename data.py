"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           scale_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           rotate_limit=(-3.14, 3.14),  # [-3.14,3.14]
                           aspect_limit=(-0.25, 0.25),  # [-0.25, 1.25]
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # - 3.14 ~ 3.14
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])  # -0.25, +1.25
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])  # -0.25, +0.25
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_load(id, root):
    # img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(os.path.join(root + '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)

    # id 是去掉后缀的文件名，即图片编码
    img_path = os.path.join(root, 'images', f'{id}.png')  # 修改为新的路径和文件名格式
    mask_path = os.path.join(root, 'labels', f'{id}.png')  # 修改为新的路径和文件名格式

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)  # 转换为 float32 类型

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    # img = np.array(img, np.float32).transpose(2,0,1)/255.0
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    # mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) 
    # mask[mask > 0.5] = 1
    # mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root, crop_size=(1024, 1024)):
        if type(crop_size) is tuple:
            crop_size = list(crop_size)
        self.ids = trainlist
        self.load = default_load
        self.root = root
        self.crop_size = crop_size

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.load(id, self.root)


        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        if self.crop_size[0] >= 1024:
            return img, mask

        y, x = torch.randint(low=0, high=1024 - int(self.crop_size[0]) - 1, size=(2,))
        w, h = torch.Tensor(self.crop_size)
        y = int(y)
        x = int(x)
        croped_img = img[:, y:int(y + h), x:int(x + w)]
        croped_mask = mask[:, y:int(y + h), x:int(x + w)]

        padded_img = F.pad(croped_img, (0, self.crop_size[1] - croped_img.shape[2], 0, self.crop_size[0] - croped_img.shape[1]))
        padded_mask = F.pad(croped_mask, (0, self.crop_size[1] - croped_mask.shape[2], 0, self.crop_size[0] - croped_mask.shape[1]))

        return padded_img, padded_mask

        # return croped_img, croped_mask

    def __len__(self):
        return len(list(self.ids))
