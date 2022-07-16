# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch

random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    if img.mode != 'L':
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
    else:
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, 255)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, mag):
    return img


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        (Identity, 0, 1),
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class RandMagSingleAug:

    def __init__(self, idx):
        self.idx = idx

    def __call__(self, img):
        if self.idx == -1:
            return img
        aug_name = augment_list_name_plus_identity(for_autoaug=False)[self.idx]
        aug_mag = random.choice(range(10))
        img = apply_augment(img, aug_name, aug_mag/10)
        return img


def augment_list_name(for_autoaug=True):
    l_name = ['ShearX',   # 0
              'ShearY',   # 1
              'TranslateX',   # 2
              'TranslateY',   # 3
              'Rotate',   # 4
              'AutoContrast',   # 5
              'Invert',   # 6
              'Equalize',   # 7
              'Solarize',   # 8
              'Posterize',   # 9
              'Contrast',   # 10
              'Color',   # 11
              'Brightness',   # 12
              'Sharpness',   # 13
              'Cutout',   # 14
            ]
    if for_autoaug:
        l_name += ['CutoutAbs',  # compatible with auto-augment
                   'Posterize2',  # 9
                   'TranslateXAbs',  # 9
                   'TranslateYAbs',  # 9
                  ]
    return l_name


def augment_list_name_plus_identity(for_autoaug=True):
    l_name = ['ShearX',   # 0
              'ShearY',   # 1
              'TranslateX',   # 2
              'TranslateY',   # 3
              'Rotate',   # 4
              'AutoContrast',   # 5
              'Invert',   # 6
              'Equalize',   # 7
              'Solarize',   # 8
              'Posterize',   # 9
              'Contrast',   # 10
              'Color',   # 11
              'Brightness',   # 12
              'Sharpness',   # 13
              'Cutout',   # 14
              'Identity', # 15
            ]
    if for_autoaug:
        l_name += ['CutoutAbs',  # compatible with auto-augment
                   'Posterize2',  # 9
                   'TranslateXAbs',  # 9
                   'TranslateYAbs',  # 9
                  ]
    return l_name


def get_aug_op_index_dict(for_autoaug=True):
    aug_op_lst = augment_list_name_plus_identity(for_autoaug)

    op_idx = 0

    # Key: the aug op name; Value: the index of the aug op
    aug_op_index_dict = dict()

    for aug_op in aug_op_lst:
        aug_op_index_dict[aug_op] = op_idx
        op_idx += 1

    return aug_op_index_dict


def get_total_augment_policy(for_autoaug=True):
    op1 = augment_list_name_plus_identity(for_autoaug)
    op2 = augment_list_name_plus_identity(for_autoaug)

    aug_op_index_dict = get_aug_op_index_dict(for_autoaug)

    aug_policy_index = 0

    # Key: the arm index of the aug policy; Value: aug policy (op1, mag1, op2, mag1)
    aug_policy_index_dict = dict()
    # Key: the arm index of the aug policy; Value: the one-hot encoding of the aug policy
    aug_policy_encoding_dict = dict()
    # Key: the aug policy; Value: the index of the aug policy
    aug_policy_get_index_dict = dict()

    for op1_item in op1:
        for op2_item in op2:
            for mag1 in range(10):
                aug_policy_index_dict[aug_policy_index] = (op1_item, mag1, op2_item, mag1)
                aug_policy_get_index_dict[(op1_item, mag1, op2_item, mag1)] = aug_policy_index

                aug_policy_encoding = np.zeros(len(op1) + 10, dtype=np.float32)

                op1_idx = aug_op_index_dict[op1_item]
                op2_idx = aug_op_index_dict[op2_item]

                aug_policy_encoding[op1_idx] = 1
                aug_policy_encoding[op2_idx] = 1
                aug_policy_encoding[len(op1)+mag1] = 1

                aug_policy_encoding_dict[aug_policy_index] = aug_policy_encoding

                aug_policy_index += 1

    return aug_policy_index_dict, aug_policy_encoding_dict, aug_policy_get_index_dict


def get_total_augment_policy_op(for_autoaug=True):
    op1 = augment_list_name_plus_identity(for_autoaug)
    op2 = augment_list_name_plus_identity(for_autoaug)
    op3 = augment_list_name_plus_identity(for_autoaug)

    aug_op_index_dict = get_aug_op_index_dict(for_autoaug)

    aug_policy_index = 0

    # Key: the arm index of the aug policy; Value: aug policy (op1, mag1, op2, mag1)
    aug_policy_index_dict = dict()
    # Key: the arm index of the aug policy; Value: the one-hot encoding of the aug policy
    aug_policy_encoding_dict = dict()
    # Key: the aug policy; Value: the index of the aug policy
    aug_policy_get_index_dict = dict()

    for op1_item in op1:
        for op2_item in op2:
            for op3_item in op3:
                aug_policy_index_dict[aug_policy_index] = (op1_item, op2_item, op3_item)
                aug_policy_get_index_dict[(op1_item, op2_item, op3_item)] = aug_policy_index

                aug_policy_encoding = np.zeros(len(op1) + 10, dtype=np.float32)

                aug_policy_index += 1

    return aug_policy_index_dict, aug_policy_encoding_dict, aug_policy_get_index_dict


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
