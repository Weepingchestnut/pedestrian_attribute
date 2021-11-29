import base64
import os
from io import BytesIO

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        """
            root: 图像路径
            label: 标签文件，如test.txt
            transform: 图像转换
            loader: Image.open(path).convert('RGB')
        """
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 接收一个索引，返回一个样本
        img_name, label = self.images[index]
        # print("img_name: {}".format(img_name))
        # print("label: {}".format(label))
        img = self.loader(os.path.join(self.root, img_name))
        # raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


class get_test_data(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        # images = []
        # img_names = open(label).readlines()
        # for img_name in img_names:
        #     if os.path.isfile(os.path.join(root, img_name)):
        #         images.append(img_name)
        #     else:
        #         print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        # self.images = images
        self.images = label
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        # raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.images)


attr_nums = {
    'rap': 51,
    'my_rap2': 62,
    'ped_attr': 61
}

description = {'rap': ['Female',
                       'AgeLess16',
                       'Age17-30',
                       'Age31-45',
                       'BodyFat',
                       'BodyNormal',
                       'BodyThin',
                       'Customer',
                       'Clerk',
                       'BaldHead',  # 秃头
                       'LongHair',  # 长头发
                       'BlackHair',  # 黑发
                       'Hat',  # 帽子
                       'Glasses',  # 眼镜
                       'Muffler',  # 围巾
                       'Shirt',  # 衬衫
                       'Sweater',  # 毛衣
                       'Vest',  # 背心
                       'TShirt',  # T恤
                       'Cotton',  # 棉衣
                       'Jacket',  # 夹克衫
                       'Suit-Up',  # 西装上衣
                       'Tight',  # 紧身衣
                       'ShortSleeve',  # 短袖
                       'LongTrousers',  # 长裤
                       'Skirt',  # 裙子
                       'ShortSkirt',  # 短裙
                       'Dress',  # 连衣裙
                       'Jeans',  # 牛仔裤
                       'TightTrousers',  # 紧身裤
                       'LeatherShoes',  # 皮鞋
                       'SportShoes',  # 运动鞋
                       'Boots',  # 靴子
                       'ClothShoes',  # 布鞋
                       'CasualShoes',  # 休闲鞋
                       'Backpack',  # 双肩背包
                       'SSBag',  # 单肩包
                       'HandBag',  # 手提包
                       'Box',  # 盒子
                       'PlasticBag',  # 塑料袋
                       'PaperBag',  # 纸袋子
                       'HandTrunk',  # 手推车
                       'OtherAttchment',  #
                       'Calling',  # 打电话
                       'Talking',  # 交谈
                       'Gathering',
                       'Holding',
                       'Pusing',
                       'Pulling',
                       'CarryingbyArm',
                       'CarryingbyHand']}

description_chinese = {'rap': ['性别',
                               '少年（16以下）',
                               '青年（17-30）',
                               '中年（31-45）',
                               '微胖',
                               '标准',
                               '偏瘦',
                               '顾客',
                               '柜员',
                               '光头',
                               '长发',
                               '黑发',
                               '戴帽',
                               '眼镜',
                               '围巾',
                               '衬衣',
                               '毛衣',
                               '马甲',
                               'T恤',
                               '棉服',
                               '夹克',
                               '西服',
                               '卫衣',
                               '短袖',
                               '长裤',
                               '裙子',
                               '短裙',
                               '连衣裙',
                               '牛仔裤',
                               '包腿裤',
                               '皮鞋',
                               '运动鞋',
                               '靴子',
                               '布鞋',
                               '休闲鞋',
                               '双肩包',
                               '单肩包',
                               '手提包',
                               '箱子',
                               '塑料袋',
                               '纸袋',
                               '车',
                               '无',
                               '打电话',
                               '交谈',
                               '聚集',
                               '抱东西',
                               '推东西',
                               '拉拽东西',
                               '夹带东西',
                               '拎东西'],
                       'my_rap2': [
                           '衬衣', '毛衣', '马甲', 'T恤', '棉服', '夹克', '西服', '卫衣', '短袖', '其他',              # 上衣
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 上衣颜色
                           '长裤', '短裤', '裙子', '短裙', '长裙', '连衣裙', '牛仔裤', '包腿裤',                             # 下衣
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 下衣颜色
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 鞋子颜色
                           '双肩包', '单肩包'                                                                     # 是否背包
                       ],

                       'ped_attr': [
                           '衬衣', '毛衣', '马甲', 'T恤', '棉服', '夹克', '西服', '卫衣', '短袖', '其他',  # 上衣
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 上衣颜色
                           '长裤', '短裤', '裙子', '短裙', '长裙', '连衣裙', '牛仔裤', '包腿裤',  # 下衣
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 下衣颜色
                           '黑', '白', '灰', '红', '绿', '蓝', '银', '黄', '棕', '紫', '粉', '橙', '混色', '其他',  # 鞋子颜色
                           '是否背包'
                       ]}


def Get_Dataset(experiment, data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),  # 以给定的概率随机地翻转给定的图像，默认p=0.5
        # transforms.ColorJitter(hue=.05, saturation=.05),  # hue色相参数，saturation饱和度参数
        # transforms.RandomRotation(20, resample=Image.BILINEAR),   # 随机旋转图片，中心旋转20°，重采样方法为双线性插值
        # transforms.RandomAffine(degrees=0, translate=(0.5, 0), fillcolor=(0, 0, 0)),    # 随机仿射变换，模拟左右遮挡
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])

    if experiment == 'rap':
        train_dataset = MultiLabelDataset(root=data_path,
                                          label='data_list/rap/train.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root=data_path,
                                        label='data_list/rap/test.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['rap'], description['rap']

    if experiment == 'my_rap2':
        train_dataset = MultiLabelDataset(root=data_path,
                                          label='data_list/my_rap2/train_all.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root=data_path,
                                        label='data_list/my_rap2/test_all.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['my_rap2'], description_chinese['my_rap2']
