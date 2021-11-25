import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.common_tools import set_seed, transform_invert
from utils.datasets import MultiLabelDataset

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((256, 128)),

    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),
    # transforms.ColorJitter(hue=.05),

    # 3 Grayscale
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.5, 0), fillcolor=(0, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),

    # 1 RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    # transforms.RandomOrder([transforms.RandomRotation(15),
    #                         transforms.Pad(padding=32),
    #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
data_path = '/home/lzk/data/face_data/RAP2/RAP_dataset'
train_dataset = MultiLabelDataset(root=data_path, label='/home/lzk/workspace/pedestrain_attribute/iccv19_attribute/data_list/my_rap2/train.txt', transform=train_transform)
val_dataset = MultiLabelDataset(root=data_path, label='/home/lzk/workspace/pedestrain_attribute/iccv19_attribute/data_list/my_rap2/test.txt', transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

save_path = '/home/lzk/workspace/pedestrain_attribute/iccv19_attribute/work_dir/data_aug'
# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels, img_name = data   # B C H W
        img_name = str(img_name)
        img_name = img_name[2:-3]
        # img_name = img_name[:-3] + 'jpg'
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        save_img = os.path.join(save_path, img_name)
        print("save_img = {}".format(save_img))
        # print(img.mode, img.size, img.format)
        img.save(save_img)
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.5)
        # plt.close()





