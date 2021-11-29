import argparse
import warnings

import PIL.Image
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.backends import cudnn

import model as models
from utils.attri_dict import *
from utils.datasets import attr_nums

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')

args = parser.parse_args()

resume_path = 'checkpoint/ublb_12_ma74-44_train_all_bs32.pth.tar'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])


def prepare_model():
    # create model
    model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['my_rap2'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = False
    cudnn.deterministic = True

    return model


pa_model = prepare_model()


def get_attr_list(output_list):
    attr_list = ['未知', '未知', '未知', '未知', '未知', '0']

    ub_list = output_list[0:10]
    ub_color_list = output_list[10:24]
    lb_list = output_list[24:32]
    lb_color_list = output_list[32:46]
    shoes_color_list = output_list[46: 60]
    bag_list = output_list[60:62]

    if max(ub_list) < 0.5:
        pass
    else:
        attr_list[0] = rap2_ub_dict[ub_list.index(max(ub_list))]

    if max(ub_color_list) < 0.5:
        pass
    else:
        attr_list[1] = rap2_color_dict[ub_color_list.index(max(ub_color_list))]

    if max(lb_list) < 0.5:
        pass
    else:
        attr_list[2] = rap2_lb_dict[lb_list.index(max(lb_list))]

    if max(lb_color_list) < 0.4:
        pass
    else:
        attr_list[3] = rap2_color_dict[lb_color_list.index(max(lb_color_list))]

    if max(shoes_color_list) < 0.4:
        pass
    else:
        attr_list[4] = rap2_color_dict[shoes_color_list.index(max(shoes_color_list))]
    # bag = rap2_bag_dict[bag_list.index(max(bag_list))]
    if max(bag_list) > 0.5:
        attr_list[5] = '1'

    return attr_list


# one image inference
def ped_attr(img_input):
    pa_model.eval()
    # 图片预处理
    if not PIL.Image.isImageType(img_input):
        # cv2 to PIL
        img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    else:
        img = img_input
    img = transform_test(img)
    img = torch.unsqueeze(img, 0)
    # print("input.size() = {}".format(input.size()))
    img = img.cuda(non_blocking=True)
    # print("output = model(input)")
    # 模型推理
    output = pa_model(img)
    # maximum voting
    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    output = torch.sigmoid(output.data).cpu().numpy()
    # print("output = {}".format(output))
    output_list = output[0].tolist()
    # 返回人体属性字典
    return get_attr_list(output_list)


if __name__ == '__main__':
    img = cv2.imread('test_data/CAM17_2014-02-20_20140220175154-20140220175854_tarid124_frame2893_line1.png')
    # img = Image.open('test_data/CAM17_2014-02-20_20140220175154-20140220175854_tarid124_frame2893_line1.png').convert('RGB')
    print(ped_attr(img))
