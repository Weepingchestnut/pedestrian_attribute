import argparse
import os
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn

import model as models
from utils.datasets import attr_nums, get_test_data
from utils.display import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--resume_path', default='checkpoint/rap_epoch_9_attr51.pth.tar', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--td_path', default='test_data/rap_test/RAP_dataset', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('-s', '--show', dest='show', action='store_true', required=False, help='show attribute in imag')
parser.add_argument('--save_path', default='work_dir/my_rap2_output_img', type=str, required=False,
                    help='(default=%(default)s)')

args = parser.parse_args()

# resume_path = 'checkpoint/3_ma7265_bs32.pth.tar'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])


def prepare_model():
    # create model
    model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['rap'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    checkpoint = torch.load(args.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = False
    cudnn.deterministic = True

    return model


pa_model = prepare_model()


def batch_test(test_data_path):
    # make img_name list
    img_names = []
    for file in os.listdir(test_data_path):
        # 判断是否为空图像，跳过空图像，防止后续读取图像是报错
        if os.stat(os.path.join(test_data_path, file)).st_size == 0:
            pass
        else:
            img_names.append(file)

    test_dataset = get_test_data(root=test_data_path, label=img_names, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    a = datetime.now()
    test(test_loader, pa_model)
    b = datetime.now()
    during = (b - a).seconds
    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("image_num = {}".format(test_dataset.__len__()))
    print("time = {}".format(during))
    print("infer speed = {}".format(test_dataset.__len__() / during))


def test(val_loader, model):
    model.eval()

    for i, _ in enumerate(val_loader):
        input, img_name = _
        input = input.cuda(non_blocking=True)
        output = model(input)
        bs = input.size(0)

        # maximum voting
        # if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        output = torch.sigmoid(output.data).cpu().numpy()

        for one_bs in range(bs):
            print("img_name: {}".format(img_name[one_bs]))
            one_img_name = img_name[one_bs]
            output_list = output[one_bs].tolist()
            # print("output_list = {}".format(output_list))
            attr_dict = ped_attr_dict(output_list)
            print(attr_dict)
            if args.show:
                show_attribute_img(one_img_name, attr_dict)


if __name__ == '__main__':
    batch_test(args.test_data_path)
