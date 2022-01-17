import argparse
import logging
import traceback
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from tqdm import tqdm

import model as models
from mylogger import logger_init
from utils.datasets import attr_nums, get_test_data
from utils.display import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--resume_path', default='checkpoint/ublb_12_ma74-44_train_all_bs32.pth.tar', type=str,
                    required=False, help='(default=%(default)s)')
parser.add_argument('--test_data_path', default='test_data/rap_test_1k', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('-c', '--confidence', dest='confidence', action='store_true', required=False,
                    help='print attribute confidence in imag')
parser.add_argument('-s', '--show', dest='show', action='store_true', required=False, help='show attribute in imag')
parser.add_argument('-sp', '--speed', dest='speed', action='store_true', required=False, help='test infer speed')
parser.add_argument('-spf', '--speed_print', dest='speed_print', action='store_true', required=False,
                    help='test infer speed and print attribute')
parser.add_argument('--save_path', default='work_dir/F1_ped_test_output', type=str, required=False,
                    help='(default=%(default)s)')

args = parser.parse_args()

# ####
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
#
# task_log_path = os.path.join(log_path, 'speed')
# if not os.path.exists(task_log_path):
#     os.makedirs(task_log_path)
# log_name = task_log_path + '/' + rq + '.log'
# logfile = log_name
# fh = logging.FileHandler(logfile, mode='w')
# fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# logger.addHandler(fh)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])


def prepare_model():
    # create model
    if args.speed or args.speed_print:
        model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['ped_attr_tiny'])
    else:
        model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['my_rap2'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if args.speed or args.speed_print:
        checkpoint = torch.load('checkpoint/9_mA_81-72.pth.tar')
    else:
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
    # if batch_size default and speed test, batch_size = 64
    # speed test but you set batch_size, batch_size = what you set
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    a = datetime.now()
    test(test_loader, pa_model)
    b = datetime.now()
    during = (b - a).seconds

    if args.speed or args.speed_print:
        # print("=" * 100)
        logging.info("=" * 100)
        # print("batch_size = {}".format(args.batch_size))
        logging.info("batch_size = {}".format(args.batch_size))
        # print("num_workers = {}".format(args.num_workers))
        logging.info("num_workers = {}".format(args.num_workers))
        # print("image_num = {} 张".format(test_dataset.__len__()))
        logging.info("image_num = {} 张".format(test_dataset.__len__()))
        # print("time = {} s".format(during))
        logging.info("time = {} s".format(during))
        try:
            # print("infer speed = {} 张/s".format(round(test_dataset.__len__() / during, 2)))
            logging.info("infer speed = {} 张/s".format(round(test_dataset.__len__() / during, 2)))
        except ZeroDivisionError:
            # print("推理时间不足1s")
            logging.info("推理时间不足1s")
        # print("=" * 100)
        logging.info("=" * 100)


def test(val_loader, model):
    model.eval()

    if args.speed:
        for i, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
            # for i, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
            input, img_name = _
            input = input.cuda(non_blocking=True)
            output = model(input)
            bs = input.size(0)

            # maximum voting
            # if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

            output = torch.sigmoid(output.data).cpu().numpy()
    else:
        for i, _ in enumerate(val_loader):
            # for i, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
            input, img_name = _
            input = input.cuda(non_blocking=True)
            output = model(input)
            bs = input.size(0)

            # maximum voting
            # if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

            output = torch.sigmoid(output.data).cpu().numpy()

            for one_bs in range(bs):
                # print("img_name: {}".format(img_name[one_bs]))
                # logging.info("img_name: {}".format(img_name[one_bs]))
                one_img_name = img_name[one_bs]
                output_list = output[one_bs].tolist()
                # print("output_list = {}".format(output_list))
                if args.confidence:
                    attr_dict = my_rap2_dict_c(output_list)
                elif args.speed_print:
                    attr_dict = my_rap2_tiny_dict(output_list)
                else:
                    attr_dict = my_rap2_dict(output_list)
                # print(attr_dict)
                logging.info("img_name: {} ".format(img_name[one_bs]) + str(attr_dict))
                if args.show:
                    show_attribute_img(one_img_name, attr_dict)


if __name__ == '__main__':
    logger_init(log_file_name='infer_log', log_level=logging.INFO, log_dir='./logs/inference_log/')
    try:
        batch_test(args.test_data_path)
    except:
        logging.error(str(traceback.format_exc()))
