import argparse
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from tqdm import tqdm

import model as models
from utils.datasets import attr_nums, MultiLabelDataset, description_chinese
from utils.display import *

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--resume_path', default='checkpoint/ublb_12_ma74-44_train_all_bs32.pth.tar', type=str,
                    required=False, help='(default=%(default)s)')
parser.add_argument('--test_data_path', default='test_data/F1_ped_test', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--test_data_label', default='data_list/F1_face_test.txt', type=str, required=False,
                    help='(default=%(default)s)')

args = parser.parse_args()


def Get_TestDataset(data_path, data_label):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = MultiLabelDataset(root=data_path, label=data_label, transform=transform_test)

    return test_dataset, attr_nums['ped_attr'], description_chinese['ped_attr']


def prepare_model():
    # create model
    model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['my_rap2'])

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


def F1_test():
    test_dataset, attr_num, description = Get_TestDataset(args.test_data_path, args.test_data_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    a = datetime.now()
    test(test_loader, pa_model, attr_num, description)
    b = datetime.now()
    during = (b - a).seconds
    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("image_num = {} 张".format(test_dataset.__len__()))
    print("time = {} s".format(during))
    try:
        print("infer speed = {} 张/s".format(test_dataset.__len__() / during))
    except ZeroDivisionError:
        print("推理时间不足1s")


def test(val_loader, model, attr_num, description):
    model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
        input, target = _
        input = input.cuda(non_blocking=True)
        output = model(input)
        bs = input.size(0)

        # maximum voting
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        batch_size = target.size(0)
        tol = tol + batch_size
        output = torch.sigmoid(output.data).cpu().numpy()

        # output = np.where(output > 0.5, 1, 0)
        # target = target.cpu().numpy()
        # print(type(output))
        # print(output.shape)
        # print(output)
        # print("="*200)

        output_adj = []
        for one_bs in range(bs):
            output_list = output[one_bs].tolist()
            # print("output_list = {}".format(output_list))
            attr_dict, one_bs_output = my_rap2_dict_F1(output_list)
            # print(attr_dict)
            # print("one_bs_output = {}".format(one_bs_output))
            output_adj.append(one_bs_output)

        output = np.array(output_adj)
        target = target.cpu().numpy()   # [32, 62]
        if target.shape == (bs, attr_nums['my_rap2']):      # 以公共数据集测试用
            target = target[:, :attr_num + 1]

        # print(output.size())
        # print(type(output))
        # print(output.shape)
        # print(type(output.shape))
        # print(output)

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        if attr_num == 1:
            continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if (output[jt][it] == 1 and target[jt][it] == 1) or (output[jt][it] == -1 and target[jt][it] == 1):
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu + 1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)

    print('=' * 100)
    if attr_num != 1:
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)

        print('\t' + 'Precision: ' + str(prec))
        print('\t' + 'Recall:    ' + str(recall))
        print('\t' + 'F1_Score:  ' + str(f1))
    print('=' * 100)


if __name__ == '__main__':
    F1_test()
