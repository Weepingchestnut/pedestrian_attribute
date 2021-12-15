import argparse
import base64
import json
import time
import warnings
from io import BytesIO
from pprint import pprint

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import tornado
from PIL import Image
from torch.backends import cudnn

import model as models
from utils.datasets import attr_nums
from utils.display import my_rap2_dict

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


def isBase64(s):
    """Check s is Base64.b64encode"""
    if not isinstance(s, str) or not s:
        return False

    _base64_code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
                    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
                    '2', '3', '4', '5', '6', '7', '8', '9', '+',
                    '/', '=']

    # Check base64 OR codeCheck % 4
    code_fail = [i for i in s if i not in _base64_code]
    if code_fail or len(s) % 4 != 0:
        return False
    return True


def res_message(img_index, mode):
    if mode == 'id':
        return '第' + str(img_index) + '张图片id为空'
    elif mode == 'base64':
        return '第' + str(img_index) + '张图片损坏/非base64编码'
    elif mode == 'dict_format_0':
        return '第' + str(img_index) + '张图片JSON格式缺失键img_id与base64_code'
    elif mode == 'dict_format_1':
        return '第' + str(img_index) + '张图片JSON缺失键base64_code'
    elif mode == 'dict_format_10':
        return '第' + str(img_index) + '张图片JSON缺失键img_id'


def is_need_dict(img_dict):
    record = 0
    if 'img_id' in img_dict.keys():
        record = record + 1
    if 'base64_code' in img_dict.keys():
        record = record + 10
    return record
    # return 'img_id' in img_dict.keys() and 'base64_code' in img_dict.keys()


class base64_api(tornado.web.RequestHandler):
    def initialize(self, gconf):
        self.config = gconf
        self.pool = gconf.get("threadpool", None)

    # @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        start_time = time.time()
        request = json.loads(self.request.body)
        img_list = request.get('img_list', '')

        test_dataset = get_interface_data(imgs_list=img_list)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        stat = True
        response = {'success': True, 'message': ['属性提取成功'], 'attribute': [], 'spendTime': '0 ms'}
        for i, img_dict in enumerate(img_list):
            # 判断JSON是否满足dict格式要求
            get_record = is_need_dict(img_dict)
            if get_record < 11:
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'dict_format_' + str(get_record)))
                stat = False
            else:
                # 图片id为空（字符串长度为0 or 字符串均为空格）
                if len(str(img_dict['img_id'])) == 0 or str(img_dict['img_id']).isspace is True:
                    response['success'] = False
                    response['message'][0] = '属性提取失败'
                    response['message'].append(res_message(i + 1, 'id'))
                    stat = False

                if not isBase64(img_dict['base64_code']):
                    response['success'] = False
                    response['message'][0] = '属性提取失败'
                    response['message'].append(res_message(i + 1, 'base64'))
                    stat = False

        if not stat:
            pass
        else:
            # start_time = time.time()
            response['attribute'] = test(test_loader, pa_model)
            # end_time = time.time()
            # response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"
        end_time = time.time()
        response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"
        self.write(response)
        print(response)


def pil_to_base64(p264_img):
    img_buffer = BytesIO()
    p264_img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def base64_to_pil(base64_str):
    img = base64.b64decode(base64_str)
    img = BytesIO(img)
    img = Image.open(img)  # .convert('RGB')
    return img


class get_interface_data(data.Dataset):
    def __init__(self, imgs_list, transform=transform_test, loader=base64_to_pil):
        self.images = imgs_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_id = self.images[index]['img_id']
        img = self.loader(self.images[index]['base64_code'])

        if self.transform is not None:
            img = self.transform(img)
        return img, img_id

    def __len__(self):
        return len(self.images)


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


# one image inference
def ped_attr(img_input):
    pa_model.eval()
    # 图片预处理
    img = transform_test(img_input)
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
    return my_rap2_dict(output_list)


def interface_test():
    imag = Image.open('test_data/CAM01_2014-02-15_20140215163848-20140215165240_tarid151_frame2771_line1.png').convert(
        'RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    print("base64_1 = {}".format(base64_code))
    img1 = {
        'img_id': 'CAM01_2014-02-15_20140215163848-20140215165240_tarid151_frame2771_line1.png',
        'base64_code': base64_code
    }

    imag = Image.open('test_data/CAM01_2014-02-15_20140215163848-20140215165240_tarid158_frame2878_line1.png').convert(
        'RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    print("base64_2 = {}".format(base64_code))
    img2 = {
        'img_id': 'CAM01_2014-02-15_20140215163848-20140215165240_tarid158_frame2878_line1.png',
        'base64_code': base64_code
    }

    imag = Image.open('test_data/CAM01_2014-02-15_20140215163848-20140215165240_tarid163_frame2902_line1.png').convert(
        'RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    print("base64_3 = {}".format(base64_code))
    img3 = {
        'img_id': 'CAM01_2014-02-15_20140215163848-20140215165240_tarid163_frame2902_line1.png',
        'base64_code': base64_code
    }

    start_time = time.time()
    info_list = [img1, img2, img3]
    pprint(info_list)
    # print(info_list[0]['base64_code'])

    test_dataset = get_interface_data(imgs_list=info_list)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    stat = True
    response = {'success': True, 'message': ['属性提取成功'], 'attribute': [], 'spendTime': '0 ms'}
    for i, img_dict in enumerate(info_list):
        # 判断JSON是否满足dict格式要求
        get_record = is_need_dict(img_dict)
        if get_record < 11:
            response['success'] = False
            response['message'][0] = '属性提取失败'
            response['message'].append(res_message(i + 1, 'dict_format_' + str(get_record)))
            stat = False
        else:
            # 图片id为空（字符串长度为0 or 字符串均为空格）
            if len(str(img_dict['img_id'])) == 0 or str(img_dict['img_id']).isspace is True:
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'id'))
                stat = False

            if not isBase64(img_dict['base64_code']):
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'base64'))
                stat = False

    if not stat:
        # print("stat = False")
        pass
    else:
        # print("stat = True")
        # start_time = time.time()
        response['attribute'] = test(test_loader, pa_model)
        # end_time = time.time()
        # response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"
    end_time = time.time()
    response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"
    print(response)


def test(val_loader, model):
    model.eval()
    attr = []
    # img_dict = {'img_id': '', 'attr_dict': dict()}

    for i, _ in enumerate(val_loader):
        input, img_id = _
        input = input.cuda(non_blocking=True)
        output = model(input)
        bs = input.size(0)

        # maximum voting
        # if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

        output = torch.sigmoid(output.data).cpu().numpy()

        for one_bs in range(bs):
            # print("img_id: {}".format(img_id[one_bs]))
            img_dict = dict()
            one_img_id = img_id[one_bs]
            output_list = output[one_bs].tolist()
            # print("output_list = {}".format(output_list))
            attr_dict = my_rap2_dict(output_list)
            # print(attr_dict)
            img_dict['img_id'] = one_img_id
            img_dict['attr_dict'] = attr_dict
            attr.append(img_dict)
            # print("=" * 100)

    return attr


if __name__ == '__main__':
    interface_test()
