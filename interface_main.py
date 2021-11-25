import argparse
import base64
import json
import time
import warnings
from io import BytesIO

import torch
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
parser.add_argument('--test_data_path', default='test_data/rap_test/RAP_dataset', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('-s', '--show', dest='show', action='store_true', required=False, help='show attribute in imag')
parser.add_argument('--save_path', default='work_dir/my_rap2_output_img', type=str, required=False,
                    help='(default=%(default)s)')

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


class base64_api(tornado.web.RequestHandler):
    def initialize(self, gconf):
        self.config = gconf
        self.pool = gconf.get("threadpool", None)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *aegs, **kwargs):
        request = json.loads(self.request.body)
        img_id = request.get('img_id', '')
        base64_code = request.get('base64_code', '')

        response = dict()
        response["pedestrian_attribute"] = dict()
        # 图片id为空（字符串长度为0 or 字符串均为空格）
        if len(str(img_id)) == 0 or str(img_id).isspace is True:
            response["pedestrian_attribute"]["success"] = False
            response["pedestrian_attribute"]["message"] = '图片id为空'
            response["pedestrian_attribute"]["img_id"] = img_id
            response["pedestrian_attribute"]["attribute"] = ''
            response["spendTime"] = "0 s"
        if not isBase64(base64_code):
            response["pedestrian_attribute"]["success"] = False
            # response["pedestrian_attribute"]["img_code"] = base64_code
            response["pedestrian_attribute"]["message"] = '图片损坏/非base64编码'
            response["pedestrian_attribute"]["img_id"] = img_id
            response["pedestrian_attribute"]["attribute"] = ''
            response["spendTime"] = "0 s"
        else:
            start_time = time.time()
            try:
                image = base64_to_pil(base64_code)
                attri_dict = ped_attr(image)
                stat = True
            except:
                stat = False
            end_time = time.time()

            # response["pedestrian_attribute"] = dict()
            if not stat:
                response["pedestrian_attribute"]["message"] = '属性提取失败'
            else:
                response["pedestrian_attribute"]["message"] = '属性提取成功'
                response["pedestrian_attribute"]["img_id"] = img_id
                response["pedestrian_attribute"]["attribute"] = attri_dict
            response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
        print(response)
        self.write(response)


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


def interface_test(td_path):
    imag = Image.open(td_path).convert('RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    print("base64 = {}".format(base64_code))

    response = dict()
    if not isBase64(base64_code):
        response["pedestrian_attribute"] = dict()
        response["pedestrian_attribute"]["success"] = False
        response["pedestrian_attribute"]["img_code"] = base64_code
        response["pedestrian_attribute"]["message"] = '图片损坏/非base64编码'
        response["spendTime"] = "0 s"
    else:
        start_time = time.time()
        try:
            image = base64_to_pil(base64_code)
            attri_dict = ped_attr(image)
            stat = True
        except:
            stat = False
        end_time = time.time()

        response["pedestrian_attribute"] = dict()
        if not stat:
            response["pedestrian_attribute"]["message"] = '属性提取失败'
        else:
            response["pedestrian_attribute"]["message"] = '属性提取成功'
            response["pedestrian_attribute"]["img_id"] = 'img_test'
            response["pedestrian_attribute"]["attribute"] = attri_dict
        response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
    print(response)


if __name__ == '__main__':
    interface_test("test_data/CAM17_2014-02-20_20140220175154-20140220175854_tarid124_frame2893_line1.png")
