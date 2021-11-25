import base64
import json
import os
import time
import warnings
from io import BytesIO

import torch
import torchvision.transforms as transforms
import tornado
from PIL import Image
from torch.backends import cudnn

import model as models
from utils.display import ped_attr_dict

warnings.filterwarnings('ignore')

test_data_path = '/data2/face_data/RAP/RAP_dataset'
resume_path = 'checkpoint/rap_epoch_9.pth.tar'
attri_num = 51
test_path = 'test_data/CAM17_2014-02-20_20140220175154-20140220175854_tarid124_frame2893_line1.png'
save_path = 'test_data/save'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])


def prepare_model():
    # create model
    model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attri_num)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

    cudnn.benchmark = False
    cudnn.deterministic = True

    return model


pa_model = prepare_model()


# class base64_api(tornado.web.RequestHandler):
#     def initialize(self, gconf):
#         self.config = gconf
#         self.pool = gconf.get("threadpool", None)
#
#     @tornado.web.asynchronous
#     @tornado.gen.coroutine
#     def post(self, *aegs, **kwargs):
#         request = json.loads(self.request.body)
#         img_id = request.get('img_id', '')
#         base64_code = request.get('base64_code', '')
#
#         start_time = time.time()
#         try:
#             image = base64_to_pil(base64_code)
#             attri_dict = ped_attri(image, pa_model)
#             stat = True
#         except:
#             stat = False
#         end_time = time.time()
#
#         response = dict()
#         # response["pedestrian_attribute"] = dict()
#         if not stat:
#             response["message"] = '提取失败'
#         else:
#             response["message"] = '提取成功'
#             response["img_id"] = img_id
#             response["pedestrian_attribute"] = attri_dict
#         response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
#         print(response)
#         self.write(response)


def ped_attri(img_input, model):
    model.eval()

    input = transform_test(img_input)
    input = torch.unsqueeze(input, 0)
    # print("input.size() = {}".format(input.size()))
    input = input.cuda(non_blocking=True)

    # print("output = model(input)")
    output = model(input)

    # maximum voting
    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])

    output = torch.sigmoid(output.data).cpu().numpy()
    # print("output = {}".format(output))

    output_list = output[0].tolist()

    return ped_attr_dict(output_list)


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


if __name__ == '__main__':
    imag = Image.open(test_path).convert('RGB')
    base64_code = pil_to_base64(imag)
    print("base64 = {}".format(base64_code))
    # image = base64_to_pil(image)

    start_time = time.time()
    try:
        image = base64_to_pil(base64_code)
        attri_dict = ped_attri(image, pa_model)
        stat = True
    except:
        stat = False
    end_time = time.time()

    response = dict()
    # response["pedestrian_attribute"] = dict()
    if not stat:
        response["message"] = '提取失败'
    else:
        response["message"] = '提取成功'
        response["pedestrian_attribute"] = attri_dict
    response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
    print(response)
