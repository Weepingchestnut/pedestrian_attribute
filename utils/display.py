import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from infer_main import args
from utils.attri_dict import *

data_path = args.test_data_path
save_path = args.save_path


def show_attribute_img(img_name, attri_dict):
    # make not exist dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    img_path = os.path.join(data_path, img_name)
    print("img_path = {}".format(img_path))
    attr_list = list(attri_dict.keys())
    # print(attr_list)

    img = cv2.imread(img_path)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 字体格式设置
    font_style = ImageFont.truetype("./checkpoint/msht.ttf", size=15, encoding="utf-8")

    for index, attr in enumerate(attr_list):
        draw.text((0, 0 + index * 20), attr + ": " + attri_dict[attr], (178, 34, 34), font_style)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_path, img_name), img)
    print("save_path = {}".format(os.path.join(save_path, img_name)))
    print('~' * 100)


def color_choose(color_list, mode='shisuo'):
    if mode == 'shisuo':
        other = max(
            color_list[4],  # 绿
            color_list[5],  # 蓝
            color_list[6],  # 银
            color_list[8],  # 棕
            color_list[9],  # 紫
            color_list[10],  # 粉
            color_list[11],  # 橙
            color_list[13]  # 其他
        )
        return [color_list[0],  # 黑
                color_list[1],  # 白
                color_list[2],  # 灰
                color_list[3],  # 红
                color_list[7],  # 黄
                color_list[12],  # 混色
                other]
    elif mode == 'all':
        return color_list


def my_rap2_dict(output_list):
    # for i, value in enumerate(output_list):
    #     print("{}: {}".format(i, value))
    pa_dict = {
        '上衣类型': '未知',
        '上衣颜色': '未知',
        '下衣类型': '未知',
        '下衣颜色': '未知',
        '鞋子颜色': '未知',
        '是否背包': '未知'
    }
    ub_list = output_list[0:10]
    ub_color_list = output_list[10:24]
    # ub_color_list = color_choose(ub_color_list)
    lb_list = output_list[24:32]
    lb_color_list = output_list[32:46]
    # lb_color_list = color_choose(lb_color_list)
    shoes_color_list = output_list[46: 60]
    # shoes_color_list = color_choose(shoes_color_list)
    bag_list = output_list[60:62]
    # print("bag_list = {}".format(bag_list))
    if max(ub_list) < 0.5:
        pass
    else:
        pa_dict['上衣类型'] = rap2_ub_dict[ub_list.index(max(ub_list))]

    if max(ub_color_list) < 0.5:
        pass
    else:
        pa_dict['上衣颜色'] = rap2_color_dict[ub_color_list.index(max(ub_color_list))]

    if max(lb_list) < 0.5:
        pass
    else:
        pa_dict['下衣类型'] = rap2_lb_dict[lb_list.index(max(lb_list))]

    if max(lb_color_list) < 0.4:
        pass
    else:
        pa_dict['下衣颜色'] = rap2_color_dict[lb_color_list.index(max(lb_color_list))]

    if max(shoes_color_list) < 0.4:
        pass
    else:
        pa_dict['鞋子颜色'] = rap2_color_dict[shoes_color_list.index(max(shoes_color_list))]
    # bag = rap2_bag_dict[bag_list.index(max(bag_list))]
    if max(bag_list) > 0.5:
        pa_dict['是否背包'] = '是'
    else:
        pa_dict['是否背包'] = '否'

    return pa_dict


def my_rap2_tiny_dict(output_list):
    pa_dict = {
        '上衣类型': '未知',
        '上衣颜色': '未知',
        '下衣类型': '未知',
        '下衣颜色': '未知',
        '鞋子颜色': '未知',
        '是否背包': '未知'
    }
    ub_list = output_list[0:5]
    ub_color_list = output_list[5:11]
    lb_list = output_list[11:15]
    lb_color_list = output_list[15:18]
    shoes_color_list = output_list[18: 22]
    bag_list = output_list[22]

    if max(ub_list) < 0.5:
        pass
    else:
        pa_dict['上衣类型'] = rap2_ub_tiny_dict[ub_list.index(max(ub_list))]

    if max(ub_color_list) < 0.5:
        pass
    else:
        pa_dict['上衣颜色'] = rap2_ub_color_tiny_dict[ub_color_list.index(max(ub_color_list))]

    if max(lb_list) < 0.5:
        pass
    else:
        pa_dict['下衣类型'] = rap2_lb_tiny_dict[lb_list.index(max(lb_list))]

    if max(lb_color_list) < 0.4:
        pass
    else:
        pa_dict['下衣颜色'] = rap2_lb_color_tiny_dict[lb_color_list.index(max(lb_color_list))]

    if max(shoes_color_list) < 0.4:
        pass
    else:
        pa_dict['鞋子颜色'] = rap2_shoes_color_tiny_dict[shoes_color_list.index(max(shoes_color_list))]
    # bag = rap2_bag_dict[bag_list.index(max(bag_list))]
    if bag_list > 0.5:
        pa_dict['是否背包'] = '是'
    else:
        pa_dict['是否背包'] = '否'

    return pa_dict


def my_rap2_dict_F1(output_list):
    # for i, value in enumerate(output_list):
    #     print("{}: {}".format(i, value))
    pa_dict = {
        '上衣类型': '未知',
        '上衣颜色': '未知',
        '下衣类型': '未知',
        '下衣颜色': '未知',
        '鞋子颜色': '未知',
        '是否背包': '未知'
    }
    ub_list = output_list[0:10]
    ub_color_list = output_list[10:24]
    # ub_color_list = color_choose(ub_color_list)
    lb_list = output_list[24:32]
    lb_color_list = output_list[32:46]
    # lb_color_list = color_choose(lb_color_list)
    shoes_color_list = output_list[46: 60]
    # shoes_color_list = color_choose(shoes_color_list)
    bag_list = output_list[60:62]
    # print("bag_list = {}".format(bag_list))
    F1_list = [0] * 61

    if max(ub_list) < 0.5:
        for i in range(0, 10):
            F1_list[i] = -1
    else:
        ub_list_sort = sorted(ub_list)  # 排序后ub_list
        ub_list_index = ub_list.index(max(ub_list))
        pa_dict['上衣类型'] = rap2_ub_dict[ub_list_index]
        F1_list[ub_list_index] = 1
        # 取置信度第二的属性也赋1
        if ub_list_sort[-2] > 0.5:
            F1_list[ub_list.index(ub_list_sort[-2])] = 1

    if max(ub_color_list) < 0.5:
        for i in range(10, 24):
            F1_list[i] = -1
    else:
        ub_color_list_index = ub_color_list.index(max(ub_color_list))
        pa_dict['上衣颜色'] = rap2_color_dict[ub_color_list_index]
        F1_list[10 + ub_color_list_index] = 1

    if max(lb_list) < 0.5:
        for i in range(24, 32):
            F1_list[i] = -1
    else:
        lb_list_index = lb_list.index(max(lb_list))
        pa_dict['下衣类型'] = rap2_lb_dict[lb_list_index]
        F1_list[24 + lb_list_index] = 1

    if max(lb_color_list) < 0.4:
        for i in range(32, 46):
            F1_list[i] = -1
    else:
        lb_color_list_index = lb_color_list.index(max(lb_color_list))
        pa_dict['下衣颜色'] = rap2_color_dict[lb_color_list_index]
        F1_list[32 + lb_color_list_index] = 1

    if max(shoes_color_list) < 0.4:
        for i in range(46, 60):
            F1_list[i] = -1
    else:
        shoes_color_list_index = shoes_color_list.index(max(shoes_color_list))
        pa_dict['鞋子颜色'] = rap2_color_dict[shoes_color_list_index]
        F1_list[46 + shoes_color_list_index] = 1
    # bag = rap2_bag_dict[bag_list.index(max(bag_list))]
    if max(bag_list) > 0.5:
        pa_dict['是否背包'] = '是'
        F1_list[60] = 1
    else:
        pa_dict['是否背包'] = '否'

    # print("F1_list = {}".format(F1_list))
    return pa_dict, F1_list


# print confidence
def my_rap2_dict_c(output_list):
    # for i, value in enumerate(output_list):
    #     print("{}: {}".format(i, value))
    pa_dict = {
        '上衣类型': '未知',
        '上衣颜色': '未知',
        '下衣类型': '未知',
        '下衣颜色': '未知',
        '鞋子颜色': '未知',
        '是否背包': '未知'
    }
    ub_list = output_list[0:10]
    ub_color_list = output_list[10:24]
    # ub_color_list = color_choose(ub_color_list)
    lb_list = output_list[24:32]
    lb_color_list = output_list[32:46]
    # lb_color_list = color_choose(lb_color_list)
    shoes_color_list = output_list[46: 60]
    # shoes_color_list = color_choose(shoes_color_list)
    bag_list = output_list[60:62]
    # print("bag_list = {}".format(bag_list))
    if max(ub_list) < 0.5:
        pa_dict['上衣类型'] = pa_dict['上衣类型'] + ' ' + format(max(ub_list), '.4f') + rap2_ub_dict[ub_list.index(max(ub_list))]
        # pass
    else:
        pa_dict['上衣类型'] = rap2_ub_dict[ub_list.index(max(ub_list))] + ' ' + format(max(ub_list), '.4f')

    if max(ub_color_list) < 0.5:
        pa_dict['上衣颜色'] = pa_dict['上衣颜色'] + ' ' + format(max(ub_color_list), '.4f') + rap2_color_dict[ub_color_list.index(max(ub_color_list))]
        # pass
    else:
        pa_dict['上衣颜色'] = rap2_color_dict[ub_color_list.index(max(ub_color_list))] + ' ' + format(max(ub_color_list), '.4f')

    if max(lb_list) < 0.5:
        pa_dict['下衣类型'] = pa_dict['下衣类型'] + ' ' + format(max(lb_list), '.4f') + rap2_lb_dict[lb_list.index(max(lb_list))]
        # pass
    else:
        pa_dict['下衣类型'] = rap2_lb_dict[lb_list.index(max(lb_list))] + ' ' + format(max(lb_list), '.4f')

    if max(lb_color_list) < 0.4:
        pa_dict['下衣颜色'] = pa_dict['下衣颜色'] + ' ' + format(max(lb_color_list), '.4f') + rap2_color_dict[lb_color_list.index(max(lb_color_list))]
        # pass
    else:
        pa_dict['下衣颜色'] = rap2_color_dict[lb_color_list.index(max(lb_color_list))] + ' ' + format(max(lb_color_list), '.4f')

    if max(shoes_color_list) < 0.4:
        pa_dict['鞋子颜色'] = pa_dict['鞋子颜色'] + ' ' + format(max(shoes_color_list), '.4f') + rap2_color_dict[shoes_color_list.index(max(shoes_color_list))]
        # pass
    else:
        pa_dict['鞋子颜色'] = rap2_color_dict[shoes_color_list.index(max(shoes_color_list))] + ' ' + format(max(shoes_color_list), '.4f')
    # bag = rap2_bag_dict[bag_list.index(max(bag_list))]
    if max(bag_list) > 0.5:
        pa_dict['是否背包'] = '是' + ' ' + format(max(bag_list), '.4f')  # + ' ('+bag+')'
    else:
        pa_dict['是否背包'] = '否' + ' ' + format(max(bag_list), '.4f')

    return pa_dict


def ped_attr_dict(output_list):
    # max_index存储每大类属性中置信度值最大属性的位置信息
    pa_dict = {'性别': '',
               '年龄': '',
               '体型': '',
               '头肩': '',
               '上衣': '',
               '下衣': '',
               '鞋子类型': '',
               '附属物': '',
               '行为': ''}

    age_list = output_list[1:4]
    body_list = output_list[4:7]
    hs_list = output_list[12:15]
    up_list = output_list[15:24]
    lb_list = output_list[24:30]
    shoes_list = output_list[30:35]
    attach_list = output_list[35:43]
    action_list = output_list[43:51]

    # 性别判断，1: female, 0: male
    if output_list[0] > .5:
        pa_dict['性别'] = gender_dict[1]
    else:
        pa_dict['性别'] = gender_dict[0]

    pa_dict['年龄'] = age_dict[age_list.index(max(age_list))]
    pa_dict['体型'] = body_dict[body_list.index(max(body_list))]

    max_hs_list = max(hs_list)
    if max_hs_list > .6:
        pa_dict['头肩'] = hs_dict[hs_list.index(max_hs_list)]
    else:
        pa_dict['头肩'] = hs_dict[3]

    pa_dict['上衣'] = up_dict[up_list.index(max(up_list))]
    pa_dict['下衣'] = lb_dict[lb_list.index(max(lb_list))]
    pa_dict['鞋子类型'] = shoes_dict[shoes_list.index(max(shoes_list))]
    pa_dict['附属物'] = attach_dict[attach_list.index(max(attach_list))]

    max_action_list = max(action_list)
    if max_action_list > .5:
        pa_dict['行为'] = action_dict[action_list.index(max(action_list))]
    else:
        pa_dict['行为'] = action_dict[8]

    return pa_dict
