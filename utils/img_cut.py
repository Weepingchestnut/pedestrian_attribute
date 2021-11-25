import os
import datetime

import cv2
from tqdm import tqdm

print("------图像裁剪------")
data_path = '/home/lzk/data/face_data/RAP2/RAP_dataset'
save_path_ub = '/home/lzk/workspace/pedestrain_attribute/iccv19_attribute/work_dir/rapv2_cut_img_ub'
save_path_lb = '/home/lzk/workspace/pedestrain_attribute/iccv19_attribute/work_dir/rapv2_cut_img_lb'


# get img information
def get_img_info(img_path):
    img = cv2.imread(img_path)
    H, W, C = img.shape
    # print("img_path = {}  [{}]".format(img_path, str(img.shape)))
    return C, H, W, img


# 裁剪图片
def cut_img():
    # img_path = '../test_data/CAM17_2014-02-20_20140220175154-20140220175854_tarid124_frame2893_line1.png'
    # make img_name list
    img_names = []
    for file in os.listdir(data_path):
        # 判断是否为空图像，跳过空图像，防止后续读取图像是报错
        if os.stat(os.path.join(data_path, file)).st_size == 0:
            pass
        else:
            img_names.append(file)

    for index, img_name in tqdm(enumerate(img_names), total=len(img_names)):
        # print(img_name)
        # print(img_name.split('.')[0])
        img_path = os.path.join(data_path, img_name)
        C, H, W, img = get_img_info(img_path)
        # if (H & 1) == 0:
        #     pass
        #     # 偶数
        # else:
        #     # 奇数    需要去除最后一行像素，否则均匀裁剪分会有问题
        #     for i in range(0, H - 1, H - 1):
        #         for j in range(0, W, W):
        #             img = img[i, j]
        #     pass
        # cut_h = (H - 1) // 2
        cut_h = H // 2
        cut_w = W
        count = 0
        # print("cut_h = {}, cut_w = {}".format(cut_h, cut_w))

        for i in range(0, H, cut_h):
            for j in range(0, W, cut_w):
                result = img[i:i + cut_h, j:j + cut_w]
                if result.shape[0] == 1:
                    # print("result.shape[0] = {}".format(result.shape[0]))
                    pass
                else:
                    if count == 0:
                        save_name = img_name.split('.')[0] + "_%d.png" % count
                        # save_name = "%d.jpg" % count
                        cv2.imwrite(os.path.join(save_path_ub, save_name), result)
                        # print("count = {}".format(count))
                        count += 1
                    else:
                        save_name = img_name.split('.')[0] + "_%d.png" % count
                        cv2.imwrite(os.path.join(save_path_lb, save_name), result)
                        count += 1
                    # print("save_name = {}  [{}]".format(save_name, result.shape))
        # print("~" * 200)


def cut_label():
    pass


if __name__ == '__main__':
    cut_img()
