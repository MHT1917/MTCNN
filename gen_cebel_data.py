import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

anno_src = r"E:\DataSets\CelebA\Anno\list_bbox_celeba.txt"
img_dir = r"E:\DataSets\CelebA\Img\img_celeba"

save_path = r"F:\celeba1"


for face_size in [12,24,48]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)
                    iou_max = min(w, h)/max(w, h)

                    px1 = 0#float(strs[5].strip())
                    py1 = 0#float(strs[6].strip())
                    px2 = 0#float(strs[7].strip())
                    py2 = 0#float(strs[8].strip())
                    px3 = 0#float(strs[9].strip())
                    py3 = 0#float(strs[10].strip())
                    px4 = 0#float(strs[11].strip())
                    py4 = 0#float(strs[12].strip())
                    px5 = 0#float(strs[13].strip())
                    py5 = 0#float(strs[14].strip())

                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    single_positive = 0
                    single_negative = 0
                    single_part = 0
                    while True:
                        if single_positive==5 and single_part==5:
                            break
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.2, w * 0.2)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        offset_px1 = 0#(px1 - x1_) / side_len
                        offset_py1 = 0#(py1 - y1_) / side_len
                        offset_px2 = 0#(px2 - x1_) / side_len
                        offset_py2 = 0#(py2 - y1_) / side_len
                        offset_px3 = 0#(px3 - x1_) / side_len
                        offset_py3 = 0#(py3 - y1_) / side_len
                        offset_px4 = 0#(px4 - x1_) / side_len
                        offset_py4 = 0#(py4 - y1_) / side_len
                        offset_px5 = 0#(px5 - x1_) / side_len
                        offset_py5 = 0#(py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.7 or iou > (iou_max-0.05):  # 正样本
                            if single_positive == 5:
                                continue

                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            single_positive += 1
                            positive_count += 1
                        elif iou > 0.25 and iou < 0.45:  # 部分样本
                            if single_part == 5:
                                continue

                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1,offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            single_part += 1
                            part_count += 1
                        # elif iou < 0.3:
                        #     negative_anno_file.write(
                        #         "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                        #     negative_anno_file.flush()
                        #     face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        #     negative_count += 1

                    # 生成负样本
                    _boxes = np.array(boxes)
                    freq = 0
                    while True:
                        if single_negative == 15:
                            break
                        if freq > 99:
                            break
                        freq += 1
                        if (face_size > min(img_w, img_h) / 2):
                            break
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.05:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            single_negative += 1
                            negative_count += 1
                            freq = 0
            except Exception as e:
                traceback.print_exc()


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
