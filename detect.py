import torch
from PIL import Image
from PIL import ImageDraw,ImageFont,ImageFilter
import numpy as np
from tool import utils
import nets
from torchvision import transforms
import time
import os

class Detector:

    def __init__(self, pnet_param="./param/pnet.pt", rnet_param="./param/rnet.pt", onet_param="./param/onet.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))



        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))
        del pnet_boxes,rnet_boxes
        return onet_boxes

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)-0.5
            _img_dataset.append(img_data)

            del _x1,_y1,_x2,_y2,img,img_data
        img_dataset =torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.6)
        # for idx in idxs:
        #     _box = _pnet_boxes[idx]
        #     _x1 = int(_box[0])
        #     _y1 = int(_box[1])
        #     _x2 = int(_box[2])
        #     _y2 = int(_box[3])
        #
        #     ow = _x2 - _x1
        #     oh = _y2 - _y1
        #
        #     x1 = _x1 + ow * offset[idx][0]
        #     y1 = _y1 + oh * offset[idx][1]
        #     x2 = _x2 + ow * offset[idx][2]
        #     y2 = _y2 + oh * offset[idx][3]
        #
        #     boxes.append([x1, y1, x2, y2, cls[idx][0]])
        # del _box,_x1,_y1,_x2,_y2,ow,oh,x1,y1,x2,y2
        _box = _pnet_boxes[idxs]
        _x1 = np.array(_box[:, 0],dtype=np.int)
        _y1 = np.array(_box[:, 1],dtype=np.int)
        _x2 = np.array(_box[:, 2],dtype=np.int)
        _y2 = np.array(_box[:, 3],dtype=np.int)
        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]
        boxes = np.stack([x1, y1, x2, y2, cls[idxs][:,0]], axis=1)
        return utils.nms(np.array(boxes), 0.6)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)-0.5
            _img_dataset.append(img_data)

            del _x1,_y1,_x2,_y2,img,img_data
        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.9999)
        # for idx in idxs:
        #     _box = _rnet_boxes[idx]
        #     _x1 = int(_box[0])
        #     _y1 = int(_box[1])
        #     _x2 = int(_box[2])
        #     _y2 = int(_box[3])
        #
        #     ow = _x2 - _x1
        #     oh = _y2 - _y1
        #
        #     x1 = _x1 + ow * offset[idx][0]
        #     y1 = _y1 + oh * offset[idx][1]
        #     x2 = _x2 + ow * offset[idx][2]
        #     y2 = _y2 + oh * offset[idx][3]
        #
        #
        #     boxes.append([x1, y1, x2, y2, cls[idx][0]])
        #     del _box,_x1,_y1,_x2,_y2,ow,oh,x1,y1,x2,y2
        _box = _rnet_boxes[idxs]
        _x1 = np.array(_box[:, 0], dtype=np.int)
        _y1 = np.array(_box[:, 1], dtype=np.int)
        _x2 = np.array(_box[:, 2], dtype=np.int)
        _y2 = np.array(_box[:, 3], dtype=np.int)
        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        boxes = np.stack([x1, y1, x2, y2, cls[idxs][:, 0]], axis=1)
        # return np.array(boxes)
        return utils.nms(np.array(boxes), 0.7, isMin=True)

    def __pnet_detect(self, image):
        boxes = torch.tensor([])
        img = image

        scale = 1
        w, h = img.size
        # w = int(w * scale)
        # h = int(h * scale)
        #
        # img = img.resize((w, h))
        min_side_len = min(w, h)

        while min_side_len > 12:
            img_data = self.__image_transform(img)-0.5
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)

            _cls, _offest = self.pnet(img_data)

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.5))

            # for idx in idxs:
            #     boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))
            boxes = torch.cat((boxes,self.__box(idxs, offest, cls[idxs[:,0], idxs[:,1]], scale)))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

            del img_data,_cls, _offest,cls, offest,idxs,_w,_h
        return utils.nms(np.array(boxes), 0.6)

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = (start_index[:,1].float() * stride) / scale
        _y1 = (start_index[:,0].float() * stride) / scale
        _x2 = (start_index[:,1].float() * stride + side_len) / scale
        _y2 = (start_index[:,0].float() * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[:,0], start_index[:,1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return torch.stack([x1, y1, x2, y2, cls],dim=1)


if __name__ == '__main__':
    # 正常输出
    # image_file = "img/single/2.jpg"
    image_file = "img9.jpg"
    detector = Detector()
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
    with Image.open(image_file) as im:
        # boxes = detector.detect(im)
        # print("----------------------------")
        boxes = detector.detect(im)

        print(im.size)
        imDraw = ImageDraw.Draw(im)

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2*0.85), outline='red')
            imDraw.text((x1+3,y1+3),fill=(0,255,0),text=str(box[4].round(8)),font=font)
        # im.save('F:/img/16.jpg')
        im.show()