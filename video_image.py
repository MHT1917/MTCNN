import cv2
import os
from PIL import Image
from detect import Detector
def video_img():
    detector = Detector()
    vc=cv2.VideoCapture(r'F:\video\mv.mp4')
    c=1
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False
    while rval:
        rval,frame=vc.read()
        im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


        boxes = detector.detect(im)

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            _y2 = int(y2 * 0.8+y1*0.2)
            cv2.rectangle(frame,(x1, y1), (x2, _y2), (0,225,0),2)
        cv2.imshow('img',frame)
        # cv2.imwrite(r'F:\video\img\img'+str(c)+'.jpg',frame)
        c=c+1
        cv2.waitKey(1)
    vc.release()

def img_video():
    fps = 24
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_writer = cv2.VideoWriter(filename=r'C:\Users\A\Music\MV\result.avi', fourcc=fourcc, fps=fps, frameSize=(800, 600))
    for i in range(100, 900):
        p = i
        if os.path.exists(r'C:\Users\A\Music\MV\img\img' + str(p) + '.jpg'):  # 判断图片是否存在
            img = cv2.imread(filename=r'C:\Users\A\Music\MV\img\img' + str(p) + '.jpg')
            cv2.waitKey(1)
            video_writer.write(img)
            print(str(p) + '.jpg' + ' done!')
    video_writer.release()

if __name__ == '__main__':
    video_img()
    # img_video()