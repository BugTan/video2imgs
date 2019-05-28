# coding=utf-8

import os
import cv2
import dlib
import numpy

videos_src_path = "/home/bugtan/code/vid2vid/datasets/face/test_img/"
video_formats = [".MP4", ".MOV"]
frames_save_path = "/home/bugtan/code/vid2vid/datasets/face/test_img/0005/"
landmarkSaveDir = '/home/bugtan/code/vid2vid/datasets/face/test_keypoints/0005/'
predictorPath = '/home/bugtan/code/vid2vid/datasets/face/shape_predictor_68_face_landmarks.dat'


    # 将视频按固定间隔读取写入图片
    # :param video_src_path: 视频存放路径
    # :param formats:　包含的所有视频格式
    # :param frame_save_path:　保存路径
    # :param frame_width:　保存帧宽
    # :param frame_height:　保存帧高
    # :param interval:　保存帧间隔
    # :return:　帧图片


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorPath)

cap = cv2.VideoCapture(videos_src_path+'destVideo.mov')
frame_index = 0
if cap.isOpened():
    success = True
else:
    success = False
    print("读取失败!")

while(success):
    success, frame = cap.read()
    if success:
        print("---> 正在读取第%d帧:" % (frame_index+1), success)

        cv2.imwrite( frames_save_path + str(frame_index).zfill(5)+'.jpg', frame)
        dets = detector(frame,1)

        fidImgLandmarkTxt = open(landmarkSaveDir+str(frame_index).zfill(5)+'.txt','w')
        for k,d in enumerate(dets):
            shape = predictor(frame, d)
            landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
            for p in shape.parts():
                fidImgLandmarkTxt.write(str(p.x)+','+str(p.y)+'\n')
        fidImgLandmarkTxt.close()
        frame_index = frame_index + 1
    else:
        break


cap.release()


