import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
<<<<<<< HEAD
=======
from mmdet.core import get_classes
>>>>>>> Worm_Detection
import cv2
import numpy as np
import os
cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None
list_path = os.listdir("Image_test/")
<<<<<<< HEAD
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'work_dirs/latest.pth')

# test a single image
for i in list_path:
  if i.endswith(".jpg"):
    print("################%s################" %i)
    # 
    img = mmcv.imread("Image_test/" + i)
    result = inference_detector(model, img, cfg)
    print(result)
    for Check_result in result:
      if len(Check_result) != 0:
        for wrong_result in Check_result:
          if wrong_result[4] < 0.8 : 
              wrong_result[4] = 0
 
    show_result(img, result)
=======
path_video = "Video_Test/1.wmv"
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'work_dirs/latest.pth')


# Test Video
cap = cv2.VideoCapture(path_video)
while True:
    ret, frame = cap.read()
    if ret:
        if frame is None:
            break
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        
        # fgMask = backSub.apply(gray_frame)
        
        # cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fgMask)
        
        result = inference_detector(model, frame, cfg)
        # for Check_result in result:
        #   if len(Check_result) != 0:
        #     for wrong_result in Check_result:
        #       if wrong_result[4] < 0.8 : 
        #           wrong_result[4] = 0
            
        # #print(result)
        # # show_result(frame, result)
        # class_names = get_classes("coco")
        # list_labels = [
        #     np.full(bbox.shape[0], i, dtype=np.int32)
        #     for i, bbox in enumerate(result)
        # ]
        # print(list_labels)
        # labels = np.concatenate(list_labels)
        # bboxes = np.vstack(result)
        # print(labels)
        # print(bboxes)
        class_names = get_classes("coco")
        lst_bboxes = []
        lst_lable = [] 
        for i,data in enumerate(result):
            if len(data) != 0:
                labels = np.full(data.shape[0], i, dtype=np.int32)
                for j,x in enumerate(data): 
                    if x[-1] > 0.8:
                        lst_bboxes.append(np.vstack(x))
                        lst_lable.append(labels[j])
        print(lst_bboxes)
        print(lst_lable)

        for index in range(len(lst_bboxes)):
            bbox = tuple(map(tuple,lst_bboxes[index]))
            cv2.rectangle(frame,bbox[0]+bbox[1] ,bbox[2]+bbox[3],(0,255,0),4)
            cv2.putText(frame,class_names[lst_lable[index]],bbox[0]+bbox[1],cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
       
        cv2.imshow('Frame', frame)
        # mmcv.imshow_det_bboxes(
        #     frame.copy(),
        #     bboxes,
        #     labels,
        #     class_names=class_names,
        #     score_thr=0.3)

        keyboard = cv2.waitKey(10)
        if keyboard == 'q' or keyboard == 27:
            break
    else:
        break
# cv2.waitKey(500)
cv2.destroyAllWindows()



# # test a single image
# for i in list_path:
#   if i.endswith(".jpg"):
#     print("################%s################" %i)
#     # 
#     img = mmcv.imread("Image_test/" + i)
#     result = inference_detector(model, img, cfg)

#     for Check_result in result:
#       if len(Check_result) != 0:
#         for wrong_result in Check_result:
#           if wrong_result[4] < 0.8 : 
#               wrong_result[4] = 0
          
#     #print(result)
#     show_result(img, result)
>>>>>>> Worm_Detection
