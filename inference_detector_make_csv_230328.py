from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os 

import glob
import pandas as pd
import numpy as np

import cv2
import dicomsdl as dicom

def softmax(score_list):
    return (np.exp(score_list) / np.sum(np.exp(score_list)))


def load_dcm_and_date_time_dicomsdl(file_name, clahe=False):
    dcm_dictionary = dicom.open(file_name)
    data = dcm_dictionary.pixelData()
    if dcm_dictionary["PhotometricInterpretation"] == "MONOCHROME1":
        data = 1 - data
    # data = data.astype(np.uint8)

  
    data = (data - data.min()) / (data.max() - data.min())
    data = (data * 255).astype(np.uint8)

    if clahe == True:
        # print(data.shape)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        data = clahe.apply(data)
    # date = int(dcm_dictionary["ContentDate"])
    # time = float(dcm_dictionary["ContentTime"])
    data_3ch = np.stack([data,data,data], axis= 2)
    return data_3ch

config_file = '/root/home/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_binary_train_230324.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/root/home/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco_binary_train_230324/best_bbox_mAP_epoch_11.pth'
base_path = '/root/home/mmdetection/results_img'
img_path ='/root/home/mmdetection/data/vindr-cxr/train'


dir_name = config_file.split('/')[-1][:-3]

save_path = os.path.join(base_path, dir_name,'VinDR')
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


png_list = glob.glob(os.path.join(img_path, '*.png'))
jpg_list = glob.glob(os.path.join(img_path, '*.jpg'))
dcm_list = glob.glob(os.path.join(img_path, '*.dicom'))
val_list = png_list + jpg_list + dcm_list

## 첫번쨰 list는 class 설정
## 두번째 리스트에서 부터 4 point와 confidence 설정..
img_name = []
softmax_list= []

for img in mmcv.track_iter_progress(val_list):
    after_img = load_dcm_and_date_time_dicomsdl(img)
    img_name.append(img.split('/')[-1])
    result = inference_detector(model, after_img)

    max_confi_list = []

    for class_id, cls_list in enumerate(result):
        # print(cls_list)
        max_confi = 0
        for point_with_confi in cls_list:
            # print(class_id)
            # print(point_with_confi)
            max_confi = max(max_confi, point_with_confi[-1])

        # print(max_confi)
        max_confi_list.append(max_confi)
    
    # softmax(max_confi_list)
    softmax_list.append(softmax(max_confi_list))

##pred 값 나눠서 0, 1로 따로 리스트로 저장
preds_0_list = softmax_list[:,0]
preds_1_list = softmax_list[:,1]
# for a, b in softmax_list:
#     preds_0_list.append(a)
#     preds_1_list.append(b)




results_df = pd.DataFrame(
    {'img_name': img_name, 'preds_0': preds_0_list, 'preds_1' : preds_1_list})
results_df.to_csv('/root/home/mmdetection/data/vindr-cxr/faster_rcnn_preds.csv', header=True, index=False)