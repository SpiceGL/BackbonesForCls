# -*- coding: gbk -*-
from argparse import ArgumentParser
import os
import sys
sys.path.insert(0,os.getcwd())
import torch
import cv2
import numpy as np

from utils.inference import inference_model, init_model
from core.visualization.image import imshow_infos
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='classes map of datasets')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    parser.add_argument('--show', action='store_true', help='Show video')
    args = parser.parse_args()

    classes_names, _ = get_info(args.classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg,train_pipeline,val_pipeline,no_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    
    cap = cv2.VideoCapture(args.video)
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(
            args.save_path, fourcc, fps,
            size)

    #��֡������
    c = 1
    timeF = 1

    #ƽ���������
    #����3֡ͼ����Ϊһ�࣬�򻭾��Σ���ֹ����
    count_wrong = 0
    count_normal = 0

    #������ֵ
    threshold = 0.5

    while True:
        flag, frame = cap.read()
        # print("flag: \n", flag)
        if not flag:
            break

        ###��ת
        w, h = frame.shape[1], frame.shape[0]
        length_new = round(np.sqrt(np.square(frame.shape[0]) + np.square(frame.shape[1])))
        w_new, h_new = length_new, length_new
        if (w_new - w) % 2 != 0:
            w_new += 1
        if (h_new - h) % 2 != 0:
            h_new += 1
        frame = cv2.copyMakeBorder(frame, (h_new - h) // 2, (h_new - h) // 2, (w_new - w) // 2, (w_new - w) // 2,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        warp_mat = cv2.getRotationMatrix2D((w_new / 2, h_new / 2), 40, 1.0)
        frame = cv2.warpAffine(frame, warp_mat, (w_new, h_new))

        # print("n=== \n")
        b = c % timeF
        # print("b= \n", b)

        img_copy = frame.copy()
        # cv2.namedWindow("display", cv2.WINDOW_AUTOSIZE)
        # img_cut = img[t:b, l:r]  # [h,w]
        # img_cut = img_copy[194:378, 166:591]  #��������ʶ��  #left_top=(166,194), right_bottle=(591,378)
        # img_cut = img_copy[184:368, 106:531]
        # img_cut = img_copy[197:384, 135:557] ##z106�������������ʶ��
        # img_cut = img_copy[273:477, 161:583]
        # img_cut = img_copy[87:478, 18:639] ##ZTC1000V663-1
        img_cut0 = img_copy[1114:1543, 810:1092]  ##zhuqi
        img_cut1 = img_copy[1066:1630, 1272:1684]  ##zhuqi

        #��֡���
        if (b == 0):
            # print("res= \n", c%timeF)
            # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            result0 = inference_model(model, img_cut0, val_pipeline, classes_names)

            result1 = inference_model(model, img_cut1, val_pipeline, classes_names)



            # ��ֵ
            if float(result0["pred_score"]) or float(result1["pred_score"]) > 0.8:
                #������
                print("result0= \n", result0)
                print("result1= \n", result1)
                if result0['pred_class'] == 'Wrong':
                        cv2.rectangle(frame, (810, 1114), (1092, 1543), (0, 0, 255), 2)  # �� zhuqi
                else:
                        cv2.rectangle(frame, (810, 1114), (1092, 1543), (0, 255, 0), 2)  # �� zhuqi

                if result1['pred_class'] == 'Wrong':
                        cv2.rectangle(frame, (1272, 1066), (1684, 1630), (0, 0, 255), 2)  # �� zhuqi
                else:
                        cv2.rectangle(frame, (1272, 1066), (1684, 1630), (0, 255, 0), 2)  # �� zhuqi

                # put the results to img
                # cv2.imshow('video_frame', frame)
                img = imshow_infos(frame, result0, show=False)

                if args.show:
                    cv2.namedWindow('video', 0)
                    cv2.imshow('video', img)

                if args.save_path:
                    video_writer.write(img)

    # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        c = c + 1
    print("m==== \n")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
