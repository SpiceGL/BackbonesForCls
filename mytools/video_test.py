# -*- coding: gbk -*-
from argparse import ArgumentParser
import os
import sys
sys.path.insert(0,os.getcwd())
import torch
import cv2

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

    #隔帧检测参数
    c = 1
    timeF = 1

    #平滑处理参数
    #连续3帧图像检测为一类，则画矩形，防止跳变
    count_wrong = 0
    count_normal = 0

    #设置阈值
    threshold = 0.5

    while True:
        flag, frame = cap.read()
        # print("flag: \n", flag)
        if not flag:
            break

        # print("n=== \n")
        b = c % timeF
        # print("b= \n", b)

        img_copy = frame.copy()
        # cv2.namedWindow("display", cv2.WINDOW_AUTOSIZE)
        # img_cut = img[t:b, l:r]  # [h,w]
        # img_cut = img_copy[194:378, 166:591]  #卷扬区域识别  #left_top=(166,194), right_bottle=(591,378)
        # img_cut = img_copy[184:368, 106:531]
        # img_cut = img_copy[197:384, 135:557] ##z106用于阴天情况的识别
        # img_cut = img_copy[273:477, 161:583]
        img_cut = img_copy[87:478, 18:639] ##ZTC1000V663-1

        #隔帧检测
        if (b == 0):
            # print("res= \n", c%timeF)
            # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            result = inference_model(model, img_cut, val_pipeline, classes_names)

            # result = inference_model(model, frame, val_pipeline, classes_names)

        # get single test results
        # result = inference_model(model, frame, val_pipeline, classes_names)

            # 阈值
            if float(result["pred_score"]) > 0.8:
                #画矩形
                print("result= \n", result)
                if result['pred_class'] == 'Wrong':
                    # count_wrong = count_wrong + 1
                    # count_normal = 0
                    # if (count_wrong >= 3 and count_wrong != 0):
                    #     cv2.rectangle(frame, (165, 192), (590, 376), (0, 0, 255), 2) #红
                        # cv2.rectangle(frame, (135, 197), (557, 384), (0, 0, 255), 2)  # 红 ##z106用于阴天情况的识别
                    #     cv2.rectangle(frame, (161,273),(583,477), (0, 0, 255), 2)  # 红
                        cv2.rectangle(frame, (18, 87), (639, 478), (0, 0, 255), 2)  # 红
                    # else:
                    #     pass


                else:
                    # count_normal = count_normal + 1
                    # count_wrong = 0
                    # if (count_normal >= 3 and count_normal != 0):
                    #     cv2.rectangle(frame, (165, 192), (590, 376), (0, 255, 0), 2)
                        # cv2.rectangle(frame, (135, 197), (557, 384), (0, 255, 0), 2)  # 红 ##z106用于阴天情况的识别
                    #     cv2.rectangle(frame, (161,273),(583,477), (0, 255, 0), 2)
                        cv2.rectangle(frame, (18, 87), (639, 478), (0, 255, 0), 2)  # 红
                    # else:
                    #     pass


                # put the results to img
                cv2.imshow('video_frame', frame)
                img = imshow_infos(frame, result, show=False)

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
