import os
import cv2
import sys
import onnx
import onnxsim
import onnxruntime as ort

sys.path.insert(0, os.getcwd())
import argparse

import copy
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score
import matplotlib.pyplot as plt
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

from myutils.dataloader import Mydataset, collate
from myutils.train_utils import get_info, netcfg2dict, pipelinecfg2dict
from mycfg.build_for_export import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--config', default="mycfg/models/mobilenet_v3_small.py", help='train config file path')
    parser.add_argument('--device', default="cpu", help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_cfg, data_cfg, lr_config, optimizer_cfg = netcfg2dict(args.config)
    train_pipeline, val_pipeline, no_pipeline, net_size = pipelinecfg2dict("mycfg/pipeline.py")
    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_map = 'mydatas/annotations.txt'
    test_annotations = 'mydatas/test2.txt'
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f:
        test_datas = f.readlines()
    
    """
    生成模型、加载权重
    """
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])
    model = init_model(model, data_cfg, device=device, mode='eval')

    ### pth2onnx
    print("======================= .pth 2 .onnx... ========================")
    input_names = ["x"]
    output_names = ["y"]
    inp = torch.randn(1, 3, net_size[0], net_size[1])  ##(1,3,h,w)
    pth_path = data_cfg.get("test")["ckpt"]
    onnx_path = os.path.splitext(pth_path)[0] + ".onnx"
    onnxsim_path = os.path.splitext(pth_path)[0] + "_sim.onnx"
    print(onnx_path)
    torch.onnx.export(model, inp, onnx_path, opset_version=12, verbose=True, input_names=input_names,
                      output_names=output_names)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    onnxsim_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnxsim_model, onnxsim_path)
    else:
        print("export onnxsim model failed!")
    ### pth2pt
    print("======================= .pth 2 .pt... ========================")
    pt_path = os.path.splitext(pth_path)[0] + ".pt"
    mod = torch.jit.trace(model, inp)
    torch.jit.save(mod, pt_path)
    #---
    
    ###比较pytorch和onnx的推理结果
    image_path = r"E:\Datasets\YOLOv8_cls\Windings\val2\Normal\Z106_Normal_Fri_May_26_07_16_11_2023_00000.jpg"
    x = cv2.imread(image_path)
    x = cv2.resize(x, (net_size[1], net_size[0]))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0
    x = x.astype("float32")
    print(x[100:102, 160:165, 2])
    x = np.expand_dims(x, axis=0)
    in_onnx = np.transpose(x, (0, 3, 1, 2))
    # print(in_onnx[0][2)
    in_tensor = torch.from_numpy(in_onnx)
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(onnx_path)

    onnxsim_model = onnx.load(onnxsim_path)
    onnx.checker.check_model(onnxsim_model)
    ortsim_sess = ort.InferenceSession(onnxsim_path)
    
    res_tensor = model(in_tensor)
    print("torch model result:", res_tensor)
    try:
        outputs = ort_sess.run(["y"], {'x': in_onnx})
        res_onnx = outputs[0]
        print("onnx model result:", res_onnx)
    except:
        pass
    try:
        outputs = ortsim_sess.run(["y"], {'x': in_onnx})
        res_onnx = outputs[0]
        print("onnxsim model result:", res_onnx)
    except:
        pass
    


if __name__ == "__main__":
    main()
