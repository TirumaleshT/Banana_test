#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:41:04 2022

@author: strato1
"""
import os
import math
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T

import env_config

import warnings
warnings.filterwarnings("ignore")


def get_model_file_path(class_name):
    model_files = os.listdir('model_weights')

    if class_name == 'cb':
        file_prefix = 'cb_detection_weight'
    elif class_name == 'box':
        file_prefix = 'box_weight'
    else:
        file_prefix = ''

    file_name = [m for m in model_files if file_prefix in m][0]

    return os.path.join('model_weights', file_name)


# CheckBox and Box Detection Model Loading   
def loadModel(class_name, num_class):
    #model_path_CB = get_model_file_path(class_name)
    model_path_CB = os.path.join('model_weights', 'model.pth')
    if class_name == 'cb':
        cfg = get_cfg()
        cfg.MODEL.DEVICE = env_config.DEVICE
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (class_name,)
        cfg.DATASETS.TEST = ()
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class #your number of classes 
        cfg.MODEL.WEIGHTS = model_path_CB
        cfg.DATASETS.TEST = (class_name, )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg) 
        
        MetadataCatalog.get("cb").thing_classes = ["mk", "umk"]
        test_metadata = MetadataCatalog.get("cb")
        # predictor = DefaultPredictor(cfg) 
        return predictor
    else:
        cfg = get_cfg()
        cfg.MODEL.DEVICE = env_config.DEVICE
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (class_name,)
        cfg.DATASETS.TEST = ()
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class #your number of classes 
        cfg.MODEL.WEIGHTS = model_path_CB
        cfg.DATASETS.TEST = (class_name, )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg) 
        return predictor
 
# Checkbox Prediction 
def checkBox_detection(image,predictor):
    
    height,width = image.shape[:2]
    outputs = predictor(image)

    output_score = outputs['instances'].scores
    pred_box = outputs["instances"].pred_boxes
    pred_class = outputs["instances"].pred_classes
    # coord = []
    coordist = []
    for i,scr,classes in zip(pred_box.__iter__(),output_score.__iter__(),pred_class.__iter__()):
        xmin,ymin,xmax,ymax = i.cpu().numpy()
        cb_cls = classes.cpu().numpy()
        score = scr.cpu().numpy()
        scores = score.flatten()
        fScores = float(scores)

        if round(fScores,2) >= 0.99:
            x1 = int(xmin)
            y1 = int(ymin)
            x2 = int(xmax)
            y2 = int(ymax)
            
            distance = math.sqrt((y1)**2+(x1)**2)
            
            coordist.append([int(cb_cls),x1,y1,x2,y2,distance])
    cbIndex = sorted(coordist, key=lambda x:x[-1], reverse=False)
    
            # if x1 >= width * 0.20:            
            #     coord.append([int(cb_cls),x1,y1,x2,y2,'x_sort'])
            # else:
            #     coord.append([int(cb_cls),x1,y1,x2,y2,'y_sort'])
            
    return cbIndex

# Outer Box Prediction 
def boxDetection(image, model):
    
    height,width = image.shape[:2]
    outputs = model(image)
    
    output_score = outputs['instances'].scores
    output_pred_boxes = outputs["instances"].pred_boxes

    zipped = zip(output_pred_boxes.__iter__(),output_score.__iter__())
    max_box, max_score = max(zipped, key=lambda x: float(x[1].cpu()))
    max_box = [int(i) for i in max_box.cpu()]
    
    return max_box


# CheckBox Pre and Post Processing to ge the True or False Value  
def checkBox_Ops(dataCbv, snippet, predictor):

    cbIndex = checkBox_detection(snippet, predictor)
    
    if len(cbIndex) >=1:
        cbValue = dataCbv['element_ID'].split(",")
        
        if len(cbIndex) == len(cbValue):
            newcbIndex = cbIndex
        elif len(cbIndex) > len(cbValue):
            newcbIndex = cbIndex[:len(cbValue)]
        elif len(cbIndex) < len(cbValue):
            newcbIndex = cbIndex
        
        checkedBox = []
        if newcbIndex is not None:
            for ib,cbIdx in enumerate(newcbIndex):
                if cbIdx[0] == 0:
                    checkedBox.append(cbValue[ib])                    
        else:
            checkedBox.append("empty")

        dictVal = {}
        for cbVal in cbValue:
            if cbVal in checkedBox:
                dictVal[cbVal] = "True"
            else:
                dictVal[cbVal] = "False"                
        dataCbv['value_text'] = dictVal        
        return dataCbv
    else:
        failedCb = dataCbv['value_text'].split(",")        
        dictVal1 = {}
        for fail in failedCb:
            dictVal1[fail] = 'False'        
        dataCbv['value_text'] = dictVal1        
        return dataCbv

