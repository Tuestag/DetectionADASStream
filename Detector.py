from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog #, DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np


class Detector:
  def __init__(self):
      self.cfg = get_cfg()
      
      #Cargar el modelo
      self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
      self.cfg.MODEL.WEIGHTS = os.path.join("https://github.com/Tuestag/DetectionADASStream/releases/download/adasdetectronCO/Detectron2.pth")
      self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
      self.cfg.MODEL.DEVICE = "cpu"
      
      self.predictor = DefaultPredictor(self.cfg)

  def onVideo(self, videoPath):
    cap = cv2.VideoCapture(videoPath)
    
    if (cap.isOpened()==False):
      print("Error opening the file...")
      return
    
    (succes,image) = cap.read()
    
    while success:
      if self.model_type != "PS":
        predictions = self.predictor(image)
        
        viz = Visualizer(image[:,:,::-1],metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),instance_mode = ColorMode.IMAGE)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
       
      else:
        predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
        viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
       
        output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
      cv2.imshow("Result",output.get_image()[:,:,::-1])
      key = cv2.waitKey(1) / 0xFF
      
      if key == ord("q"):
        break
       
      (succes,image) = cap.read()

import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

