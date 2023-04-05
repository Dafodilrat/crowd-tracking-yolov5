
import cv2
import numpy as np
import sys
import glob

import time
import torch



class YoloDetector():

    def __init__(self, model_name):
        
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        #print(self.classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = torch.device("cpu")
        print("Using Device: ", self.device)

    def load_model(self, model_name):
   
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width,height))
        #frame = frame.to(self.device)
        
        results = self.model(frame)
        
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        return labels, cord

    def class_to_label(self, x):
   
        return self.classes[int(x)]


    def plot_boxes(self, results, frame, height, width, confidence=0.3,cls='head'):
    
        labels, cord = results
        detections = []
        net_dir=[] 

        n = len(labels)
        x_shape, y_shape = width, height
    
        for i in range(n):
            row = cord[i]
            
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[i]) == cls:
                
                    x_center = x1 + (x2 - x1)
                    y_center = y1 + ((y2 - y1) / 2)
                    
                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = cls
                    
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(),cls))
                
        
        return frame, detections
        
        
        



  