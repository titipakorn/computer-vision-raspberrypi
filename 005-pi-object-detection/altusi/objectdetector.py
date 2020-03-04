"""
Object Detector class
===================

Module for Object Detection
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg

from altusi.ie_tools import load_ie_model



class ObjectDetector:
    """Wrapper class for detector"""

    def __init__(self, ie, model_path=cfg.PERSON_DET_XML, conf=.5, device='MYRIAD', ext_path='', max_num_frames=1):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames)
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def get_detections(self, frames):
        """Returns all detections on frames"""
        assert len(frames) <= self.max_num_frames

        all_detections = []
        for i in range(len(frames)):
            self.net.forward_async(frames[i])
        outputs = self.net.grab_all_async()

        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, frames[i].shape)
            all_detections.append(detections)

        return all_detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections
# class ObjectDetector:
#     """Object Detector class"""

#     def __init__(self,
#             xml_path=cfg.PERSON_DET_XML,
#             bin_path=cfg.PERSON_DET_BIN):
#         """Initialize Object detector object"""
#         self.__net = cv.dnn.readNet(xml_path, bin_path)

#         # with NCS support
#         self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


#     def getObjects(self, image, def_score=0.5):
#         """Detect objects in an input image with given threshold"""
#         H, W = image.shape[:2]
#         blob = cv.dnn.blobFromImage(image, size=(544, 320), ddepth=cv.CV_8U)
#         self.__net.setInput(blob)
#         out = self.__net.forward()

#         bboxes = []
#         scores = []
#         for det in out.reshape(-1, 7):
#             score = float(det[2])
#             if score < def_score: continue

#             x1 = max(0, int(det[3] * W))
#             y1 = max(0, int(det[4] * H))
#             x2 = min(W, int(det[5] * W))
#             y2 = min(H, int(det[6] * H))

#             bboxes.append((x1, y1, x2, y2))
#             scores.append(score)
        
#         return scores, bboxes
