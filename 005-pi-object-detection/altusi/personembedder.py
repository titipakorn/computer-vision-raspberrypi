"""
Face Embedder class
===================
Face Embedding computation class
"""


import numpy as np
import cv2 as cv

import altusi.config as cfg


class PersonEmbedder:
    """Face embedder class"""

    def __init__(self, 
            xml_path=cfg.PERSON_REID_XML,
            bin_path=cfg.PERSON_REID_BIN,max_reqs=100):
        """Initialize Face embedder object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)
        self.max_reqs=max_reqs
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def forward(self, batch):

        """get embedding from a person image"""
        futureOutputs=[]
        outputs=[]
        assert len(batch) <= self.max_reqs
        for frame in batch:
            blob = cv.dnn.blobFromImage(frame, size=(64, 160), ddepth=cv.CV_8U)
            self.__net.setInput(blob)
            futureOutputs.append(self.__net.forwardAsync())
        while futureOutputs and futureOutputs[0].wait_for(0):
            out = futureOutputs[0].get().reshape(256)
            outputs.append(np.copy([out]))

            del futureOutputs[0]
        return outputs