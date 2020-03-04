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
            bin_path=cfg.PERSON_REID_BIN):
        """Initialize Face embedder object"""
        self.__net = cv.dnn.readNet(xml_path, bin_path)
        self.__net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


    def forward(self, person_image):

        """get embedding from a face image"""
        blob = cv.dnn.blobFromImage(person_image, size=(64, 160), ddepth=cv.CV_8U)
        self.__net.setInput(blob)

        emb = self.__net.forward()

        return emb