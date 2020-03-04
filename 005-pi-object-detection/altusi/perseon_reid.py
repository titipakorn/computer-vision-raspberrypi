import numpy as np
import cv2 as cv

import altusi.config as cfg
from altusi.inference import Network

class PersonREID:
    def __init__(self,
            xml_path=cfg.PERSON_REID_XML,
            device='MYRIAD',
            inp_size=1, out_size=1,
            num_requests=2, plugin=None):
        self.__net = Network()
        self.plugin, (self.B, self.C, self.H, self.W) = self.__net.load_model(
                xml_path, device, inp_size, out_size, num_requests, plugin=plugin)


    def forward(self, face_image):
        image = cv.resize(face_image, (self.W, self.H))
        #image = image.transpose((2, 0, 1))
        image = image.reshape((self.B, self.C, self.H, self.W))

        self.__net.exec_net(0, image)
        self.__net.wait(0)

        emb = self.__net.get_output(0).reshape(256)

        return emb

    def getPlugin(self):
        return self.plugin
