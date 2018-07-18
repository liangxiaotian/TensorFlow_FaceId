from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import os
import math
import random
import cv2
import numpy as np
import glob

def get_record(face_dir):
    face_list = os.listdir(face_dir)
    person_list = []
    for i in face_list:
        person_one = []
        person_one.append(i)
        path = os.path.join(face_dir, i, '*.jpg')
        face = glob.glob(path)
        for name in face:
            person_one.append(name)
        person_list.append(person_one)
    return person_list


def get_face(person_list):
    name = person_list[0]
    images = []
    for i in person_list[1:]:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256))
        images.append(image)

    images = np.asarray(images, dtype=np.float32)
    images = images / 255
    return name, images


if __name__=="__main__":
    person_list = get_record('data/face/face_test')
    get_face(person_list)
