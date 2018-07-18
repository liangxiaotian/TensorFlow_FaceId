import os
import numpy
import glob
import random

os.path.abspath('./')
TRAIN_DATA_PATH = 'data/face/face_train'
TRAIN_OUT_PATH = 'data/face.txt'

TEST_DATA_PATH = 'data/face/face_test'
TEST_OUT_PATH = 'data/face_test.txt'

def get_face_list(DATA_PATH):
    face_list = []
    face_dir = os.listdir(DATA_PATH)
    for i in face_dir:
        img_list = glob.glob(os.path.join(DATA_PATH, i) + "/*.jpg")
        face_list.append(img_list)
    return face_list

def get_same_person(face_list):
    same_face_list = []
    for one_person in face_list:
        for one_face in one_person:
            face_one = one_face
            n = random.randint(0, len(one_person) - 1)
            face_two = one_person[n]
            same_face_list.append(face_one + " " + face_two + " 0" + "\n")
    return same_face_list

def get_dif_person(face_list):
    dif_face_list = []
    for i,one_person in enumerate(face_list):
        for one_face in one_person:
            face_one = one_face
            m = random.randint(0, len(face_list) - 1)
            if m == i:
                m = (m + 1) % (len(face_list) - 1)
            #n = random.randint(0, len(one_person) - 1)
            face_two = random.choice(face_list[m])
            dif_face_list.append(face_one + " " + face_two + " 1" + "\n")
    return dif_face_list

def main(TRAIN_DATA_PATH, TRAIN_OUT_PATH):
    out_file = open(TRAIN_OUT_PATH,"w")
    #获得所有的人脸文件
    face_list = get_face_list(TRAIN_DATA_PATH)
    #相同的人脸列表
    same_face_list = get_same_person(face_list)
    #不同的人脸列表
    dif_face_list = get_dif_person(face_list)
    for i in range(len(same_face_list)):
        out_file.write(same_face_list[i])
        out_file.write(dif_face_list[i])
    print("Done")

if __name__ == "__main__":
    main(TRAIN_DATA_PATH, TRAIN_OUT_PATH)
    main(TEST_DATA_PATH, TEST_OUT_PATH)