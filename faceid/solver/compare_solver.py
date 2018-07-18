from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import faceid
import os

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime
import cv2
import glob

from faceid.solver.solver import Solver
from faceid.dataset.get_face import get_record,get_face

class FaceidSolver_test(Solver):
    def __init__(self, dataset, net, common_params, solver_params):
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.train = bool(solver_params['test_training'])

        self.dataset = dataset
        self.net = net

        self.construct_graph()

    def construct_graph(self):
        self.face_one = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))

        with tf.variable_scope("faceid") as scope:
            self.predict_one = self.net.inference(self.face_one, train = self.train)
            print(self.predict_one.shape)


    def comput_distance(self,faces_one, faces_two):
        #print(faces_one.shape, faces_two.shape)
        #distance = np.sqrt(np.sum(np.square(faces_one - faces_two)) + 1e-6)
        eucd = tf.pow(tf.subtract(faces_one, faces_two), 2)
        eucd = tf.reduce_sum(eucd, axis = 1,keep_dims= True)
        y_pred = tf.sqrt(eucd + 1e-6, name="eucd")
        return y_pred

    def get_face_database(self, data_path):
        #读取数据库中人脸矩阵
        file_list = glob.glob(os.path.join(data_path, '*.txt'))
        print(file_list)
        persons = []
        name = []
        for i in file_list:
            name.append(i)
            person = np.loadtxt(i)
            persons.append(person)
        return persons, name



    def solve(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.pretrain_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            # person_list = get_record('data/face/face_test')
            data_path = 'data/face_database'
            persons, name = self.get_face_database(data_path)
            #print(persons[0].shape)
            images_list = glob.glob('data/*.jpg')
            for image_name in images_list:
                image = cv2.imread(image_name)
                image = cv2.resize(image, (256, 256))
                face = image / 255
                face = np.expand_dims(face, axis=0)

                pre_one = sess.run([self.predict_one],
                                   feed_dict={self.face_one: face})
                pre_one = np.asarray(pre_one, dtype=np.float32)
                pre_one = np.squeeze(pre_one)
                #pre_one = np.expand_dims(pre_one, axis=0)

                distance = []

                for person in persons:
                    dis = self.comput_distance(pre_one, person)
                    dis = sess.run(dis)
                    distance.append(np.mean(dis))
                    print(np.mean(dis))

                min_index = distance.index(min(distance))
                if distance[min_index] < 0.4:
                    print(name[min_index])
                    person_name = name[min_index].split('\\')[-1]
                    person_name = person_name.split('.')[0]
                    cv2.putText(image,person_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1, )
                else:
                    cv2.putText(image, "No", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1, )
                    print("NO!")


                cv2.imshow("Face", image)
                key = cv2.waitKey(1000) & 0xFF
                if key == ord('q'):
                    break
