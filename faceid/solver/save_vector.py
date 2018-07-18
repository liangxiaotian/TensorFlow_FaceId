from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import faceid

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from faceid.solver.solver import Solver
from faceid.dataset.get_face import get_record,get_face

class FaceidSolver_Save(Solver):
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

    def solve(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.pretrain_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            person_list = get_record('data/face/face_test')

            for person in person_list:
                name, image = get_face(person)
                print(name, image.shape)

                pre_one = sess.run([self.predict_one],
                                                 feed_dict={self.face_one:image})
                face = np.asarray(pre_one, dtype=np.float32)
                face = np.squeeze(face)
                print(face.shape)
                np.savetxt('data/face_database/face_%s.txt' % name, face)
