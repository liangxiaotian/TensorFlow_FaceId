from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from faceid.solver.solver import Solver

class FaceidSolver_test(Solver):
    def __init__(self, dataset, net, common_params, solver_params):
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.train = bool(solver_params['test_training'])

        self.dataset = dataset
        self.net = net

        self.construct_graph()

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.face_one = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.face_two = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size))
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("faceid") as scope:
            self.predict_one = self.net.inference(self.face_one, train = self.train)
            print(self.predict_one.shape)
            scope.reuse_variables()
            self.predict_two = self.net.inference(self.face_two, train = self.train)
            print(self.predict_two.shape)
        self.distance = self.net.distance(self.predict_one, self.predict_two)
        #self.loss = self.net.loss(self.predict_one, self.predict_two, self.labels)

    def solve(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.pretrain_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            acc_test = []
            for step in range(100):
                start_time = time.time()
                faces_one, faces_two, labels = self.dataset.get_batch()
                #print(faces_one, faces_two, labels)

                pre_one, pre_two, distance = sess.run([self.predict_one, self.predict_two, self.distance], feed_dict={self.face_one:faces_one, self.face_two:faces_two,
                                                     self.labels:labels})

                #print(pre_one.shape, pre_two.shape)
                print(distance)
                y_pre = distance.ravel() < 0.5
                y_labels = labels.ravel() < 0.5

                correct_prediction = np.equal(y_pre, y_labels)
                accuracy = np.mean(np.asarray(correct_prediction, np.float32))
                acc_test.append(accuracy)
                if step % 50 == 0:
                    print(step)
                sys.stdout.flush()

            print(np.mean(acc_test))