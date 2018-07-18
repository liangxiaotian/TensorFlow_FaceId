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

class FaceidSolver(Solver):
    def __init__(self, dataset, net, common_params, solver_params):
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.moment = float(solver_params['moment'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.is_training = bool(solver_params['istraining'])

        self.dataset = dataset
        self.net = net
        #built graph
        # with tf.Graph().as_default() as g:
        #     self.construct_graph_one()
        #     self.construct_graph_two()
        #     self.construct_loss()
        self.construct_graph()

    def train(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #opt = tf.train.RMSPropOptimizer(self.learning_rate)
            opt = tf.train.AdadeltaOptimizer(learning_rate= self.learning_rate)
            #opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
            grads = opt.compute_gradients(self.loss)
            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        return apply_gradient_op

    def construct_graph_one(self):
        self.face_one = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        self.keep_prob_fc1 = tf.placeholder(tf.float32)
        with tf.variable_scope("model_one") as scope:
                #model_one
            self.predict_one = self.net.inference(self.face_one,
                                                            train=self.is_training, keep_prob_fc2 = self.keep_prob_fc1)
            print(self.predict_one.shape)
            #tf.summary.scalar("predict_one", self.predict_one)
            #self.merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

    def construct_graph_two(self):
        self.face_two = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.keep_prob_fc2 = tf.placeholder(tf.float32)

        with tf.variable_scope("model_one") as scope:
            scope.reuse_variables()
            self.predict_two = self.net.inference(self.face_two,
                                                  train=self.is_training, keep_prob_fc2=self.keep_prob_fc2)
            print(self.predict_two.shape)
            #tf.summary.scalar("predict_two", self.predict_two)
            #self.merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

    def construct_loss(self):
        self.labels = tf.placeholder(tf.float32, (self.batch_size))
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope('loss') as scope:
            self.loss = self.net.loss(self.predict_one, self.predict_two, self.labels)
        self.train_op = self.train()

    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.face_one = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.face_two = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size))

        #with tf.Graph().as_default():
        with tf.variable_scope("faceid") as scope:
                #model_one
            self.predict_one = self.net.inference(self.face_one,
                                                            train=self.is_training)
            print(self.predict_one.shape)
            #self.merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

            # model_two
            scope.reuse_variables()
            self.predict_two = self.net.inference(self.face_two,
                                                              train=self.is_training)
            print(self.predict_two.shape)
            #self.merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

        with tf.variable_scope('loss') as scope:
            self.loss = self.net.loss(self.predict_one, self.predict_two, self.labels)
        self.train_op = self.train()

    def saver(self):
        var_list = tf.trainable_variables()
        if self.global_step is not None:
            var_list.append(self.global_step)
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        return saver

    def solve(self):
        #saver = tf.train.Saver(self.net.trainable_collection, write_version=tf.train.SaverDef.V2)
        init = tf.global_variables_initializer()
        saver = self.saver()

        summary_op = tf.summary.merge_all()
        # summary_op = tf.summary.merge(
        #     tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        with tf.Session() as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.pretrain_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            for step in range(self.max_iterators):
                start_time = time.time()
                faces_one, faces_two, labels = self.dataset.get_batch()
                #print(faces_one, faces_two, labels)

                _, loss = sess.run([self.train_op, self.loss], feed_dict={self.face_one:faces_one, self.face_two:faces_two,
                                                     self.labels:labels})

                duration = time.time() - start_time

                assert not np.isnan(loss), "loss = Nan"

                if step % 10 == 0:
                    examples_per_sec = self.dataset.batch_size / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss,
                                        examples_per_sec, sec_per_batch))

                sys.stdout.flush()

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.face_one:faces_one, self.face_two:faces_two,
                                                                    self.labels:labels})
                    summary_writer.add_summary(summary_str, step)
                if step % 1000 == 0:
                    saver.save(sess, self.train_dir + "/faceid.ckpt", global_step= step)
