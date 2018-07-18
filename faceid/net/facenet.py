from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from faceid.net.net import Net


class FaceNet(Net):
    def __init__(self, common_params, net_params, test=False):

        super(FaceNet, self).__init__(common_params, net_params)

        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.weight_decay = float(net_params['weight_decay'])


    def inference(self, image, train):
        """built facenet
        :arg
        image: 4-D tensor [batch_size, height, width, channel]
        :return
        pre: 2-D tensor [batch, 128]
        """
        with tf.variable_scope('facenet') as scope:
            with tf.variable_scope('conv1') as scope:

                temp_conv = self.conv2d("conv1_1", image, [3, 3, 3, 64], stride=2, padding = 'VALID', train=train)
                temp_norm = tf.layers.batch_normalization(temp_conv, training = train, name='bn_0')
                temp_conv = tf.nn.relu(temp_conv, name='conv1_relu')
                print(temp_conv.shape)

            temp_pool = self.max_pool(temp_conv, [3, 3], 2)
            print(temp_pool.shape)
            with tf.variable_scope('conv2') as scope:
                temp_conv = self.conv2d("conv2_1", temp_pool, [3, 3, 64, 16], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv2_relu')
                print(temp_conv.shape)

            with tf.variable_scope('conv3') as scope:
                temp_conv1 = self.conv2d("conv3_1", temp_conv, [3, 3, 16, 16], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv3_relu1')

                temp_conv2 = self.conv2d("conv3_2", temp_conv, [3, 3, 16, 16], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv3_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            with tf.variable_scope('conv4') as scope:
                temp_conv = self.conv2d('conv4_1', temp_concat, [3, 3, 32, 16], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv3_relu')
                print(temp_conv.shape)

            with tf.variable_scope('conv5') as scope:
                temp_conv1 = self.conv2d("conv5_1", temp_conv, [3, 3, 16, 16], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv5_relu1')

                temp_conv2 = self.conv2d("conv5_2", temp_conv, [3, 3, 16, 16], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv5_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            temp_pool = self.max_pool(temp_concat, [3, 3], 2)

            with tf.variable_scope('conv6') as scope:
                temp_conv = self.conv2d("conv6_1", temp_pool, [3, 3, 32, 32], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv6_relu1')

            with tf.variable_scope('conv7') as scope:
                temp_conv1 = self.conv2d("conv7_1", temp_conv, [3, 3, 32, 32], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv7_relu1')

                temp_conv2 = self.conv2d("conv7_2", temp_conv, [3, 3, 32, 32], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv7_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            with tf.variable_scope('conv8') as scope:
                temp_conv = self.conv2d("conv8_1", temp_concat, [3, 3, 64, 32], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv8_relu1')

            with tf.variable_scope('conv9') as scope:
                temp_conv1 = self.conv2d("conv9_1", temp_conv, [3, 3, 32, 32], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv9_relu1')

                temp_conv2 = self.conv2d("conv9_2", temp_conv, [3, 3, 32, 32], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv9_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            temp_pool = self.max_pool(temp_concat, [3, 3], 2)

            with tf.variable_scope('conv10') as scope:
                temp_conv = self.conv2d("conv10_1", temp_pool, [3, 3, 64, 48], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv10_relu1')

            with tf.variable_scope('conv11') as scope:
                temp_conv1 = self.conv2d("conv11_1", temp_conv, [3, 3, 48, 48], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv11_relu1')

                temp_conv2 = self.conv2d("conv11_2", temp_conv, [3, 3, 48, 48], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv11_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            with tf.variable_scope('conv12') as scope:
                temp_conv = self.conv2d("conv12_1", temp_concat, [3, 3, 96, 64], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv12_relu1')

            with tf.variable_scope('conv13') as scope:
                temp_conv1 = self.conv2d("conv13_1", temp_conv, [3, 3, 64, 64], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv13_relu1')

                temp_conv2 = self.conv2d("conv13_2", temp_conv, [3, 3, 64, 64], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv13_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)

            with tf.variable_scope('conv14') as scope:
                temp_conv = self.conv2d("conv14_1", temp_concat, [3, 3, 128, 64], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv14_relu1')
                print(temp_conv.shape)

            with tf.variable_scope('conv15') as scope:
                temp_conv1 = self.conv2d("conv15_1", temp_conv, [3, 3, 64, 64], stride=1,  train=train)
                temp_conv1 = tf.nn.relu(temp_conv1, name='conv15_relu1')

                temp_conv2 = self.conv2d("conv15_2", temp_conv, [3, 3, 64, 64], stride=1,  train=train)
                temp_conv2 = tf.nn.relu(temp_conv2, name='conv15_relu2')

                temp_concat = tf.concat([temp_conv1, temp_conv2], 3)
                print(temp_concat.shape)
                temp_drop = tf.nn.dropout(temp_concat, keep_prob=0.2)

            with tf.variable_scope('conv16') as scope:
                temp_conv = self.conv2d("conv16_1", temp_concat, [3, 3, 128, 512], stride=1,  train=train)
                temp_conv = tf.nn.relu(temp_conv, name='conv16_relu1')
                print(temp_conv.shape)

            local1 = self.local('local1', temp_conv, 16 * 16 * 512, 512,  train=train)
            #local1_drop_out = tf.nn.dropout(local1, keep_prob=1)
            print(local1.shape)
            local2 = self.local('local2', local1, 512, 128,  train=train)
            local_l2_norm = tf.nn.l2_normalize(local2, dim=1)
            print(local_l2_norm.shape)
            return local_l2_norm

    def distance(self, faces_one, faces_two):
        print(faces_one.shape, faces_two.shape)
        eucd = tf.pow(tf.subtract(faces_one, faces_two), 2)
        eucd = tf.reduce_sum(eucd, axis = 1, keep_dims= True)
        y_pred = tf.sqrt(eucd + 1e-6, name="eucd")  # 开方
        return y_pred

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        #y_true = tf.expand_dims(y_true, axis=1)
        y_pred = tf.squeeze(y_pred)
        print(y_true.shape, y_pred.shape)
        margin = tf.constant(1, tf.float32)
        pos = (1 - y_true) * tf.square(y_pred)
        neg = y_true * tf.square(tf.maximum(margin - y_pred, 0.))

        loss_two = tf.add(pos, neg)
        loss = tf.reduce_mean(loss_two)
        return loss

    def loss(self, faces_one, faces_two, labels):
        y_pred = self.distance(faces_one, faces_two)
        print(y_pred.shape)
        loss = self.contrastive_loss(labels, y_pred)
        print(loss.shape)

        tf.add_to_collection('losses', loss / self.batch_size)

        tf.summary.scalar('losses', loss / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')