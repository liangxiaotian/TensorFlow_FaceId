import sys
sys.path.append('./')
import os
import numpy as np
import glob
import tensorflow as tf


def comput_distance(faces_one, faces_two):
    distance = np.sqrt(np.sum(np.square(faces_one - faces_two)) + 1e-6)
    eucd = tf.pow(tf.subtract(faces_one, faces_two), 2)
    eucd = tf.reduce_sum(eucd, axis=1, keep_dims=True)
    y_pred = tf.sqrt(eucd + 1e-6, name="eucd")
    return y_pred

data_path = 'data/face_database/'
file_list = glob.glob(os.path.join(data_path, '*.txt'))
person_one = np.loadtxt(file_list[0])
print(person_one[0].shape)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in file_list:
        person_other = np.loadtxt(i)
        distance = comput_distance(person_one[0:8], person_other[0:8])
        dis = sess.run(distance)
        print(dis)
