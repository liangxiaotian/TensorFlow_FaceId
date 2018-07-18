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
from multiprocessing import Queue
#from Queue import Queue
from threading import Thread
from faceid.dataset.dataset import DataSet



class GetDataset(DataSet):
    """数据集的读取类别"""

    def __init__(self,common_params, dataset_params):
        self.data_path = str(dataset_params['path'])
        self.batch_size = int(common_params['batch_size'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.thread_num = int(dataset_params['thread_num'])

        # 文件列表创建队列
        self.record_queue = Queue(maxsize=500)
        # 文件队列
        self.image_label_queue = Queue(maxsize=256)

        self.record_list = []
        input_file = open(self.data_path, 'r')

        for i, line in enumerate(input_file):
            line = line.strip()
            ss = line.split(' ')
            # 将数字转换为浮点型
            self.record_list.append(ss)

        self.record_point = 0
        self.record_numbet = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_numbet / self.batch_size)
        # 开启线程调用函数，读取数据
        t_record_producter = Thread(target=self.record_producter)
        # 设置所有线程一起结束
        t_record_producter.daemon = True
        t_record_producter.start()
        # print(t_record_producter)

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_customer(self):
        #队列的操作，从文件队列中取数据信息，处理信息之后放入label队列
        while True:
            #从文件队列里面读取记录
            item = self.record_queue.get()
            #print(item)
            #根据读取的记录，读取图像和label
            out = self.record_propross(item)
            #print(out)
            #添加进图像队列
            self.image_label_queue.put(out)

    def record_producter(self):
        """队列处理,图像列表随机生成"""
        while True:
            #每循环列表一次随机生成一次列表
            if self.record_point % self.record_numbet == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def record_propross(self, record):
        """数据读取与处理， record:face_one， face_two ， label
        return：image:
        labels:
        """
        face_one = cv2.imread(record[0])
        face_two = cv2.imread(record[1])
        face_one = cv2.cvtColor(face_one, cv2.COLOR_BGR2RGB)
        face_two = cv2.cvtColor(face_two, cv2.COLOR_BGR2RGB)
        face_one = cv2.resize(face_one, (self.height,self.width))
        face_two = cv2.resize(face_two, (self.height, self.width))
        label = record[2]
        #print(record)
        return [face_one, face_two, label]

    def get_batch(self):
        """
        #读取一个batch的数据
        :return:
        face_one: 4-D ndarray [batch_size, height, width, 3]
        face_two: 4-D ndarray [batch_size, height, width, 3]
        labels: 1-D ndarray [batch_size]
        """
        faces_one = []
        faces_two = []
        labels= []
        for i in range(self.batch_size):
            face_one, face_two, label = self.image_label_queue.get()
            #print(face_one.shape, face_two.shape, label)
            faces_one.append(face_one)
            faces_two.append(face_two)
            labels.append(label)
        faces_one = np.asarray(faces_one, dtype=np.float32)
        faces_two = np.asarray(faces_two, dtype=np.float32)
        faces_one = faces_one / 255
        faces_two = faces_two / 255
        labels = np.asarray(labels, dtype=np.float32)
        return faces_one, faces_two, labels

    def read_image_name(self):
        return self.record_queue.get()

