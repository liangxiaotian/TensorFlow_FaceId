from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import faceid

from faceid.dataset.get_dataset import GetDataset
from faceid.net.facenet import FaceNet
from faceid.solver.test_solver import FaceidSolver_test
from faceid.utils import process_config
import tensorflow as tf

common_params, dataset_params, net_params, solver_params = process_config.get_params()


dataset = eval(dataset_params['name'])(common_params, dataset_params)
# face_one, face_two, labels = dataset.read_image_name()
# print(face_one, face_two, labels)
#face_one, face_two, labels = dataset.get_batch()
net = eval(net_params['name'])(common_params, net_params)

#image = net.inference(face_one)
# solver = eval(solver_params['name'])(dataset, net, common_params, solver_params)
solver = FaceidSolver_test(dataset, net, common_params, solver_params)
solver.solve()