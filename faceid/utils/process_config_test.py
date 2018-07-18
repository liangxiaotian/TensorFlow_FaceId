common_params = {'image_size': '256',
                 'batch_size': '8',
                 }

dataset_params = {'name':'faceid.dataset.get_dataset.GetDataset',
                  'path':'data/face_test.txt',
                  'thread_num':'3'}

net_params = {'name': 'faceid.net.facenet.FaceNet',
            'weight_decay':'0.0005',
              }

solver_params = {'name': 'faceid.solver.faceid_solver.FaceidSolver',
            'learning_rate': '0.01',
            'moment': '0.9',
            'max_iterators': '1000000',
            'pretrain_model_path': 'models/train/',
            'train_dir': 'models/train',
            'istraining':'True',
            'test_training':'False',
}

def get_params():
    return common_params, dataset_params, net_params, solver_params