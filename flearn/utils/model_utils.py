import json
import numpy as np
import os

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)

def read_data(train_data_dir, test_data_dir):
    """读取客户端数据"""
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    # 检查是否为增强版MNIST数据集
    is_enhanced_mnist = 'enhanced_mnist' in train_data_dir
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    
    if is_enhanced_mnist:
        # 处理增强版MNIST数据集
        print("检测到增强版MNIST数据集，使用特殊处理...")
        
        # 处理训练数据
        for f in train_files:
            client_id = f.split('.')[0]  # 例如：client_0.json -> client_0
            clients.append(client_id)
            
            file_path = os.path.join(train_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            
            # 增强版MNIST中每个客户端数据格式: {'x': [...], 'y': [...], 'type': '...', ...}
            train_data[client_id] = {}
            train_data[client_id]['x'] = np.array(cdata['x'])
            train_data[client_id]['y'] = np.array(cdata['y'])
        
        # 处理测试数据
        test_file = os.path.join(test_data_dir, 'all_data.json')
        with open(test_file, 'r') as inf:
            test_cdata = json.load(inf)
        test_data['x'] = np.array(test_cdata['x'])
        test_data['y'] = np.array(test_cdata['y'])
        
        # 预处理数据 - 将数据转换为正确的格式
        from flearn.utils.enhanced_dataset_loader import preprocess_enhanced_mnist_data
        train_data, test_data = preprocess_enhanced_mnist_data(train_data, test_data)
        
    else:
        # 原始数据处理逻辑
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data


class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
	#os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
