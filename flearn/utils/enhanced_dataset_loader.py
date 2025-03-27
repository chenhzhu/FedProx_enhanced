import os
import json
import numpy as np

def load_enhanced_mnist():
    """
    加载增强版MNIST数据集
    
    返回:
        clients: 客户端ID列表
        groups: 组ID列表（这里为空）
        train_data: 训练数据字典，键为客户端ID，值为数据
        test_data: 测试数据字典
    """
    train_dir = os.path.join('data', 'enhanced_mnist', 'train')
    test_dir = os.path.join('data', 'enhanced_mnist', 'test')
    
    clients = []
    groups = []
    train_data = {}
    
    # 加载训练数据
    client_files = os.listdir(train_dir)
    for f in client_files:
        if f.endswith('.json'):
            client_name = f.split('.')[0]
            clients.append(client_name)
            
            with open(os.path.join(train_dir, f), 'r') as inf:
                cdata = json.load(inf)
            
            train_data[client_name] = {
                'x': np.array(cdata['x']),
                'y': np.array(cdata['y'])
            }
    
    # 加载测试数据
    test_data = {}
    with open(os.path.join(test_dir, 'all_data.json'), 'r') as inf:
        test_all = json.load(inf)
    
    test_data['x'] = np.array(test_all['x'])
    test_data['y'] = np.array(test_all['y'])
    
    return clients, groups, train_data, test_data

def preprocess_enhanced_mnist_data(train_data, test_data):
    """
    预处理增强版MNIST数据集
    
    参数:
        train_data: 训练数据字典
        test_data: 测试数据字典
    
    返回:
        处理后的训练和测试数据
    """
    # 确保数据形状正确
    for client in train_data:
        # 如果数据是3D的(图像)，将其展平为2D
        if len(train_data[client]['x'].shape) > 2:
            train_data[client]['x'] = train_data[client]['x'].reshape(
                train_data[client]['x'].shape[0], -1)
        
        # 将标签转换为one-hot编码
        if len(train_data[client]['y'].shape) == 1:
            y_one_hot = np.zeros((train_data[client]['y'].shape[0], 10))
            for i, label in enumerate(train_data[client]['y']):
                y_one_hot[i, int(label)] = 1
            train_data[client]['y'] = y_one_hot
    
    # 处理测试数据
    if len(test_data['x'].shape) > 2:
        test_data['x'] = test_data['x'].reshape(test_data['x'].shape[0], -1)
    
    if len(test_data['y'].shape) == 1:
        y_one_hot = np.zeros((test_data['y'].shape[0], 10))
        for i, label in enumerate(test_data['y']):
            y_one_hot[i, int(label)] = 1
        test_data['y'] = y_one_hot
    
    return train_data, test_data