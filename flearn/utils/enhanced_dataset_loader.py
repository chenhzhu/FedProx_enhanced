import json
import numpy as np
import os
import tensorflow as tf

def load_enhanced_mnist(data_dir="data/enhanced_mnist"):
    """
    加载增强版MNIST数据集
    
    Args:
        data_dir: 数据集目录
        
    Returns:
        clients: 客户端列表
        groups: 客户端组列表
        train_data: 训练数据字典
        test_data: 测试数据字典
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    # 加载元数据
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # 加载测试数据
    with open(os.path.join(test_dir, "all_data.json"), 'r') as f:
        test_data_json = json.load(f)
    
    test_data = {
        'x': np.array(test_data_json['x']),
        'y': np.array(test_data_json['y'])
    }
    
    # 加载训练数据
    clients = []
    groups = []
    train_data = {}
    
    for filename in os.listdir(train_dir):
        if filename.endswith('.json'):
            client_id = filename.split('.')[0]
            clients.append(client_id)
            
            # 根据客户端类型分组
            with open(os.path.join(train_dir, filename), 'r') as f:
                client_data = json.load(f)
            
            client_type = client_data.get('type', 'unknown')
            groups.append(client_type)
            
            # 转换数据格式
            train_data[client_id] = {
                'x': np.array(client_data['x']),
                'y': np.array(client_data['y'])
            }
    
    return clients, groups, train_data, test_data

def preprocess_enhanced_mnist_data(train_data, test_data):
    """
    预处理增强版MNIST数据集,转换为TensorFlow模型所需格式
    
    Args:
        train_data: 训练数据字典
        test_data: 测试数据字典
        
    Returns:
        处理后的训练和测试数据
    """
    # 确保数据是正确的形状
    for client_id in train_data:
        train_data[client_id]['x'] = train_data[client_id]['x'].reshape(-1, 28, 28, 1)
    
    test_data['x'] = test_data['x'].reshape(-1, 28, 28, 1)
    
    return train_data, test_data