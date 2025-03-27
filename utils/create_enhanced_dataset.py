import os
import numpy as np
import tensorflow as tf
import json
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import shutil

def create_enhanced_mnist_dataset(output_dir="data/enhanced_mnist", num_clients=100):
    """
    创建增强版MNIST数据集，具有以下特性：
    1. 非平衡客户端质量
       - 20%的客户端数据添加高斯噪声
       - 10%的客户端数据量小但质量高（纯净数据）
       - 15%的客户端数据量大但包含非典型模式（旋转、缩放等变换）
    2. 非IID分布
       - 55%的客户端按Dirichlet分布分配数据（不同浓度参数）
       - 30%的客户端只包含2-3个特定类别
       - 15%的客户端包含所有类别但分布极不平衡
    """
    print("加载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # 为测试集创建一个均衡分布
    test_data = {'x': x_test.tolist(), 'y': y_test.tolist()}
    with open(os.path.join(output_dir, "test", "all_data.json"), 'w') as f:
        json.dump(test_data, f)
    
    # 客户端类型分布
    client_types = {
        'noisy': int(0.20 * num_clients),  # 有噪声的客户端
        'small_high_quality': int(0.10 * num_clients),  # 数据量小但质量高
        'large_atypical': int(0.15 * num_clients),  # 数据量大但非典型
        'dirichlet': int(0.55 * num_clients),  # Dirichlet分布
    }
    
    # 确保总数等于num_clients
    remaining = num_clients - sum(client_types.values())
    client_types['dirichlet'] += remaining
    
    # 按类别分组数据
    class_idxs = [np.where(y_train == i)[0] for i in range(10)]
    
    # 客户端数据分配
    client_data = {}
    
    # 1. 创建有噪声的客户端
    noisy_clients = list(range(client_types['noisy']))
    for client_id in noisy_clients:
        # 随机选择数据量
        num_samples = np.random.randint(500, 1000)
        
        # 随机选择样本
        selected_idxs = np.random.choice(len(y_train), num_samples, replace=False)
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        # 添加噪声
        noise_level = np.random.uniform(0.1, 0.3)
        client_x += np.random.normal(0, noise_level, client_x.shape)
        client_x = np.clip(client_x, 0, 1)  # 确保值在[0,1]范围内
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'noisy',
            'noise_level': float(noise_level)
        }
    
    # 2. 创建数据量小但质量高的客户端
    small_high_quality_clients = list(range(client_types['noisy'], 
                                           client_types['noisy'] + client_types['small_high_quality']))
    for client_id in small_high_quality_clients:
        # 较小的数据量
        num_samples = np.random.randint(100, 300)
        
        # 随机选择样本，但确保类别平衡
        samples_per_class = num_samples // 10
        selected_idxs = []
        for class_idx in class_idxs:
            selected_idxs.extend(np.random.choice(class_idx, samples_per_class, replace=False))
        
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'small_high_quality'
        }
    
    # 3. 创建数据量大但包含非典型模式的客户端
    large_atypical_clients = list(range(
        client_types['noisy'] + client_types['small_high_quality'],
        client_types['noisy'] + client_types['small_high_quality'] + client_types['large_atypical']
    ))
    for client_id in large_atypical_clients:
        # 较大的数据量
        num_samples = np.random.randint(1000, 2000)
        
        # 随机选择样本
        selected_idxs = np.random.choice(len(y_train), num_samples, replace=False)
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        # 应用变换（旋转、缩放等）
        transform_type = np.random.choice(['rotate', 'scale', 'shift'])
        if transform_type == 'rotate':
            # 旋转图像
            angle = np.random.uniform(-30, 30)
            for i in range(len(client_x)):
                # 使用TensorFlow进行旋转
                img = tf.keras.preprocessing.image.array_to_img(client_x[i].reshape(28, 28, 1))
                img = tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.random_rotation(img, angle)
                )
                client_x[i] = img.reshape(28, 28) / 255.0
        elif transform_type == 'scale':
            # 缩放图像
            scale = np.random.uniform(0.7, 1.3)
            for i in range(len(client_x)):
                # 使用TensorFlow进行缩放
                img = tf.keras.preprocessing.image.array_to_img(client_x[i].reshape(28, 28, 1))
                img = tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.random_zoom(img, (scale, scale))
                )
                client_x[i] = img.reshape(28, 28) / 255.0
        elif transform_type == 'shift':
            # 平移图像
            shift = np.random.uniform(-0.2, 0.2)
            for i in range(len(client_x)):
                # 使用TensorFlow进行平移
                img = tf.keras.preprocessing.image.array_to_img(client_x[i].reshape(28, 28, 1))
                img = tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.random_shift(img, shift, shift)
                )
                client_x[i] = img.reshape(28, 28) / 255.0
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'large_atypical',
            'transform': transform_type
        }
    
    # 4. 创建Dirichlet分布的客户端（非IID）
    dirichlet_clients = list(range(
        client_types['noisy'] + client_types['small_high_quality'] + client_types['large_atypical'],
        num_clients
    ))
    
    # 将Dirichlet客户端分为三组
    num_dirichlet = len(dirichlet_clients)
    dirichlet_groups = {
        'few_classes': dirichlet_clients[:int(0.3 * num_dirichlet)],  # 只有2-3个类别
        'imbalanced': dirichlet_clients[int(0.3 * num_dirichlet):int(0.45 * num_dirichlet)],  # 极不平衡
        'dirichlet': dirichlet_clients[int(0.45 * num_dirichlet):]  # 标准Dirichlet
    }
    
    # 4.1 只有2-3个类别的客户端
    for client_id in dirichlet_groups['few_classes']:
        # 随机选择2-3个类别
        num_classes = np.random.randint(2, 4)
        selected_classes = np.random.choice(10, num_classes, replace=False)
        
        # 为每个选定的类别选择样本
        selected_idxs = []
        for cls in selected_classes:
            # 每个类别的样本数量
            samples_per_class = np.random.randint(200, 500)
            selected_idxs.extend(np.random.choice(class_idxs[cls], samples_per_class, replace=False))
        
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'few_classes',
            'classes': selected_classes.tolist()
        }
    
    # 4.2 分布极不平衡的客户端
    for client_id in dirichlet_groups['imbalanced']:
        # 创建极不平衡的分布
        class_probs = np.random.dirichlet(np.ones(10) * 0.1)  # 低alpha值使分布更不平衡
        
        # 总样本数
        num_samples = np.random.randint(500, 1000)
        
        # 按概率分配样本数
        samples_per_class = (class_probs * num_samples).astype(int)
        samples_per_class[-1] += num_samples - np.sum(samples_per_class)  # 确保总和等于num_samples
        
        selected_idxs = []
        for cls, n_samples in enumerate(samples_per_class):
            if n_samples > 0:
                selected_idxs.extend(np.random.choice(class_idxs[cls], n_samples, replace=False))
        
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'imbalanced',
            'class_distribution': samples_per_class.tolist()
        }
    
    # 4.3 标准Dirichlet分布的客户端
    for client_id in dirichlet_groups['dirichlet']:
        # 使用不同的alpha值
        alpha = np.random.uniform(0.1, 1.0)
        class_probs = np.random.dirichlet(np.ones(10) * alpha)
        
        # 总样本数
        num_samples = np.random.randint(500, 1000)
        
        # 按概率分配样本数
        samples_per_class = (class_probs * num_samples).astype(int)
        samples_per_class[-1] += num_samples - np.sum(samples_per_class)  # 确保总和等于num_samples
        
        selected_idxs = []
        for cls, n_samples in enumerate(samples_per_class):
            if n_samples > 0:
                selected_idxs.extend(np.random.choice(class_idxs[cls], n_samples, replace=False))
        
        client_x = x_train[selected_idxs].copy()
        client_y = y_train[selected_idxs].copy()
        
        client_data[f'client_{client_id}'] = {
            'x': client_x.tolist(),
            'y': client_y.tolist(),
            'type': 'dirichlet',
            'alpha': float(alpha),
            'class_distribution': samples_per_class.tolist()
        }
    
    # 保存客户端数据
    for client_id, data in client_data.items():
        with open(os.path.join(output_dir, "train", f"{client_id}.json"), 'w') as f:
            json.dump(data, f)
    
    # 创建元数据文件
    metadata = {
        'num_clients': num_clients,
        'client_types': client_types,
        'dirichlet_groups': {k: len(v) for k, v in dirichlet_groups.items()},
        'num_samples': {client_id: len(data['y']) for client_id, data in client_data.items()},
        'client_type_details': {client_id: data.get('type') for client_id, data in client_data.items()}
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # 创建数据集统计图表
    create_dataset_visualizations(client_data, output_dir)
    
    print(f"增强版MNIST数据集已创建，共{num_clients}个客户端")
    print(f"数据集保存在: {output_dir}")
    print(f"客户端类型分布: {client_types}")
    
    return metadata

def create_dataset_visualizations(client_data, output_dir):
    """创建数据集的可视化统计图表"""
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # 1. 客户端类型分布饼图
    client_types = [data.get('type') for data in client_data.values()]
    type_counts = {}
    for t in client_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    plt.figure(figsize=(10, 6))
    plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    plt.title('客户端类型分布')
    plt.savefig(os.path.join(output_dir, "visualizations", "client_types.png"))
    plt.close()
    
    # 2. 客户端数据量分布
    client_sizes = [len(data['y']) for data in client_data.values()]
    client_types = [data.get('type') for data in client_data.values()]
    
    plt.figure(figsize=(12, 6))
    for t in set(client_types):
        sizes = [size for i, size in enumerate(client_sizes) if client_types[i] == t]
        plt.hist(sizes, alpha=0.5, label=t, bins=20)
    
    plt.xlabel('数据量')
    plt.ylabel('客户端数量')
    plt.title('各类型客户端数据量分布')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "visualizations", "client_sizes.png"))
    plt.close()
    
    # 3. 样例图像展示
    plt.figure(figsize=(15, 10))
    
    # 为每种类型的客户端选择一个样例
    type_examples = {}
    for client_id, data in client_data.items():
        client_type = data.get('type')
        if client_type not in type_examples:
            type_examples[client_type] = client_id
    
    for i, (client_type, client_id) in enumerate(type_examples.items()):
        data = client_data[client_id]
        x = np.array(data['x'])
        
        # 显示5个样例图像
        for j in range(5):
            if j < len(x):
                plt.subplot(len(type_examples), 5, i*5 + j + 1)
                plt.imshow(x[j], cmap='gray')
                plt.title(f"{client_type}\n{client_id}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visualizations", "example_images.png"))
    plt.close()
    
    # 4. 类别分布热图
    # 选择一些有代表性的客户端
    representative_clients = {}
    for client_id, data in client_data.items():
        client_type = data.get('type')
        if client_type not in representative_clients:
            representative_clients[client_type] = []
        if len(representative_clients[client_type]) < 3:  # 每种类型选3个
            representative_clients[client_type].append(client_id)
    
    # 展平列表
    selected_clients = [c for clients in representative_clients.values() for c in clients]
    
    # 创建类别分布矩阵
    class_dist = np.zeros((len(selected_clients), 10))
    for i, client_id in enumerate(selected_clients):
        y = np.array(client_data[client_id]['y'])
        for j in range(10):
            class_dist[i, j] = np.sum(y == j) / len(y)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(class_dist, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='比例')
    plt.xlabel('类别')
    plt.ylabel('客户端')
    plt.title('代表性客户端的类别分布')
    plt.xticks(range(10))
    plt.yticks(range(len(selected_clients)), selected_clients)
    plt.savefig(os.path.join(output_dir, "visualizations", "class_distribution.png"))
    plt.close()

if __name__ == "__main__":
    create_enhanced_mnist_dataset(num_clients=100)