import numpy as np
import tensorflow as tf
import argparse
from flearn.utils.model_utils import read_data
from flearn.utils.enhanced_dataset_loader import load_enhanced_mnist, preprocess_enhanced_mnist_data
from flearn.utils.tf_utils import process_grad
import matplotlib.pyplot as plt
import os

# 导入不同的训练器
from flearn.trainers.fedavg import Server as FedAvgServer
from flearn.trainers.fedprox import Server as FedProxServer
from flearn.models.mnist.cnn import Model

def create_dataset_if_not_exists():
    """如果增强数据集不存在，则创建它"""
    if not os.path.exists("data/enhanced_mnist"):
        print("增强版MNIST数据集不存在,正在创建...")
        from utils.create_enhanced_dataset import create_enhanced_mnist_dataset
        create_enhanced_mnist_dataset()
    else:
        print("增强版MNIST数据集已存在")

def main():
    parser = argparse.ArgumentParser()
    
    # 训练参数
    parser.add_argument('--algorithm', type=str, default='fedprox', choices=['fedavg', 'fedprox'], 
                        help='使用的联邦学习算法')
    parser.add_argument('--model', type=str, default='cnn', help='模型名称')
    parser.add_argument('--dataset', type=str, default='enhanced_mnist', help='数据集名称')
    parser.add_argument('--num_rounds', type=int, default=100, help='通信轮数')
    parser.add_argument('--eval_every', type=int, default=1, help='每隔多少轮评估一次')
    parser.add_argument('--clients_per_round', type=int, default=10, help='每轮选择的客户端数量')
    parser.add_argument('--batch_size', type=int, default=10, help='本地批次大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--drop_percent', type=float, default=0.0, help='掉线客户端百分比')
    
    # FedProx特定参数
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx的mu参数')
    
    # 增强版FedProx特定参数
    parser.add_argument('--use_enhanced', type=bool, default=True, help='是否使用增强版FedProx')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='模型选择相似度阈值')
    parser.add_argument('--reference_data_size', type=int, default=100, help='参考数据集大小')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    # 确保数据集存在
    create_dataset_if_not_exists()
    
    # 加载数据集
    print("加载数据集:", args.dataset)
    if args.dataset == 'enhanced_mnist':
        clients, groups, train_data, test_data = load_enhanced_mnist()
        train_data, test_data = preprocess_enhanced_mnist_data(train_data, test_data)
    else:
        # 使用原始FedProx数据加载方式
        train_path = os.path.join('data', args.dataset, 'data', 'train')
        test_path = os.path.join('data', args.dataset, 'data', 'test')
        clients, groups, train_data, test_data = read_data(train_path, test_path)
    
    # 创建模型
    if args.model == 'cnn':
        model = Model(args.seed, args.learning_rate)
    else:
        raise ValueError("不支持的模型")
    
    # 设置训练参数
    params = {
        'num_rounds': args.num_rounds,
        'eval_every': args.eval_every,
        'clients_per_round': args.clients_per_round,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'drop_percent': args.drop_percent,
        'mu': args.mu,
        'use_enhanced': args.use_enhanced,
        'similarity_threshold': args.similarity_threshold,
        'reference_data_size': args.reference_data_size
    }
    
    # 创建服务器
    if args.algorithm == 'fedavg':
        server = FedAvgServer(params, model, clients, groups, train_data, test_data)
    elif args.algorithm == 'fedprox':
        server = FedProxServer(params, model, clients, groups, train_data, test_data)
    else:
        raise ValueError("不支持的算法")
    
    # 开始训练
    print("开始训练...")
    server.train()
    
    # 保存结果
    save_results(server, args)

def save_results(server, args):
    """保存训练结果和可视化"""
    results_dir = f"results/{args.algorithm}_{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存准确率历史
    accuracy_history = []
    for stats in server.metrics.accuracies:
        accuracy = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        accuracy_history.append(accuracy)
    
    # 保存训练准确率历史
    train_accuracy_history = []
    for stats in server.metrics.train_accuracies:
        accuracy = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        train_accuracy_history.append(accuracy)
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracy_history)), accuracy_history, label='测试准确率')
    plt.plot(range(len(train_accuracy_history)), train_accuracy_history, label='训练准确率')
    plt.xlabel('通信轮数')
    plt.ylabel('准确率')
    plt.title(f'{args.algorithm} 在 {args.dataset} 上的准确率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
    
    # 保存参数和结果
    results = {
        'algorithm': args.algorithm,
        'dataset': args.dataset,
        'params': vars(args),
        'test_accuracy': accuracy_history,
        'train_accuracy': train_accuracy_history,
        'final_test_accuracy': accuracy_history[-1],
        'final_train_accuracy': train_accuracy_history[-1]
    }
    
    # 保存结果为JSON
    import json
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"结果已保存到 {results_dir}")

if __name__ == "__main__":
    main()