import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess

def run_experiments(dataset, drop_percent=0.0, mu=0.01):
    """
    运行原始FedProx和增强版FedProx实验
    
    参数:
        dataset: 数据集名称
        drop_percent: 掉线客户端百分比
        mu: FedProx的mu参数
    """
    # 运行原始FedProx
    print("运行原始FedProx...")
    cmd = f"python main.py --dataset={dataset} --optimizer=fedprox --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 --eval_every=1 --batch_size=10 --num_epochs=20 --model=mclr --drop_percent={drop_percent} --mu={mu}"
    subprocess.run(cmd, shell=True)
    
    # 运行增强版FedProx
    print("运行增强版FedProx...")
    cmd = f"python main_enhanced.py --dataset={dataset} --optimizer=fedprox --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 --eval_every=1 --batch_size=10 --num_epochs=20 --model=mclr --drop_percent={drop_percent} --mu={mu} --use_enhanced=true --similarity_threshold=0.5 --reference_data_size=100"
    subprocess.run(cmd, shell=True)

def compare_results(dataset, drop_percent=0.0, mu=0.01):
    """
    比较原始FedProx和增强版FedProx的结果
    
    参数:
        dataset: 数据集名称
        drop_percent: 掉线客户端百分比
        mu: FedProx的mu参数
    """
    # 加载原始FedProx结果
    original_path = f"results/fedprox_{dataset}/results.json"
    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            original_results = json.load(f)
    else:
        print(f"找不到原始FedProx结果: {original_path}")
        return
    
    # 加载增强版FedProx结果
    enhanced_path = f"results/fedprox_{dataset}_enhanced/results.json"
    if os.path.exists(enhanced_path):
        with open(enhanced_path, 'r') as f:
            enhanced_results = json.load(f)
    else:
        print(f"找不到增强版FedProx结果: {enhanced_path}")
        return
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    
    # 测试准确率
    plt.subplot(2, 1, 1)
    plt.plot(original_results['test_accuracy'], label='原始FedProx')
    plt.plot(enhanced_results['test_accuracy'], label='增强版FedProx')
    plt.xlabel('通信轮数')
    plt.ylabel('测试准确率')
    plt.title(f'FedProx vs 增强版FedProx 在 {dataset} 上的测试准确率比较')
    plt.legend()
    plt.grid(True)
    
    # 训练准确率
    plt.subplot(2, 1, 2)
    plt.plot(original_results['train_accuracy'], label='原始FedProx')
    plt.plot(enhanced_results['train_accuracy'], label='增强版FedProx')
    plt.xlabel('通信轮数')
    plt.ylabel('训练准确率')
    plt.title(f'FedProx vs 增强版FedProx 在 {dataset} 上的训练准确率比较')
    plt.legend()
    plt.grid(True)
    
    # 保存比较结果
    results_dir = f"results/comparison_{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
    
    # 打印最终准确率比较
    print("\n最终准确率比较:")
    print(f"原始FedProx 测试准确率: {original_results['final_test_accuracy']:.4f}")
    print(f"增强版FedProx 测试准确率: {enhanced_results['final_test_accuracy']:.4f}")
    print(f"提升: {(enhanced_results['final_test_accuracy'] - original_results['final_test_accuracy']) * 100:.2f}%")
    
    # 保存比较结果为JSON
    comparison = {
        'dataset': dataset,
        'drop_percent': drop_percent,
        'mu': mu,
        'original_fedprox': {
            'final_test_accuracy': original_results['final_test_accuracy'],
            'final_train_accuracy': original_results['final_train_accuracy']
        },
        'enhanced_fedprox': {
            'final_test_accuracy': enhanced_results['final_test_accuracy'],
            'final_train_accuracy': enhanced_results['final_train_accuracy']
        },
        'improvement': {
            'test_accuracy': enhanced_results['final_test_accuracy'] - original_results['final_test_accuracy'],
            'train_accuracy': enhanced_results['final_train_accuracy'] - original_results['final_train_accuracy']
        }
    }
    
    with open(os.path.join(results_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='enhanced_mnist', help='数据集名称')
    parser.add_argument('--drop_percent', type=float, default=0.0, help='掉线客户端百分比')
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx的mu参数')
    parser.add_argument('--run', action='store_true', help='是否运行实验')
    args = parser.parse_args()
    
    if args.run:
        run_experiments(args.dataset, args.drop_percent, args.mu)
    
    compare_results(args.dataset, args.drop_percent, args.mu)

if __name__ == "__main__":
    main()