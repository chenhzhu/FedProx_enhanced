import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_results(algorithm, dataset, enhanced=False):
    """加载实验结果"""
    if enhanced:
        results_dir = f"results/{algorithm}_{dataset}_enhanced"
    else:
        results_dir = f"results/{algorithm}_{dataset}"
    
    with open(os.path.join(results_dir, 'results.json'), 'r') as f:
        results = json.load(f)
    
    return results

def compare_algorithms(dataset='enhanced_mnist'):
    """比较原始FedProx和增强版FedProx的性能"""
    # 加载结果
    original_results = load_results('fedprox', dataset, enhanced=False)
    enhanced_results = load_results('fedprox', dataset, enhanced=True)
    
    # 提取准确率历史
    original_acc = original_results['test_accuracy']
    enhanced_acc = enhanced_results['test_accuracy']
    
    # 确保长度一致
    min_len = min(len(original_acc), len(enhanced_acc))
    original_acc = original_acc[:min_len]
    enhanced_acc = enhanced_acc[:min_len]
    
    # 计算性能提升
    improvement = np.array(enhanced_acc) - np.array(original_acc)
    avg_improvement = np.mean(improvement)
    final_improvement = enhanced_acc[-1] - original_acc[-1]
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    
    # 准确率对比
    plt.subplot(2, 1, 1)
    plt.plot(range(min_len), original_acc, label='原始FedProx')
    plt.plot(range(min_len), enhanced_acc, label='增强版FedProx')
    plt.xlabel('通信轮数')
    plt.ylabel('测试准确率')
    plt.title(f'FedProx vs 增强版FedProx 在 {dataset} 上的准确率对比')
    plt.legend()
    plt.grid(True)
    
    # 性能提升
    plt.subplot(2, 1, 2)
    plt.bar(range(min_len), improvement)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('通信轮数')
    plt.ylabel('准确率提升')
    plt.title(f'增强版FedProx相对于原始FedProx的性能提升 (平均: {avg_improvement:.4f}, 最终: {final_improvement:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存结果
    results_dir = f"results/comparison_{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comparison.png'))
    
    # 保存详细比较结果
    comparison = {
        'dataset': dataset,
        'original_fedprox': {
            'final_accuracy': original_acc[-1],
            'params': original_results['params']
        },
        'enhanced_fedprox': {
            'final_accuracy': enhanced_acc[-1],
            'params': enhanced_results['params']
        },
        'improvement': {
            'average': float(avg_improvement),
            'final': float(final_improvement),
            'percentage': float(final_improvement / original_acc[-1] * 100)
        }
    }
    
    with open(os.path.join(results_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"比较结果已保存到 {results_dir}")
    print(f"平均准确率提升: {avg_improvement:.4f}")
    print(f"最终准确率提升: {final_improvement:.4f} ({final_improvement / original_acc[-1] * 100:.2f}%)")
    
    return comparison

# def run_experiments(dataset='enhanced_mnist'):
#     """运行原始FedProx和增强版FedProx实验"""
#     # 运行原始FedProx
#     print("运行原始FedProx...")
#     os.system(f"python main_enhanced.py --algorithm fedprox --dataset {dataset} --use_enhanced False --num_rounds 50")
    
#     # 运行增强版FedProx
#     print("运行增强版FedProx...")
#     os.system(f"python main_enhanced.py --algorithm fedprox --dataset {dataset} --use_enhanced True --num_rounds 50")
    
#     # 比较结果
#     return compare_algorithms(dataset)

def run_experiments(dataset='enhanced_mnist'):
    """运行原始FedProx和增强版FedProx实验"""
    # 运行原始FedProx
    print("运行原始FedProx...")
    os.system(f"python main_enhanced.py --algorithm fedprox --dataset {dataset} --use_enhanced False --model mclr --num_rounds 50")
    
    # 运行增强版FedProx
    print("运行增强版FedProx...")
    os.system(f"python main_enhanced.py --algorithm fedprox --dataset {dataset} --use_enhanced True --model mclr --num_rounds 50")
    
    # 比较结果
    return compare_algorithms(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='enhanced_mnist', help='数据集名称')
    parser.add_argument('--run', action='store_true', help='是否运行实验')
    args = parser.parse_args()
    
    if args.run:
        run_experiments(args.dataset)
    else:
        compare_algorithms(args.dataset)