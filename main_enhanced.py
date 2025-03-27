import numpy as np
import argparse
import importlib
import random
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
from flearn.utils.model_utils import read_data
from flearn.utils.enhanced_dataset_loader import load_enhanced_mnist, preprocess_enhanced_mnist_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 'enhanced_mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # 添加了enhanced_mnist

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'enhanced_mnist.mclr': (10,), # num_classes - 添加增强版MNIST的参数
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ) # num_classes
}

def create_dataset_if_not_exists():
    """如果增强数据集不存在，则创建它"""
    if not os.path.exists("data/enhanced_mnist"):
        print("增强版MNIST数据集不存在，正在创建...")
        from utils.create_enhanced_dataset import create_enhanced_mnist_dataset
        create_enhanced_mnist_dataset()
    else:
        print("增强版MNIST数据集已存在")

def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedprox')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='enhanced_mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=50)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.0)
    parser.add_argument('--use_enhanced', 
                        type=str, 
                        default='true', 
                        help='use enhanced FedProx')
    parser.add_argument('--similarity_threshold', 
                        type=float, 
                        default=0.5, 
                        help='similarity threshold for model selection')
    parser.add_argument('--reference_data_size', 
                        type=int, 
                        default=100, 
                        help='reference dataset size')

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])  # 使用TF1.x的随机种子设置方法

    # 确保增强版MNIST数据集存在
    if parsed['dataset'] == 'enhanced_mnist':
        create_dataset_if_not_exists()

    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'].replace('enhanced_', ''), parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:]).replace('enhanced_', '')]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    # convert string 'true'/'false' to boolean
    use_enhanced = (parsed['use_enhanced'].lower() == 'true')

    # add new parameters to params dictionary
    params = {
        'dataset': parsed['dataset'],
        'optimizer': parsed['optimizer'],
        'learning_rate': parsed['learning_rate'],
        'num_rounds': parsed['num_rounds'],
        'clients_per_round': parsed['clients_per_round'],
        'eval_every': parsed['eval_every'],
        'batch_size': parsed['batch_size'], 
        'num_epochs': parsed['num_epochs'],
        'model': parsed['model'],
        'drop_percent': parsed['drop_percent'],
        'mu': parsed['mu'],
        'use_enhanced': use_enhanced,
        'similarity_threshold': parsed['similarity_threshold'],
        'reference_data_size': parsed['reference_data_size']
    }

    return parsed, learner, optimizer, params

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer, params = read_options()

    # read data
    if options['dataset'] == 'enhanced_mnist':
        clients, groups, train_data, test_data = load_enhanced_mnist()
        train_data, test_data = preprocess_enhanced_mnist_data(train_data, test_data)
        dataset = (clients, groups, train_data, test_data)
    else:
        # 使用原始FedProx数据加载方式
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()
    
    # 保存结果
    save_results(t, options)

def save_results(server, options):
    """保存训练结果和可视化"""
    import matplotlib.pyplot as plt
    
    results_dir = f"results/{options['optimizer']}_{options['dataset']}"
    if options['use_enhanced'].lower() == 'true':
        results_dir += "_enhanced"
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
    plt.title(f"{options['optimizer']} 在 {options['dataset']} 上的准确率")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
    
    # 保存参数和结果
    results = {
        'algorithm': options['optimizer'],
        'dataset': options['dataset'],
        'params': options,
        'test_accuracy': accuracy_history,
        'train_accuracy': train_accuracy_history,
        'final_test_accuracy': accuracy_history[-1] if accuracy_history else 0,
        'final_train_accuracy': train_accuracy_history[-1] if train_accuracy_history else 0
    }
    
    # 保存结果为JSON
    import json
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"结果已保存到 {results_dir}")

if __name__ == "__main__":
    main()