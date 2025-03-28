import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ) # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)
    parser.add_argument('--use_enhanced', type=str, default='false', help='use enhanced FedProx')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='similarity threshold for model selection')
    parser.add_argument('--reference_data_size', type=int, default=100, help='reference dataset size')


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.random.set_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

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
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer, params = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(params, learner, dataset)
    t.train()
    
if __name__ == '__main__':
    main()
