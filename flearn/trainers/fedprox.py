import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.utils.enhanced_fedprox import EnhancedFedProx


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Enhanced Federated Prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        
        # enhanced FedProx parameters
        self.use_enhanced = params.get('use_enhanced', True)
        self.similarity_threshold = params.get('similarity_threshold', 0.5)
        self.reference_data_size = params.get('reference_data_size', 100)
        
        super(Server, self).__init__(params, learner, dataset)
        
        # initialize enhanced FedProx module
        if self.use_enhanced:
            self.enhanced_fedprox = EnhancedFedProx(
                self, 
                similarity_threshold=self.similarity_threshold,
                reference_data_size=self.reference_data_size
            )
            print(f'Enhanced FedProx enabled with similarity threshold: {self.similarity_threshold}')

    def train(self):
        '''Train using Enhanced Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # 测试准确率
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

            # calculate global gradient and client gradient difference
            model_len = process_grad(self.latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))

            # select clients
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # ensure FedProx and FedAvg stragglers are the same
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            # set parameters
            self.inner_opt.set_params(self.latest_model, self.client_model)

            # use enhanced FedProx module or original FedProx
            if self.use_enhanced:
                # enhanced FedProx
                self.latest_model, client_info = self.enhanced_fedprox.enhance_training_round(
                    i, active_clients, selected_clients
                )
                # output enhanced FedProx information
                tqdm.write('Enhanced FedProx: {} of {} clients selected as reliable (avg similarity: {:.4f})'.format(
                    client_info['reliable_clients'], 
                    client_info['total_clients'],
                    client_info['avg_similarity']
                ))
            else:
                # original FedProx
                csolns = [] # buffer for receiving client solutions
                
                for idx, c in enumerate(selected_clients.tolist()):
                    # send latest model
                    c.set_params(self.latest_model)
                    
                    total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)
                    
                    # solve minimization problem locally
                    if c in active_clients:
                        soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                    else:
                        #soln, stats = c.solve_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size)
                        soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)
                    
                    # collect solutions from clients
                    csolns.append((c.num_samples, soln))
                    
                    # track communication cost
                    self.metrics.update(rnd=i, cid=c.id, stats=stats)
                
                # update model
                self.latest_model = self.aggregate(csolns)
            
            # set client model parameters
            self.client_model.set_params(self.latest_model)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
