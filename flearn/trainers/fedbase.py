import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        # 更新: 使用tf.compat.v1.reset_default_graph()替代tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()
        # self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        # self.clients = self.setup_clients(dataset, self.client_model)
        # print('{} Clients in Total'.format(len(self.clients)))
        # self.latest_model = self.client_model.get_params()
        # 确保client_model正确初始化
        try:
            self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
            self.clients = self.setup_clients(dataset, self.client_model)
            print('{} Clients in Total'.format(len(self.clients)))
            self.latest_model = self.client_model.get_params()
        except Exception as e:
            print(f"初始化client_model时出错: {e}")
            raise

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        """获取训练错误率和损失"""
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            try:
                # 尝试获取标准返回值
                ct, cl, ns = c.train_error_and_loss()
            except ValueError:
                # 如果返回值不是3个，尝试处理
                try:
                    # 假设返回了两个值：正确数和损失
                    ct, cl = c.train_error_and_loss()
                    ns = getattr(c, 'num_samples', 0)
                    if ns == 0:
                        # 如果无法获取样本数，使用默认值
                        print(f"Warning: Client {c.id} train_error_and_loss() returned only 2 values and no num_samples attribute.")
                        # 尝试从训练数据获取样本数
                        if hasattr(c, 'train_data') and hasattr(c.train_data, '__len__'):
                            ns = len(c.train_data)
                        else:
                            ns = 1  # 默认值
                except Exception as e:
                    print(f"Error in train_error_and_loss for client {c.id}: {e}")
                    ct, cl, ns = 0, 0, 0  # 出错时使用默认值
            
            # 处理ct可能是字典的情况
            if isinstance(ct, dict):
                if 'accuracy' in ct:
                    # 如果字典包含准确率，转换为正确预测数
                    accuracy = ct['accuracy']
                    correct_samples = accuracy * ns
                    ct = correct_samples
                else:
                    # 如果字典中没有准确率，使用0
                    print(f"Warning: Client {c.id} returned a dict without 'accuracy' key in train_error_and_loss(). Using 0.")
                    ct = 0
            
            # 确保所有值都是数值类型
            tot_correct.append(float(ct))
            num_samples.append(ns)
            losses.append(float(cl))
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''Tests self.latest_model on given clients'''
        num_samples = []
        tot_correct = []
        losses = []
        
        for c in self.clients:
            try:
                # 尝试获取3个返回值
                ct, cl, ns = c.test()
            except ValueError:
                # 如果只返回2个值，假设缺少的是ns（样本数）
                try:
                    ct, cl = c.test()
                    # 尝试从客户端属性获取样本数
                    ns = getattr(c, 'num_samples', 0)
                    if ns == 0:
                        # 如果客户端没有num_samples属性，使用一个默认值
                        print(f"Warning: Client {c.id} test() returned only 2 values and no num_samples attribute. Using default.")
                        # 尝试从测试数据中获取样本数
                        if hasattr(c, 'eval_data') and hasattr(c.eval_data, '__len__'):
                            ns = len(c.eval_data)
                        else:
                            ns = 1  # 如果无法确定样本数，使用1作为默认值
                except Exception as e:
                    print(f"Error testing client {c.id}: {e}")
                    ct, cl, ns = 0, 0, 0  # 出错时使用默认值
            
            # 确保ct是数值而不是字典
            if isinstance(ct, dict):
                if 'accuracy' in ct:
                    accuracy = ct['accuracy']
                    correct_samples = accuracy * ns
                    ct = correct_samples
                else:
                    print(f"Warning: Client {c.id} returned a dict without 'accuracy' key. Using 0.")
                    ct = 0
            
            tot_correct.append(float(ct))
            num_samples.append(ns)
            losses.append(float(cl))
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        
        return ids, groups, num_samples, tot_correct, losses

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        """
        Aggregate weight solutions from selected clients
        
        Args:
            wsolns: List of (weight, solution) tuples from clients
            
        Returns:
            Aggregated solution
        """
        total_weight = 0.0
        base = [0] * len(self.latest_model)
        
        for (w, soln) in wsolns:  # w is the weight, soln is the solution
            # 确保 w 是一个数值
            if not isinstance(w, (int, float)):
                print(f"Warning: Expected numeric weight, got {type(w)}. Converting to float.")
                try:
                    w = float(w)
                except:
                    print(f"Error converting weight to float. Using weight=1.0")
                    w = 1.0
            
            total_weight += w
            
            # 遍历解决方案中的每个参数
            for i, v in enumerate(soln):
                if i >= len(base) or i >= len(self.latest_model):
                    continue  # 跳过超出范围的参数
                    
                # 确保 v 是可以进行数值运算的类型
                if isinstance(v, np.ndarray):
                    # 对数组使用 astype
                    base[i] += w * v.astype(np.float64)
                elif isinstance(v, (int, float)):
                    # 对标量直接进行数值计算
                    base[i] += w * float(v)
                else:
                    # 对其他类型，尝试转换后计算
                    try:
                        base[i] += w * np.array(v, dtype=np.float64)
                    except:
                        print(f"Warning: Could not process parameter at index {i} with type {type(v)}")
        
        # 检查总权重是否为零，避免除零错误
        if total_weight == 0:
            return self.latest_model
        
        # 按总权重归一化
        for i in range(len(base)):
            if isinstance(base[i], np.ndarray):
                base[i] = base[i] / total_weight
            elif isinstance(base[i], (int, float)):
                base[i] = base[i] / total_weight
            else:
                try:
                    base[i] = base[i] / total_weight
                except:
                    base[i] = self.latest_model[i]  # 如果归一化失败，使用最新模型的参数
        
        return base

