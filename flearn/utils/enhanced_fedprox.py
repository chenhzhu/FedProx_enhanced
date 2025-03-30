import numpy as np
from flearn.utils.fltrust_model_selection import FLTrustModelSelector
from flearn.utils.dynamic_aggregation import DynamicAggregator

class EnhancedFedProx:
    """
    Implementation of Enhanced FedProx with FLTrust-inspired model selection
    and dynamic aggregation weights.
    """
    def __init__(self, server, similarity_threshold=0.5, reference_data_size=100):
        """
        Initialize the Enhanced FedProx module.
        
        Args:
            server: The server instance
            similarity_threshold: Threshold for model selection
            reference_data_size: Size of reference dataset
        """
        self.server = server
        self.model_selector = FLTrustModelSelector(server, similarity_threshold, reference_data_size)
        self.aggregator = DynamicAggregator(server)
    
    def enhance_training_round(self, round_num, active_clients, selected_clients):
        """
        执行一轮增强的FedProx训练
        
        参数:
            round_num: 当前轮次
            active_clients: 活跃的客户端
            selected_clients: 选中的客户端
            
        返回:
            更新后的模型
        """
        # 收集客户端解决方案
        solutions = []
        client_weights = []
        stats_list = []
        
        for c in selected_clients.tolist():
            # 发送最新模型
            c.set_params(self.server.latest_model)
            
            # 本地求解
            if c in active_clients:
                soln, stats = c.solve_inner(num_epochs=self.server.num_epochs, batch_size=self.server.batch_size)
            else:
                # 为掉线客户端训练更少的轮次
                epochs = np.random.randint(low=1, high=self.server.num_epochs)
                soln, stats = c.solve_inner(num_epochs=epochs, batch_size=self.server.batch_size)
            
            # 收集解决方案和权重
            solutions.append((c.num_samples, soln))
            client_weights.append(c.num_samples)
            stats_list.append(stats)
            
            # 跟踪通信成本
            self.server.metrics.update(rnd=round_num, cid=c.id, stats=stats)
        
        # 模型选择：过滤不可靠的客户端
        reliable_solutions, similarities = self.model_selector.select_reliable_clients(
            self.server.latest_model, solutions, stats_list
        )
        
        # 聚合可靠的解决方案
        client_weights = [soln[0] for soln in reliable_solutions]  # 获取样本数量
        aggregated_model = self.aggregator.dynamic_aggregate(
            reliable_solutions, client_weights, similarities, stats_list
        )
        
        # 返回聚合模型和客户端信息
        return aggregated_model, {
            'reliable_clients': len(reliable_solutions),
            'total_clients': len(solutions),
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0
        } 