import numpy as np

class DynamicAggregator:
    """
    Implementation of dynamic aggregation for FedProx.
    This class handles the calculation of γKT parameters and dynamic model aggregation.
    """
    def __init__(self, server):
        """
        Initialize the dynamic aggregator.
        
        Args:
            server: The server instance
        """
        self.server = server
    
    def calculate_gamma_kt(self, similarity, stats):
        """
        Calculate dynamic aggregation weight (γKT) based on model similarity and training stats.
        
        Args:
            similarity: Cosine similarity between client gradient and reference gradient
            stats: Client training statistics
        
        Returns:
            γKT weight for client aggregation
        """
        # γKT could be function of similarity, local loss, local gradient norm, etc.
        try:
            # 基础权重是相似度
            gamma_kt = similarity
            
            # 可以根据客户端的损失调整权重
            if 'loss' in stats and stats['loss'] > 0:
                # 损失越低，权重越高
                gamma_kt *= (1.0 / (1.0 + stats['loss']))
            
            # 可以根据客户端的准确率调整权重
            if 'acc' in stats and stats['acc'] > 0:
                # 准确率越高，权重越高
                gamma_kt *= (1.0 + stats['acc'])
                
            return max(0.1, gamma_kt)  # 确保权重有一个下限
        except:
            print("Warning: Failed to calculate gamma_kt. Using default value.")
            return 1.0  # 返回默认值
    
    def dynamic_aggregate(self, client_solutions, client_weights, client_similarities, client_stats):
        """
        Aggregate client solutions with dynamic weights based on γKT.
        
        Args:
            client_solutions: List of client model parameters
            client_weights: List of client dataset sizes
            client_similarities: List of client gradient similarities
            client_stats: List of client training statistics
            
        Returns:
            Aggregated model parameters
        """
        if not client_solutions:
            return self.server.latest_model  # 如果没有可用解决方案，返回最新模型
        
        total_weight = 0.0
        base = [0] * len(client_solutions[0][1])  # 初始化聚合基础
        
        for idx, (w, soln) in enumerate(client_solutions):
            # 计算动态权重γKT
            similarity = client_similarities[idx] if idx < len(client_similarities) else 1.0
            stats = client_stats[idx] if idx < len(client_stats) else {}
            gamma_kt = self.calculate_gamma_kt(similarity, stats)
            
            # 应用γKT权重
            weight = w * gamma_kt  # 客户端数据量 * γKT
            total_weight += weight
            
            for i, v in enumerate(soln):
                # 修改这里：检查v是否为标量，如果是则直接使用，不调用astype
                if isinstance(v, (int, float)):
                    base[i] += weight * v
                else:
                    print(f"var v is a {type(v)} with length {len(v)}")
                    base[i] += weight * v.astype(np.float64)
        
        if total_weight == 0:
            return self.server.latest_model  # 防止除零错误
        
        # 计算加权平均
        averaged_soln = [v / total_weight for v in base]
        return averaged_soln 