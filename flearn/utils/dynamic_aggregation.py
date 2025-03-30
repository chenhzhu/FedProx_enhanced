import numpy as np

class DynamicAggregator:
    """
    Dynamic aggregator for weighted model averaging with client similarity metrics.
    """
    
    def __init__(self, server):
        """
        Initialize the dynamic aggregator.
        
        Args:
            server: The server instance
        """
        self.server = server
    
    def calculate_gamma_kt(self, similarity, stats=None):
        """
        Calculate the dynamic weight factor γKT based on gradient similarity.
        
        Args:
            similarity: Cosine similarity between client and server gradients
            stats: Client training statistics (可能是字典或其他类型)
            
        Returns:
            γKT weight factor (between 0 and 1)
        """
        if hasattr(self.server, 'dataset') and 'nist' in str(self.server.dataset):
            # FEMNIST特定优化
            base_weight = 0.3  # 降低基础权重以增加差异化
            
            # 如果有有效的统计信息，进行额外调整
            if stats and isinstance(stats, dict) and 'num_samples' in stats:
                if stats['num_samples'] > 100:
                    base_weight += 0.1  # 更大幅度提高基础权重
            
            # 使用二次函数提高相似度权重差异
            gamma = base_weight + (1 - base_weight) * (similarity ** 1.5)
            return max(0.15, min(0.95, gamma))  # 更宽的权重范围
        else:
            # 其他数据集使用原始逻辑
            return max(0.0, min(1.0, similarity))
    
    def dynamic_aggregate(self, client_solutions, client_weights=None, client_similarities=None, client_stats=None):
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
        # 如果没有解决方案，直接返回最新模型
        if not client_solutions:
            return self.server.latest_model
        
        # 检测数据集类型
        is_femnist = False
        if hasattr(self.server, 'dataset') and 'nist' in str(self.server.dataset):
            is_femnist = True
        
        try:
            # 确保我们有合适的模型结构
            model_structure = list(self.server.latest_model)
            
            # 获取客户端解决方案数量
            num_solutions = len(client_solutions)
            if num_solutions == 0:
                return self.server.latest_model
            
            # 准备加权聚合
            if is_femnist and client_similarities is not None:
                weights = []
                for idx, similarity in enumerate(client_similarities):
                    if client_weights is not None and idx < len(client_weights):
                        client_size = client_weights[idx]
                    else:
                        client_size = 1.0
                    
                    # FEMNIST特定：考虑客户端数据量的影响
                    gamma_kt = self.calculate_gamma_kt(similarity, None)
                    
                    # 尝试从client_stats获取信息
                    stats_info = None
                    if client_stats and idx < len(client_stats):
                        try:
                            if isinstance(client_stats[idx], dict):
                                # 字典类型
                                stats_info = client_stats[idx]
                            elif isinstance(client_stats[idx], tuple) and len(client_stats[idx]) > 0:
                                # 元组类型，尝试创建一个基本的字典
                                stats_info = {'num_samples': client_stats[idx][0] if isinstance(client_stats[idx][0], (int, float)) else 0}
                        except Exception as e:
                            print(f"Warning: Error processing stats in aggregator for client {idx}: {e}")
                    
                    # 使用处理后的stats_info
                    gamma_kt = self.calculate_gamma_kt(similarity, stats_info)
                    
                    # 对大数据量客户端给予额外奖励
                    if client_size > np.median([w for w in client_weights if w is not None]):
                        weight = client_size * gamma_kt * 1.1  # 10%的额外权重
                    else:
                        weight = client_size * gamma_kt
                    weights.append(weight)
                
                # 归一化权重
                total_weight = sum(weights) + 1e-10  # 避免除零
                normalized_weights = [w / total_weight for w in weights]
                
                # 打印权重信息
                print(f"FEMNIST aggregation with dynamic weights: min={min(normalized_weights):.4f}, max={max(normalized_weights):.4f}, avg={sum(normalized_weights)/len(normalized_weights):.4f}")
            else:
                # 对于其他数据集，使用等权重
                normalized_weights = [1.0 / num_solutions] * num_solutions
            
            # 初始化聚合模型
            aggregated_model = [np.zeros_like(param) for param in model_structure]
            
            # 聚合每个客户端的解决方案
            for idx, solution in enumerate(client_solutions):
                # 解析解决方案格式
                if isinstance(solution, tuple) and len(solution) == 2:
                    # 标准格式: (num_samples, params)
                    params = solution[1]
                else:
                    # 直接使用solution作为params
                    params = solution
                
                # 获取此解决方案的权重
                weight = normalized_weights[idx]
                
                # 将每个参数添加到聚合模型中
                for i, param in enumerate(params):
                    if i >= len(aggregated_model):
                        continue
                    
                    # FEMNIST特定：层级权重调整
                    if is_femnist:
                        if i <= 2:  
                            layer_weight = weight * 1.2  # 提升20%
                        # 最后一层（分类层）权重降低
                        elif i == len(aggregated_model) - 1:
                            layer_weight = weight * 0.9  # 降低10%
                        else:
                            layer_weight = weight
                    
                    # 处理不同类型的参数
                    if isinstance(param, (int, float)):
                        # 数值稳定性：限制极端值
                        safe_param = max(-1e15, min(1e15, param))
                        aggregated_model[i] += safe_param * layer_weight
                    elif isinstance(param, np.ndarray):
                        # 对numpy数组应用数值稳定性处理
                        safe_param = np.clip(param, -1e15, 1e15)
                        aggregated_model[i] += safe_param * layer_weight
                    elif isinstance(param, list):
                        # 转换列表为numpy数组进行安全处理
                        try:
                            arr = np.array(param, dtype=np.float64)
                            safe_arr = np.clip(arr, -1e15, 1e15)
                            aggregated_model[i] += safe_arr * layer_weight
                        except:
                            # 如果无法转换，跳过此参数
                            pass
                    elif isinstance(param, tuple):
                        # 对元组中的每个元素单独处理
                        try:
                            if isinstance(aggregated_model[i], tuple) and len(aggregated_model[i]) == len(param):
                                # 创建新的聚合元组
                                new_tuple = []
                                for j, p in enumerate(param):
                                    if isinstance(p, np.ndarray):
                                        safe_p = np.clip(p, -1e15, 1e15)
                                        if j < len(aggregated_model[i]):
                                            agg_item = aggregated_model[i][j] + safe_p * layer_weight
                                        else:
                                            agg_item = safe_p * layer_weight
                                        new_tuple.append(agg_item)
                                    elif isinstance(p, (int, float)):
                                        safe_p = max(-1e15, min(1e15, p))
                                        if j < len(aggregated_model[i]):
                                            agg_item = aggregated_model[i][j] + safe_p * layer_weight
                                        else:
                                            agg_item = safe_p * layer_weight
                                        new_tuple.append(agg_item)
                                
                                # 更新聚合模型
                                aggregated_model[i] = tuple(new_tuple)
                        except:
                            # 如果处理失败，保持不变
                            pass
            
            # FEMNIST特定：添加自适应噪声
            if is_femnist:
                for i in range(len(aggregated_model)):
                    if isinstance(aggregated_model[i], np.ndarray):
                        # 添加自适应噪声
                        noise_scale = 1e-7 * np.mean(np.abs(aggregated_model[i]))
                        noise = np.random.normal(0, noise_scale, aggregated_model[i].shape)
                        aggregated_model[i] = aggregated_model[i] + noise
            
            return aggregated_model
            
        except Exception as e:
            print(f"Warning: Error during aggregation: {e}")
            return self.server.latest_model 