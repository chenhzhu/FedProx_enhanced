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
            stats: Client training statistics 
            
        Returns:
            γKT weight factor (between 0 and 1)
        """
        if hasattr(self.server, 'dataset') and 'nist' in str(self.server.dataset):
            base_weight = 0.3  # reduce the base weight to increase differentiation
            
            if stats and isinstance(stats, dict) and 'num_samples' in stats:
                if stats['num_samples'] > 100:
                    base_weight += 0.1 
            
            gamma = base_weight + (1 - base_weight) * (similarity ** 1.5)
            return max(0.15, min(0.95, gamma))  
        else:
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
        if not client_solutions:
            return self.server.latest_model
        
        is_femnist = False
        if hasattr(self.server, 'dataset') and 'nist' in str(self.server.dataset):
            is_femnist = True
        
        try:
            model_structure = list(self.server.latest_model)
            
            num_solutions = len(client_solutions)
            if num_solutions == 0:
                return self.server.latest_model
            
            if is_femnist and client_similarities is not None:
                weights = []
                for idx, similarity in enumerate(client_similarities):
                    if client_weights is not None and idx < len(client_weights):
                        client_size = client_weights[idx]
                    else:
                        client_size = 1.0
                    
                    gamma_kt = self.calculate_gamma_kt(similarity, None)
                    
                    stats_info = None
                    if client_stats and idx < len(client_stats):
                        try:
                            if isinstance(client_stats[idx], dict):
                                stats_info = client_stats[idx]
                            elif isinstance(client_stats[idx], tuple) and len(client_stats[idx]) > 0:
                                stats_info = {'num_samples': client_stats[idx][0] if isinstance(client_stats[idx][0], (int, float)) else 0}
                        except Exception as e:
                            print(f"Warning: Error processing stats in aggregator for client {idx}: {e}")
                    
                    gamma_kt = self.calculate_gamma_kt(similarity, stats_info)
                    
                    # give extra reward to clients with large data
                    if client_size > np.median([w for w in client_weights if w is not None]):
                        weight = client_size * gamma_kt * 1.1
                    else:
                        weight = client_size * gamma_kt
                    weights.append(weight)
                
                # normalize the weights
                total_weight = sum(weights) + 1e-10  # avoid division by zero
                normalized_weights = [w / total_weight for w in weights]
                
                print(f"FEMNIST aggregation with dynamic weights: min={min(normalized_weights):.4f}, max={max(normalized_weights):.4f}, avg={sum(normalized_weights)/len(normalized_weights):.4f}")
            else:
                normalized_weights = [1.0 / num_solutions] * num_solutions
            
            # initialize the aggregated model
            aggregated_model = [np.zeros_like(param) for param in model_structure]
            
            # aggregate each client's solution
            for idx, solution in enumerate(client_solutions):
                # parse the solution format
                if isinstance(solution, tuple) and len(solution) == 2:
                    # standard format: (num_samples, params)
                    params = solution[1]
                else:
                    # use solution as params directly
                    params = solution
                
                # get the weight of this solution
                weight = normalized_weights[idx]
                
                for i, param in enumerate(params):
                    if i >= len(aggregated_model):
                        continue
                    
                    if is_femnist:
                        if i <= 2:  
                            layer_weight = weight * 1.2  
                        elif i == len(aggregated_model) - 1:
                            layer_weight = weight * 0.9  
                        else:
                            layer_weight = weight
                    
                    if isinstance(param, (int, float)):
                        safe_param = max(-1e15, min(1e15, param))
                        aggregated_model[i] += safe_param * layer_weight
                    elif isinstance(param, np.ndarray):
                        safe_param = np.clip(param, -1e15, 1e15)
                        aggregated_model[i] += safe_param * layer_weight
                    elif isinstance(param, list):
                        try:
                            arr = np.array(param, dtype=np.float64)
                            safe_arr = np.clip(arr, -1e15, 1e15)
                            aggregated_model[i] += safe_arr * layer_weight
                        except:
                            pass
                    elif isinstance(param, tuple):
                        # process each element in the tuple separately
                        try:
                            if isinstance(aggregated_model[i], tuple) and len(aggregated_model[i]) == len(param):
                                # create a new aggregated tuple
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
                                
                                # update the aggregated model
                                aggregated_model[i] = tuple(new_tuple)
                        except:
                            pass
            
            if is_femnist:
                for i in range(len(aggregated_model)):
                    if isinstance(aggregated_model[i], np.ndarray):
                        noise_scale = 1e-7 * np.mean(np.abs(aggregated_model[i]))
                        noise = np.random.normal(0, noise_scale, aggregated_model[i].shape)
                        aggregated_model[i] = aggregated_model[i] + noise
            
            return aggregated_model
            
        except Exception as e:
            print(f"Warning: Error during aggregation: {e}")
            return self.server.latest_model 
