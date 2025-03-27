import numpy as np
import tensorflow as tf
from flearn.utils.tf_utils import process_grad

class FLTrustModelSelector:
    """
    Implementation of FLTrust-inspired model selection approach.
    This class handles selection of reliable client models based on gradient similarity.
    """
    def __init__(self, server, similarity_threshold=0.5, reference_data_size=100):
        """
        Initialize the model selector.
        
        Args:
            server: The server instance
            similarity_threshold: Threshold for cosine similarity to select models
            reference_data_size: Size of reference dataset to create
        """
        self.server = server
        self.similarity_threshold = similarity_threshold
        self.reference_data_size = reference_data_size
        self.reference_dataset = self._create_reference_dataset()
        
    def _create_reference_dataset(self):
        """Create a small reference dataset for gradient computation"""
        # 从server的数据集中随机选择一部分作为参考数据集
        # 实际实现会依赖于您数据集的具体结构
        if hasattr(self.server, 'dataset') and self.server.dataset is not None:
            # 假设dataset有一个采样方法或者能够访问
            # 以下是一个示意性实现，具体需要根据您的dataset结构调整
            try:
                all_data = self.server.dataset.get_data()
                indices = np.random.choice(len(all_data), 
                                         min(self.reference_data_size, len(all_data)), 
                                         replace=False)
                return [all_data[i] for i in indices]
            except:
                print("Warning: Failed to create reference dataset. Using empty set.")
                return []
        return []
    
    def calculate_reference_gradient(self, model):
        """Calculate gradient on reference dataset"""
        if not self.reference_dataset:
            print("Warning: Reference dataset is empty. Cannot calculate reference gradient.")
            return None
            
        # 在参考数据集上计算梯度
        # 实际实现会依赖于您模型的具体结构和TensorFlow版本
        # 以下是一个示意性实现
        try:
            with tf.GradientTape() as tape:
                # 计算参考数据集上的损失
                loss = self.server.client_model.loss(self.reference_dataset)
            # 计算梯度
            grads = tape.gradient(loss, model)
            return process_grad(grads)
        except:
            print("Warning: Failed to calculate reference gradient.")
            return None
    
    def calculate_cosine_similarity(self, grad1, grad2):
        """Calculate cosine similarity between two gradients"""
        if grad1 is None or grad2 is None:
            return 0.0
            
        # 将梯度扁平化为向量以计算余弦相似度
        try:
            flat_grad1 = grad1.flatten() if hasattr(grad1, 'flatten') else grad1
            flat_grad2 = grad2.flatten() if hasattr(grad2, 'flatten') else grad2
            
            norm1 = np.linalg.norm(flat_grad1)
            norm2 = np.linalg.norm(flat_grad2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(flat_grad1, flat_grad2) / (norm1 * norm2)
            return max(0.0, similarity)  # 保证相似度非负
        except:
            print("Warning: Failed to calculate cosine similarity.")
            return 0.0
    
    def calculate_client_gradient(self, old_model, new_model):
        """Calculate gradient between client's old and new models"""
        try:
            gradient = []
            for i in range(len(old_model)):
                if hasattr(new_model[i], 'numpy') and hasattr(old_model[i], 'numpy'):
                    gradient.append(new_model[i].numpy() - old_model[i].numpy())
                else:
                    gradient.append(new_model[i] - old_model[i])
            return np.concatenate([g.flatten() for g in gradient])
        except:
            print("Warning: Failed to calculate client gradient.")
            return None
    
    def select_reliable_clients(self, old_model, client_solutions):
        """Select reliable clients based on gradient similarity to reference"""
        reference_grad = self.calculate_reference_gradient(old_model)
        
        selected_solutions = []
        client_similarities = []
        
        for client_soln in client_solutions:
            client_grad = self.calculate_client_gradient(old_model, client_soln)
            similarity = self.calculate_cosine_similarity(reference_grad, client_grad)
            
            if similarity >= self.similarity_threshold:
                selected_solutions.append(client_soln)
                client_similarities.append(similarity)
                
        return selected_solutions, client_similarities 