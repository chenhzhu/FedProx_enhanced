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
    
    def enhance_training_round(self, round_idx, active_clients, selected_clients):
        """
        Enhance a single round of FedProx training with advanced model selection and aggregation.
        
        Args:
            round_idx: Current training round index
            active_clients: List of active clients
            selected_clients: List of selected clients
            
        Returns:
            Tuple of (aggregated_model, selected_clients_info)
        """
        csolns = []  # client solutions buffer
        client_stats = []  # client training stats buffer
        
        # train each client
        for idx, c in enumerate(selected_clients.tolist()):
            # send latest model
            c.set_params(self.server.latest_model)
            
            total_iters = int(self.server.num_epochs * c.num_samples / self.server.batch_size) + 2
            
            # solve minimization problem
            if c in active_clients:
                # full training
                soln, stats = c.solve_inner(num_epochs=self.server.num_epochs, 
                                           batch_size=self.server.batch_size)
            else:
                # partial training (slow devices)
                soln, stats = c.solve_inner(
                    num_epochs=np.random.randint(low=1, high=self.server.num_epochs), 
                    batch_size=self.server.batch_size
                )
            
            # collect client solutions
            csolns.append((c.num_samples, soln))
            client_stats.append(stats)
            
            # track communication cost
            self.server.metrics.update(rnd=round_idx, cid=c.id, stats=stats)
        
        # use FLTrust method to select reliable client models
        reliable_solutions, client_similarities = self.model_selector.select_reliable_clients(
            self.server.latest_model, csolns
        )
        
        # if no reliable solutions, use all solutions
        if not reliable_solutions:
            print("Warning: No reliable solutions found. Using all solutions.")
            reliable_solutions = csolns
            client_similarities = [1.0] * len(csolns)
        
        # use dynamic aggregation
        aggregated_model = self.aggregator.dynamic_aggregate(
            reliable_solutions, 
            [c.num_samples for c in selected_clients], 
            client_similarities,
            client_stats
        )
        
        # collect selected clients info, for recording and analysis
        selected_clients_info = {
            'total_clients': len(selected_clients),
            'reliable_clients': len(reliable_solutions),
            'avg_similarity': np.mean(client_similarities) if client_similarities else 0
        }
        
        return aggregated_model, selected_clients_info 