import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''
    
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.random.set_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.compat.v1.train.Saver
        self.sess = tf.compat.v1.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            metadata = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.compat.v1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        """创建模型"""
        # 输入占位符
        features = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name='labels')
        
        # 创建模型
        dense_layer = tf.compat.v1.keras.layers.Dense(
            units=self.num_classes,
            kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.001)
        )
        
        # 应用层到输入
        logits = dense_layer(features)
        
        # 计算损失
        loss = tf.reduce_mean(
            tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
        )
        
        # 预测和准确率
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        
        # 定义评估指标
        eval_metric_ops = {
            'accuracy': tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        }
        
        # 创建训练操作
        train_op = optimizer.minimize(loss)
        
        # 计算梯度
        grads = tf.gradients(loss, tf.compat.v1.trainable_variables())
        
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        """设置模型参数，添加形状检查和错误处理"""
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                
                # 确保 model_params 长度与变量数量匹配
                if len(model_params) != len(all_vars):
                    print(f"警告: 参数数量不匹配! 模型需要 {len(all_vars)} 个参数，但提供了 {len(model_params)} 个")
                    # 只处理能够匹配的部分
                    zipped_vars = zip(all_vars, model_params[:len(all_vars)]) if len(model_params) > len(all_vars) else zip(all_vars[:len(model_params)], model_params)
                else:
                    zipped_vars = zip(all_vars, model_params)
                
                for variable, value in zipped_vars:
                    # 检查形状是否匹配
                    var_shape = variable.get_shape().as_list()
                    
                    # 尝试将 value 转换为 numpy 数组以便检查形状
                    try:
                        if not isinstance(value, np.ndarray):
                            value_array = np.array(value)
                        else:
                            value_array = value
                            
                        value_shape = value_array.shape
                        
                        # 检查形状是否兼容
                        if var_shape != list(value_shape):
                            print(f"警告: 变量 {variable.name} 的形状不匹配! 需要: {var_shape}, 提供: {value_shape}")
                            
                            # 如果是标量，尝试扩展为需要的形状
                            if value_shape == () or (len(value_shape) == 1 and value_shape[0] == 1):
                                print(f"尝试将标量扩展为所需形状...")
                                value_array = np.full(var_shape, value_array.item(0) if hasattr(value_array, 'item') else value_array)
                                print(f"扩展后形状: {value_array.shape}")
                            # 如果只是尺寸不同，尝试调整大小
                            elif len(var_shape) == len(value_shape):
                                # 尝试调整大小，裁剪或填充
                                new_value = np.zeros(var_shape, dtype=value_array.dtype)
                                
                                # 对每个维度，取两个形状的最小值
                                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(var_shape, value_shape))
                                
                                # 将原始数据复制到新数组中
                                new_value[slices] = value_array[slices]
                                value_array = new_value
                                print(f"调整大小后形状: {value_array.shape}")
                            else:
                                # 如果形状完全不兼容，则跳过此变量
                                print(f"无法调整大小，跳过此变量")
                                continue
                            
                        # 执行加载
                        try:
                            self.sess.run(variable.assign(value_array))
                        except Exception as e:
                            print(f"使用 assign 设置变量 {variable.name} 失败: {e}")
                            try:
                                # 备用方法：使用 initializer
                                variable.load(value_array, self.sess)
                            except Exception as e2:
                                print(f"使用 load 设置变量 {variable.name} 也失败: {e2}")
                    except Exception as e:
                        print(f"处理变量 {variable.name} 时出错: {e}")

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
