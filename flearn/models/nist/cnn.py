import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        # 输入维度：28x28 = 784
        self.num_classes = num_classes
        self.num_samples = 0  # 添加num_samples属性
        
        # 创建TF会话和计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 使用兼容API
            tf.compat.v1.set_random_seed(seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        
        # 初始化变量
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
        # 获取模型大小（用于通信成本计算）
        self.size = self.model_bytes()
        
    def create_model(self, optimizer):
        """创建CNN模型，使用TensorFlow核心操作而非高级API"""
        # 输入占位符
        features = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name='labels')
        
        # 将平铺的特征重塑为2D图像
        x = tf.reshape(features, [-1, 28, 28, 1])
        
        # 第一个卷积层 - 使用核心TF操作
        # 创建卷积权重和偏置
        W_conv1 = tf.compat.v1.get_variable('W_conv1', shape=[5, 5, 1, 32], 
                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.compat.v1.get_variable('b_conv1', shape=[32], 
                                 initializer=tf.compat.v1.constant_initializer(0.1))
        # 应用卷积
        conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        # 添加偏置和激活函数
        h_conv1 = tf.nn.relu(conv1 + b_conv1)
        # 第一个池化层
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # 第二个卷积层 - 使用核心TF操作
        W_conv2 = tf.compat.v1.get_variable('W_conv2', shape=[5, 5, 32, 64], 
                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.compat.v1.get_variable('b_conv2', shape=[64], 
                                 initializer=tf.compat.v1.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # 展平层
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        
        # 全连接层 - 使用核心TF操作
        W_fc1 = tf.compat.v1.get_variable('W_fc1', shape=[7 * 7 * 64, 512], 
                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b_fc1 = tf.compat.v1.get_variable('b_fc1', shape=[512], 
                               initializer=tf.compat.v1.constant_initializer(0.1))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # Dropout层
        keep_prob = tf.compat.v1.placeholder_with_default(
            1.0, shape=[], name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)
        
        # 输出层
        W_fc2 = tf.compat.v1.get_variable('W_fc2', shape=[512, self.num_classes], 
                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b_fc2 = tf.compat.v1.get_variable('b_fc2', shape=[self.num_classes], 
                               initializer=tf.compat.v1.constant_initializer(0.1))
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        # 计算损失
        loss = tf.reduce_mean(
            tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
        )
        
        # L2正则化
        regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                        tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
                        tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                        tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
        loss = loss + 0.001 * regularizers
        
        # 预测和准确率
        predictions = tf.argmax(logits, 1)
        correct_predictions = tf.equal(predictions, labels)
        
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
        """设置模型参数"""
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        """获取模型参数"""
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len=None):
        """计算给定数据的梯度"""
        with self.graph.as_default():
            # 保存样本数量
            self.num_samples = len(data['y'])
            
            # 计算梯度
            grads = self.sess.run(self.grads, 
                feed_dict={self.features: data['x'], self.labels: data['y']})
            
            # 展平梯度
            if model_len is not None:
                from flearn.utils.tf_utils import process_sparse_grad
                grads = process_sparse_grad(grads)
            
            return self.num_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        """本地训练模型"""
        # 保存样本数量
        self.num_samples = len(data['y'])
        
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)
        
        # 计算通信成本
        bytes_w = self.size
        flops = self.compute_flops(len(data['y']), batch_size, num_epochs)
        bytes_r = self.size
        
        return self.get_params(), (bytes_w, flops, bytes_r)

    def run_epoch(self, data, batch_size):
        """运行一个训练周期"""
        batches = self.batch_data(data, batch_size)
        for b in batches:
            with self.graph.as_default():
                self.sess.run(self.train_op, 
                    feed_dict={self.features: b['x'], self.labels: b['y']})

    def test(self, data):
        """在给定数据上测试模型"""
        with self.graph.as_default():
            metrics = self.sess.run(self.eval_metric_ops, 
                feed_dict={self.features: data['x'], self.labels: data['y']})
            
        return metrics['accuracy'], self.sess.run(self.loss, 
            feed_dict={self.features: data['x'], self.labels: data['y']}), len(data['y'])

    def close(self):
        """关闭会话"""
        self.sess.close()

    def model_bytes(self):
        """估计模型大小"""
        with self.graph.as_default():
            params = self.get_params()
            size = 0
            for param in params:
                size += param.size * 4  # 假设每个参数是4字节浮点数
        return size

    def compute_flops(self, num_samples, batch_size, num_epochs):
        """估计FLOPs（浮点运算次数）"""
        # CNN模型的FLOPs估计
        ops_per_sample = 12281344  # 这是一个粗略估计
        total_ops = ops_per_sample * num_samples * num_epochs
        return total_ops

    def batch_data(self, data, batch_size):
        """将数据分成多个批次"""
        data_x = data['x']
        data_y = data['y']
        
        # 数据样本数
        num_samples = len(data_y)
        
        # 批次数
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # 批次列表
        batches = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, num_samples)
            
            batches.append({
                'x': data_x[start:end],
                'y': data_y[start:end]
            })
        
        return batches 