@echo off
echo 创建增强版MNIST数据集...
python utils\create_enhanced_dataset.py

echo 运行原始FedProx和增强版FedProx实验...
python compare_fedprox.py --run --dataset enhanced_mnist

echo 实验完成,查看results/comparison_enhanced_mnist目录下的结果