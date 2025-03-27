#!/usr/bin/env bash
echo "创建增强版MNIST数据集..."
python utils/create_enhanced_dataset.py

echo "运行原始FedProx和增强版FedProx实验..."
# 使用main_enhanced.py而不是main.py
python main_enhanced.py --dataset=enhanced_mnist --optimizer=fedprox \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model=mclr \
            --drop_percent=0.0 \
            --mu=0.01 \
            --use_enhanced=true \
            --similarity_threshold=0.5 \
            --reference_data_size=100

echo "实验完成,查看results/fedprox_enhanced_mnist_enhanced目录下的结果"