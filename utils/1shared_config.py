"""
共享配置模块，确保训练和评估使用相同的设置
"""
import os
import json

def get_common_config():
    """获取通用配置参数"""
    return {
        'data_path': 'data/processed/subsets/dataset_max10_per_class.csv',
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 16,
        'num_epochs': 150,
        'patience': 30,
    }

def load_model_config(model_dir):
    """加载模型配置"""
    config_path = os.path.join(model_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_model_config(config, model_dir):
    """保存模型配置"""
    config_path = os.path.join(model_dir, 'model_config.json')
    os.makedirs(model_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
