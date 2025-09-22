import os
import torch
import json
import pandas as pd

def save_model(model, path):
    """保存模型到指定路径"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return path

def load_model(model, path, device='cpu'):
    """从指定路径加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def save_results(results, path):
    """保存评估结果到CSV文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 如果结果是字典列表，转换为DataFrame
    if isinstance(results, list) and isinstance(results[0], dict):
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
    # 如果结果已经是DataFrame，直接保存
    elif isinstance(results, pd.DataFrame):
        results.to_csv(path, index=False)
    else:
        raise ValueError("结果格式不支持，请提供字典列表或DataFrame")
    
    print(f"结果已保存至 {path}")

def save_training_config(config, path):
    """保存训练配置到JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 确保所有值都是可JSON序列化的
    for k, v in config.items():
        if not isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config[k] = str(v)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    print(f"训练配置已保存至 {path}")
