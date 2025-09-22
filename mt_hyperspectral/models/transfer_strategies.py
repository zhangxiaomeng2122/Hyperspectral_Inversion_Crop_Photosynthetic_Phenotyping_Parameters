import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

class TaskRelationshipTransfer(nn.Module):
    """基于任务关系的迁移学习策略
    
    通过建模任务之间的关系来优化知识迁移过程。
    该方法适用于源任务和目标任务存在关联但不完全相同的场景。
    """
    def __init__(self, base_model, source_tasks, target_tasks, 
                 task_similarity_matrix=None, feature_dim=512, dropout=0.2):
        super().__init__()
        self.base_model = base_model  # 预训练的基础模型
        self.source_tasks = source_tasks
        self.target_tasks = target_tasks
        
        # 如果没有提供任务相似度矩阵，创建一个默认的
        self.task_similarity = task_similarity_matrix or self._create_default_similarity()
        
        # 记录任务映射关系
        self.task_mapping = {}
        for target_task in target_tasks:
            if target_task in source_tasks:
                # 如果目标任务在源任务中，直接映射
                self.task_mapping[target_task] = target_task
            else:
                # 否则找到最相似的源任务
                most_similar = self._find_most_similar_source_task(target_task)
                self.task_mapping[target_task] = most_similar
        
        # 修复task_adapter创建和使用的问题
        self.task_adapters = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, 1)
            ) for i, task in enumerate(source_tasks) if task not in target_tasks
        })
        
        # 特征增强 - 波段注意力机制
        self.spectral_attention = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.GELU(),
                nn.Linear(128, self.base_model.input_proj.in_features),  # 匹配原始输入特征数
                nn.Sigmoid()
            ) for task in target_tasks if task in ['LAW', 'Cx']  # 特别关注的参数
        })
    
    def _create_default_similarity(self):
        """创建默认的任务相似度矩阵，基于领域知识"""
        # 根据植物生理学知识定义任务组
        photosynthesis_tasks = ['Pn', 'Pmax', 'Gs', 'Ci', 'Tr', 'RUE']
        chlorophyll_tasks = ['SPAD', 'Chl-a', 'Chl-b', 'Chl']
        structure_tasks = ['LAI', 'LAW', 'LMA']
        nutrition_tasks = ['LNC', 'LPC', 'LKC']
        pigment_tasks = ['Car', 'Cx']  # Cx是胡萝卜素类参数
        water_tasks = ['WUE', 'RWC', 'LAW']  # LAW与结构和水分都相关
        
        # 定义任务组之间的相关性
        group_similarity = {
            'photosynthesis': {'chlorophyll': 0.7, 'structure': 0.4, 'nutrition': 0.5, 'pigment': 0.5, 'water': 0.6},
            'chlorophyll': {'photosynthesis': 0.7, 'structure': 0.4, 'nutrition': 0.6, 'pigment': 0.8, 'water': 0.3},  # 增强叶绿素和色素的相关性
            'structure': {'photosynthesis': 0.4, 'chlorophyll': 0.4, 'nutrition': 0.3, 'pigment': 0.2, 'water': 0.7},
            'nutrition': {'photosynthesis': 0.5, 'chlorophyll': 0.6, 'structure': 0.3, 'pigment': 0.5, 'water': 0.3},
            'pigment': {'photosynthesis': 0.5, 'chlorophyll': 0.8, 'structure': 0.2, 'nutrition': 0.5, 'water': 0.3},  # 增强色素和叶绿素的相关性
            'water': {'photosynthesis': 0.6, 'chlorophyll': 0.3, 'structure': 0.7, 'nutrition': 0.3, 'pigment': 0.3}
        }
        
        # 所有任务列表
        all_tasks = list(set(self.source_tasks + self.target_tasks))
        
        # 任务所属组映射
        task_to_group = {}
        for task in all_tasks:
            if task in photosynthesis_tasks:
                task_to_group[task] = 'photosynthesis'
            elif task in chlorophyll_tasks:
                task_to_group[task] = 'chlorophyll'
            elif task in structure_tasks:
                task_to_group[task] = 'structure'
            elif task in nutrition_tasks:
                task_to_group[task] = 'nutrition'
            elif task in pigment_tasks:
                task_to_group[task] = 'pigment'
            elif task in water_tasks:
                task_to_group[task] = 'water'
            else:
                # 默认分配到最近的组
                task_to_group[task] = 'chlorophyll'  # 默认相似度
        
        # 创建任务相似度矩阵
        similarity = {}
        for t1 in all_tasks:
            similarity[t1] = {}
            for t2 in all_tasks:
                # 同一任务相似度为1
                if t1 == t2:
                    similarity[t1][t2] = 1.0
                # Cx和SPAD特殊处理，增加它们之间的相似度
                elif (t1 == 'Cx' and t2 == 'SPAD') or (t1 == 'SPAD' and t2 == 'Cx'):
                    similarity[t1][t2] = 0.85  # 设置更高的相似度
                # 同一组内任务相似度较高
                elif task_to_group[t1] == task_to_group[t2]:
                    similarity[t1][t2] = 0.8
                # 不同组之间基于预定义的组间相似度
                else:
                    try:
                        group_sim = group_similarity[task_to_group[t1]][task_to_group[t2]]
                        similarity[t1][t2] = group_sim
                    except KeyError:
                        similarity[t1][t2] = 0.3  # 默认相似度
        
        return similarity
    
    def _find_most_similar_source_task(self, target_task):
        """找到与目标任务最相似的源任务"""
        max_similarity = -1
        most_similar_task = self.source_tasks[0]
        
        for source_task in self.source_tasks:
            similarity = self.task_similarity.get(target_task, {}).get(source_task, 0)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_task = source_task
        
        return most_similar_task
    
    def forward(self, x):
        # 将原始特征直接传递给base_model
        base_outputs = self.base_model(x)
        
        # 计算编码器输出用于任务适配
        # 投影到隐藏维度
        x_projected = self.base_model.input_proj(x)
        # 扩展维度并添加位置编码
        x_transformed = x_projected.unsqueeze(1) + self.base_model.pos_embedding
        
        # 手动遍历编码器模块
        for encoder_layer in self.base_model.encoders:
            x_transformed = encoder_layer(x_transformed)
        
        # 提取表示 - 定义encoder_out变量
        encoder_out = x_transformed.squeeze(1)  # [batch_size, hidden_dim]
        
        # 生成任务特定输出
        task_outputs = []
        for i, source_task in enumerate(self.source_tasks):
            # 如果源任务在目标任务中，直接使用对应的base_model输出
            if source_task in self.target_tasks:
                idx = self.target_tasks.index(source_task)
                if idx < base_outputs.size(1):  # 确保索引有效
                    task_outputs.append(base_outputs[:, idx].unsqueeze(1))
            # 否则，通过任务适配器进行映射（如果适配器存在）
            else:
                task_id_str = str(i)  # 使用任务索引作为键
                if task_id_str in self.task_adapters:
                    adapter_module = self.task_adapters[task_id_str]
                    adapter_output = adapter_module(encoder_out)
                    task_outputs.append(adapter_output)
        
        # 组合任务输出（仅包含有效的输出）
        if task_outputs:
            final_output = torch.cat(task_outputs, dim=1)  # [batch_size, num_target_tasks]
        else:
            # 如果没有有效输出，返回基础模型输出
            final_output = base_outputs
            
        return final_output


class FeatureAttentionTransfer(nn.Module):
    """基于特征注意力的迁移学习策略
    
    通过学习光谱特征与目标参数之间的相关性来提高迁移性能。
    特别适用于处理高维光谱数据和多样的植物生理参数。
    """
    def __init__(self, base_model, target_tasks, input_dim=273, feature_dim=512, dropout=0.2):
        super().__init__()
        self.base_model = base_model
        self.target_tasks = target_tasks
        
        # 波段注意力模块
        self.spectral_attention = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            ) for task in target_tasks
        })
        
        # 任务特定的混合层
        self.task_mixers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim)
            ) for task in target_tasks
        })
        
        # 每个任务单独的预测头
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(feature_dim, 1) for task in target_tasks
        })
    
    def forward(self, x):
        # 计算编码器输出，避免重复处理
        x_projected = self.base_model.input_proj(x)
        x_transformed = x_projected.unsqueeze(1) + self.base_model.pos_embedding
        
        # 手动遍历编码器模块
        for encoder_layer in self.base_model.encoders:
            x_transformed = encoder_layer(x_transformed)
            
        # 提取表示 - 定义encoder_out变量
        encoder_out = x_transformed.squeeze(1)
        
        # 兼容不同模型结构：检查base_model是否有decoder属性或task_decoders属性
        if hasattr(self.base_model, 'decoder'):
            # 标准的MultiTaskViT模型结构
            base_outputs = self.base_model.decoder(encoder_out)
        elif hasattr(self.base_model, 'task_decoders'):
            # EnhancedMultiTaskViT模型结构
            task_outputs = []
            for decoder in self.base_model.task_decoders:
                task_out = decoder(encoder_out)
                task_outputs.append(task_out)
            base_outputs = torch.cat(task_outputs, dim=1)  # [batch_size, num_tasks]
        else:
            # 如果都没有找到，尝试直接使用前向传播
            try:
                base_outputs = self.base_model(x)
                # 剩余的光谱注意力逻辑仍然需要执行
            except Exception as e:
                raise AttributeError(f"不兼容的模型结构: {type(self.base_model).__name__} 没有decoder或task_decoders属性，且前向传播失败。错误: {e}")
        
        # 计算光谱注意力
        spectral_attn = self.spectral_attention(x)  # [batch_size, input_size]
        enhanced_features = x * spectral_attn  # 注意力加权特征
        
        # 处理增强的特征 - 避免重复计算
        enhanced_projected = self.base_model.input_proj(enhanced_features)
        enhanced_x = enhanced_projected.unsqueeze(1) + self.base_model.pos_embedding
        
        for encoder_layer in self.base_model.encoders:
            enhanced_x = encoder_layer(enhanced_x)
            
        enhanced_encoder_out = enhanced_x.squeeze(1)
        
        # 使用与上面相同的逻辑处理增强特征
        if hasattr(self.base_model, 'decoder'):
            enhanced_outputs = self.base_model.decoder(enhanced_encoder_out)
        elif hasattr(self.base_model, 'task_decoders'):
            enhanced_task_outputs = []
            for decoder in self.base_model.task_decoders:
                enhanced_task_out = decoder(enhanced_encoder_out)
                enhanced_task_outputs.append(enhanced_task_out)
            enhanced_outputs = torch.cat(enhanced_task_outputs, dim=1)
        else:
            # 如果已经尝试过前向传播且成功了
            enhanced_outputs = self.base_model(enhanced_features)
        
        # 合并基础输出和增强输出
        alpha = torch.sigmoid(self.fusion_alpha)
        final_outputs = alpha * base_outputs + (1 - alpha) * enhanced_outputs
        
        return final_outputs

    def get_transfer_specific_params(self):
        """获取仅属于迁移学习部分的参数（不包括基础模型参数）"""
        # 根据具体实现收集仅属于迁移层的参数
        transfer_params = []
        
        if hasattr(self, 'task_mixers'):
            for mixer in self.task_mixers.values():
                transfer_params += list(mixer.parameters())
        
        if hasattr(self, 'spectral_attention'):
            transfer_params += list(self.spectral_attention.parameters())
        
        if hasattr(self, 'task_heads'):
            for head in self.task_heads.values():
                transfer_params += list(head.parameters())
        
        # 添加其他迁移学习特定参数
        # ...
        
        return transfer_params

    def get_parameter_groups(self, base_lr):
        """获取不同学习率的参数组"""
        # 按名称分组参数
        base_model_params = []
        transfer_specific_params = []
        
        for name, param in self.named_parameters():
            if 'task_mixers' in name or 'spectral_attention' in name or 'task_heads' in name:
                # 迁移学习特定参数 - 使用更高学习率
                transfer_specific_params.append(param)
            else:
                # 基础模型参数 - 使用基本学习率
                base_model_params.append(param)
        
        param_groups = [
            {'params': base_model_params, 'lr': base_lr * 0.1},  # 基础模型参数使用较低学习率
            {'params': transfer_specific_params, 'lr': base_lr * 2.0}  # 迁移学习参数使用较高学习率
        ]
        
        return param_groups

def create_transfer_model(strategy, base_model, source_tasks, target_tasks, feature_dim=512, dropout=0.0):
    """基于不同策略创建迁移模型"""
    print(f"创建迁移模型: 策略={strategy}, 源任务数={len(source_tasks)}, 目标任务数={len(target_tasks)}")
    
    # 确保源任务列表和目标任务列表有效
    if not set(target_tasks).issubset(set(source_tasks + target_tasks)):
        print(f"警告: 目标任务集合 {target_tasks} 不是源任务集合 {source_tasks} 的子集")
        # 调整源任务列表，确保包含所有目标任务
        adjusted_source_tasks = list(set(source_tasks + target_tasks))
        print(f"已调整源任务列表: {adjusted_source_tasks}")
        source_tasks = adjusted_source_tasks
    
    if strategy == 'task_relationship':
        return TaskRelationshipTransfer(base_model, source_tasks, target_tasks, feature_dim=feature_dim, dropout=dropout)
    elif strategy == 'feature_attention':
        return FeatureAttentionTransfer(base_model, target_tasks, feature_dim=feature_dim, dropout=dropout)
    else:
        raise ValueError(f"不支持的迁移策略: {strategy}")

def get_source_task_mapping(source_tasks, target_tasks):
    """获取源任务到目标任务的映射关系"""
    mapping = {}
    for target in target_tasks:
        if target in source_tasks:
            mapping[target] = target  # 完全相同的任务直接映射
        else:
            # 建立任务相似性的启发式规则
            if target in ['Pn', 'Pmax', 'Gs', 'Ci', 'Tr']:
                # 光合参数相互关联
                candidates = [t for t in source_tasks if t in ['Pn', 'Pmax', 'Gs', 'Ci', 'Tr']]
                if candidates:
                    mapping[target] = candidates[0]
                    continue
            
            # 修改: Cx 优先从SPAD迁移
            if target == 'Cx':
                if 'SPAD' in source_tasks:
                    mapping[target] = 'SPAD'
                    continue
                elif 'Chl' in source_tasks or 'Chl-a' in source_tasks or 'Chl-b' in source_tasks:
                    # 如果没有SPAD，则从其他叶绿素参数迁移
                    chlorophyll_candidates = [t for t in source_tasks if t in ['Chl', 'Chl-a', 'Chl-b']]
                    if chlorophyll_candidates:
                        mapping[target] = chlorophyll_candidates[0]
                        continue
                        
            if target in ['SPAD', 'Chl', 'Chl-a', 'Chl-b']:
                # 叶绿素参数相互关联
                candidates = [t for t in source_tasks if t in ['SPAD', 'Chl', 'Chl-a', 'Chl-b']]
                if candidates:
                    mapping[target] = candidates[0]
                    continue
            
            if target in ['LAW', 'LAI', 'LMA']:
                # 叶片结构参数相互关联
                candidates = [t for t in source_tasks if t in ['LAW', 'LAI', 'LMA']]
                if candidates:
                    mapping[target] = candidates[0]
                    continue
                    
            if target in ['Cx', 'Car']:
                # 类胡萝卜素参数相互关联 - 此逻辑保留但已被上面的特定Cx映射规则覆盖
                candidates = [t for t in source_tasks if t in ['Cx', 'Car']]
                if candidates:
                    mapping[target] = candidates[0]
                    continue
            
            # 默认情况：选择第一个源任务
            mapping[target] = source_tasks[0]
    
    return mapping