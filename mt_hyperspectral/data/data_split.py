import numpy as np

def random_train_test_split(data, test_size=0.2, seed=42):
    """简单随机划分训练集和测试集"""
    np.random.seed(seed)
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_size = int(test_size * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return train_indices, test_indices

def stratified_train_test_split(data, target_cols, test_size=0.2, seed=42):
    """分层抽样，确保训练集和测试集具有相似的标签分布"""
    from sklearn.model_selection import StratifiedKFold
    
    # 对多目标数据进行分层，首先需要将连续变量离散化
    targets = data[target_cols]
    y_discrete = np.zeros(len(data))
    for i, col in enumerate(target_cols):
        # 将每个目标变量按照四分位数分成3类
        quartiles = np.percentile(targets[col], [33.33, 66.66])
        y_discrete_i = np.zeros(len(data))
        y_discrete_i[targets[col] <= quartiles[0]] = 0
        y_discrete_i[(targets[col] > quartiles[0]) & (targets[col] <= quartiles[1])] = 1
        y_discrete_i[targets[col] > quartiles[1]] = 2
        # 将离散化后的类别添加到总的类别标签中（使用不同的位权重）
        y_discrete += y_discrete_i * (3**i)
    
    # 使用分层K折交叉验证进行划分
    skf = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(data, y_discrete))
    
    return train_idx, val_idx

def random_train_val_test_split(data, val_size=0.1, test_size=0.1, seed=42):
    """随机划分训练集、验证集和测试集"""
    np.random.seed(seed)
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_count = int(test_size * n_samples)
    val_count = int(val_size * n_samples)
    
    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]
    
    return train_indices, val_indices, test_indices

# def stratified_train_val_test_split(data, target_cols, val_size=0.1, test_size=0.1, seed=42):
#     """分层抽样，将数据划分为训练集、验证集和测试集"""
#     from sklearn.model_selection import StratifiedShuffleSplit
    
#     # 对多目标数据进行分层，首先需要将连续变量离散化
#     targets = data[target_cols]
#     y_discrete = np.zeros(len(data))
#     for i, col in enumerate(target_cols):
#         # 将每个目标变量按照四分位数分成3类
#         quartiles = np.percentile(targets[col], [33.33, 66.66])
#         y_discrete_i = np.zeros(len(data))
#         y_discrete_i[targets[col] <= quartiles[0]] = 0
#         y_discrete_i[(targets[col] > quartiles[0]) & (targets[col] <= quartiles[1])] = 1
#         y_discrete_i[targets[col] > quartiles[1]] = 2
#         # 将离散化后的类别添加到总的类别标签中（使用不同的位权重）
#         y_discrete += y_discrete_i * (3**i)
    
#     # 首先分离出测试集
#     test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
#     train_val_idx, test_idx = next(test_splitter.split(data, y_discrete))
    
#     # 然后从剩余数据中分离出验证集
#     y_discrete_remaining = y_discrete[train_val_idx]
#     val_size_adjusted = val_size / (1 - test_size)  # 调整验证集的比例
#     val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=seed)
#     train_idx_temp, val_idx_temp = next(val_splitter.split(data.iloc[train_val_idx], y_discrete_remaining))
    
#     # 映射回原始索引
#     train_idx = train_val_idx[train_idx_temp]
#     val_idx = train_val_idx[val_idx_temp]
    
#     return train_idx, val_idx, test_idx

def stratified_train_val_test_split(data, target_cols, val_size=0.1, test_size=0.1, seed=42):
    """分层抽样，将数据划分为训练集、验证集和测试集"""
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 对多目标数据进行分层，首先需要将连续变量离散化
    targets = data[target_cols]
    y_discrete = np.zeros(len(data))
    for i, col in enumerate(target_cols):
        # 将每个目标变量按照四分位数分成3类
        quartiles = np.percentile(targets[col], [33.33, 66.66])
        y_discrete_i = np.zeros(len(data))
        y_discrete_i[targets[col] <= quartiles[0]] = 0
        y_discrete_i[(targets[col] > quartiles[0]) & (targets[col] <= quartiles[1])] = 1
        y_discrete_i[targets[col] > quartiles[1]] = 2
        # 将离散化后的类别添加到总的类别标签中（使用不同的位权重）
        y_discrete += y_discrete_i * (3**i)
    
    # 首先分离出测试集
    test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(test_splitter.split(data, y_discrete))
    
    # 然后从剩余数据中分离出验证集
    y_discrete_remaining = y_discrete[train_val_idx]
    val_size_adjusted = val_size / (1 - test_size)  # 调整验证集的比例
    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=seed)
    train_idx_temp, val_idx_temp = next(val_splitter.split(data.iloc[train_val_idx], y_discrete_remaining))
    
    # 映射回原始索引
    train_idx = train_val_idx[train_idx_temp]
    val_idx = train_val_idx[val_idx_temp]
    
    return train_idx, val_idx, test_idx