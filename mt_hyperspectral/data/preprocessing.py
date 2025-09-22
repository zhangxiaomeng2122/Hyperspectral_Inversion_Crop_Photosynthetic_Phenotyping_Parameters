import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca_preprocessing(X_train, X_val, X_test, n_components=50):
    """应用PCA预处理到训练集、验证集和测试集
    
    参数:
        X_train: 训练数据
        X_val: 验证数据
        X_test: 测试数据
        n_components: PCA组件数量
        
    返回:
        X_train_pca: PCA处理后的训练数据
        X_val_pca: PCA处理后的验证数据
        X_test_pca: PCA处理后的测试数据
        pca: 训练好的PCA模型
    """
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 应用PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"应用PCA预处理: 输入特征维度 {X_train.shape[1]} -> {X_train_pca.shape[1]}")
    
    return X_train_pca, X_val_pca, X_test_pca, pca
