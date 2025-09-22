import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

class PLSRMultiOutputModel:
    """
    多目标输出的PLSR模型类
    """
    def __init__(self, n_components=10, max_iter=500):
        self.n_components = n_components
        self.max_iter = max_iter
        self.models = {}  # 存储为每个目标训练的PLSR模型
        self.scalers_X = {}  # 特征缩放器
        self.scalers_y = {}  # 目标变量缩放器
        
    def fit(self, X, y, target_names):
        """
        训练模型
        X: 输入特征
        y: 目标变量，形状为 [n_samples, n_targets]
        target_names: 目标变量名称列表
        """
        for i, target_name in enumerate(target_names):
            # 为每个目标变量创建一个PLSR模型
            self.models[target_name] = PLSRegression(n_components=self.n_components, max_iter=self.max_iter)
            
            # 创建并应用缩放器
            self.scalers_X[target_name] = StandardScaler()
            self.scalers_y[target_name] = StandardScaler()
            
            X_scaled = self.scalers_X[target_name].fit_transform(X)
            y_i = y[:, i].reshape(-1, 1)  # 获取当前目标变量并重塑为列向量
            y_scaled = self.scalers_y[target_name].fit_transform(y_i)
            
            # 训练模型
            self.models[target_name].fit(X_scaled, y_scaled)
            
        return self
    
    def predict(self, X, target_names):
        """
        预测
        X: 输入特征
        target_names: 目标变量名称列表
        返回: 预测结果数组，形状为 [n_samples, n_targets]
        """
        predictions = np.zeros((X.shape[0], len(target_names)))
        
        for i, target_name in enumerate(target_names):
            X_scaled = self.scalers_X[target_name].transform(X)
            y_pred_scaled = self.models[target_name].predict(X_scaled)
            predictions[:, i] = self.scalers_y[target_name].inverse_transform(y_pred_scaled).flatten()
            
        return predictions
    
    def evaluate(self, X, y, target_names):
        """
        评估模型性能
        X: 输入特征
        y: 真实目标值
        target_names: 目标变量名称列表
        返回: 评估结果字典
        """
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],  # 添加RPD指标
            'predictions': y_pred,
            'targets': y
        }
        
        for i, target_name in enumerate(target_names):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            
            # 计算RPD (Ratio of Performance to Deviation)
            std_dev = np.std(y_true)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results
    
    def get_feature_importance(self, target_names):
        """
        获取特征重要性
        target_names: 目标变量名称列表
        返回: 特征重要性字典
        """
        feature_importance = {}
        
        for target_name in target_names:
            # 对于PLSR，使用VIP (Variable Importance in Projection) 或系数作为特征重要性
            # 这里使用模型系数的绝对值作为简化的特征重要性度量
            model = self.models[target_name]
            importance = np.abs(model.coef_).flatten()
            feature_importance[target_name] = importance
            
        return feature_importance

class SVRMultiOutputModel:
    """
    多目标输出的SVR模型类
    """
    def __init__(self, kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.models = {}  # 存储为每个目标训练的SVR模型
        self.scaler_X = StandardScaler()  # 全局特征缩放器
        self.scalers_y = {}  # 每个目标变量的缩放器
        
    def fit(self, X, y, target_names):
        """
        训练模型
        X: 输入特征
        y: 目标变量，形状为 [n_samples, n_targets]
        target_names: 目标变量名称列表
        """
        # 标准化特征
        X_scaled = self.scaler_X.fit_transform(X)
        
        for i, target_name in enumerate(target_names):
            # 获取当前目标变量
            y_i = y[:, i].reshape(-1, 1)
            
            # 创建并应用目标变量缩放器
            self.scalers_y[target_name] = StandardScaler()
            y_scaled = self.scalers_y[target_name].fit_transform(y_i)
            
            # 创建并训练SVR模型
            svr = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma
            )
            
            # 训练模型
            svr.fit(X_scaled, y_scaled.ravel())
            
            # 存储模型
            self.models[target_name] = svr
            
        return self
    
    def predict(self, X, target_names=None):
        """
        预测
        X: 输入特征
        target_names: 目标变量名称列表，如果为None则使用所有模型
        返回: 预测结果数组，形状为 [n_samples, n_targets]
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        predictions = np.zeros((X.shape[0], len(target_names)))
        
        # 标准化特征
        X_scaled = self.scaler_X.transform(X)
        
        for i, target_name in enumerate(target_names):
            # 预测
            y_pred_scaled = self.models[target_name].predict(X_scaled).reshape(-1, 1)
            
            # 反标准化预测结果
            predictions[:, i] = self.scalers_y[target_name].inverse_transform(y_pred_scaled).flatten()
            
        return predictions
    
    def evaluate(self, X, y, target_names=None):
        """
        评估模型性能
        X: 输入特征
        y: 真实目标值
        target_names: 目标变量名称列表
        返回: 评估结果字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],
            'predictions': y_pred,
            'targets': y
        }
        
        for i, target_name in enumerate(target_names):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            
            # 计算RPD
            std_dev = np.std(y_true)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results

class RFMultiOutputModel:
    """
    多目标输出的随机森林模型类
    """
    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.models = {}  # 存储为每个目标训练的随机森林模型
        self.feature_importances = {}  # 存储特征重要性
        
    def fit(self, X, y, target_names):
        """
        训练模型
        X: 输入特征
        y: 目标变量，形状为 [n_samples, n_targets]
        target_names: 目标变量名称列表
        """
        for i, target_name in enumerate(target_names):
            # 获取当前目标变量
            y_i = y[:, i]
            
            # 创建并训练随机森林模型
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # 训练模型
            rf.fit(X, y_i)
            
            # 存储模型
            self.models[target_name] = rf
            
            # 存储特征重要性
            self.feature_importances[target_name] = rf.feature_importances_
            
        return self
    
    def predict(self, X, target_names=None):
        """
        预测
        X: 输入特征
        target_names: 目标变量名称列表，如果为None则使用所有模型
        返回: 预测结果数组，形状为 [n_samples, n_targets]
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        predictions = np.zeros((X.shape[0], len(target_names)))
        
        for i, target_name in enumerate(target_names):
            # 预测
            predictions[:, i] = self.models[target_name].predict(X)
            
        return predictions
    
    def evaluate(self, X, y, target_names=None):
        """
        评估模型性能
        X: 输入特征
        y: 真实目标值
        target_names: 目标变量名称列表
        返回: 评估结果字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],
            'predictions': y_pred,
            'targets': y
        }
        
        for i, target_name in enumerate(target_names):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            
            # 计算RPD
            std_dev = np.std(y_true)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results
    
    def get_feature_importance(self, target_names=None):
        """
        获取特征重要性
        target_names: 目标变量名称列表
        返回: 特征重要性字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        return {target: self.feature_importances[target] for target in target_names}

class XGBMultiOutputModel:
    """
    多目标输出的XGBoost模型类
    """
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.models = {}  # 存储为每个目标训练的XGBoost模型
        self.feature_importances = {}  # 存储特征重要性
        
    def fit(self, X, y, target_names, X_val=None, y_val=None):
        """
        训练模型
        X: 输入特征
        y: 目标变量，形状为 [n_samples, n_targets]
        target_names: 目标变量名称列表
        X_val: 验证集特征 (可选)
        y_val: 验证集目标变量 (可选)
        """
        for i, target_name in enumerate(target_names):
            # 获取当前目标变量
            y_i = y[:, i]
            
            # 创建DMatrix对象
            dtrain = xgb.DMatrix(X, label=y_i)
            
            # 如果提供了验证集，则创建验证集DMatrix
            evals = []
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val[:, i])
                evals = [(dtrain, 'train'), (dval, 'validation')]
            
            # XGBoost参数
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': self.max_depth,
                'eta': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'seed': self.random_state
            }
            
            # 训练模型
            model_xgb = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=20 if evals else None,
                verbose_eval=False
            )
            
            # 存储模型
            self.models[target_name] = model_xgb
            
            # 存储特征重要性
            self.feature_importances[target_name] = model_xgb.get_score(importance_type='gain')
            
        return self
    
    def predict(self, X, target_names=None):
        """
        预测
        X: 输入特征
        target_names: 目标变量名称列表，如果为None则使用所有模型
        返回: 预测结果数组，形状为 [n_samples, n_targets]
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        predictions = np.zeros((X.shape[0], len(target_names)))
        
        for i, target_name in enumerate(target_names):
            # 创建DMatrix
            dtest = xgb.DMatrix(X)
            
            # 预测
            predictions[:, i] = self.models[target_name].predict(dtest)
            
        return predictions
    
    def evaluate(self, X, y, target_names=None):
        """
        评估模型性能
        X: 输入特征
        y: 真实目标值
        target_names: 目标变量名称列表
        返回: 评估结果字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],
            'predictions': y_pred,
            'targets': y
        }
        
        for i, target_name in enumerate(target_names):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            
            # 计算RPD
            std_dev = np.std(y_true)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results
    
    def get_feature_importance(self, target_names=None):
        """
        获取特征重要性
        target_names: 目标变量名称列表
        返回: 特征重要性字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        return {target: self.feature_importances[target] for target in target_names}

class MLPMultiOutputModel:
    """
    多目标输出的多层感知机模型类
    """
    def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, 
                 batch_size='auto', learning_rate='adaptive', max_iter=500, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = {}  # 存储为每个目标训练的MLP模型
        self.scaler_X = StandardScaler()  # 全局特征缩放器
        self.scalers_y = {}  # 每个目标变量的缩放器
        
    def fit(self, X, y, target_names):
        """
        训练模型
        X: 输入特征
        y: 目标变量，形状为 [n_samples, n_targets]
        target_names: 目标变量名称列表
        """
        # 标准化特征
        X_scaled = self.scaler_X.fit_transform(X)
        
        for i, target_name in enumerate(target_names):
            # 获取当前目标变量
            y_i = y[:, i].reshape(-1, 1)
            
            # 创建并应用目标变量缩放器
            self.scalers_y[target_name] = StandardScaler()
            y_scaled = self.scalers_y[target_name].fit_transform(y_i)
            
            # 创建并训练MLP模型
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            
            # 训练模型
            mlp.fit(X_scaled, y_scaled.ravel())
            
            # 存储模型
            self.models[target_name] = mlp
            
        return self
    
    def predict(self, X, target_names=None):
        """
        预测
        X: 输入特征
        target_names: 目标变量名称列表，如果为None则使用所有模型
        返回: 预测结果数组，形状为 [n_samples, n_targets]
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        predictions = np.zeros((X.shape[0], len(target_names)))
        
        # 标准化特征
        X_scaled = self.scaler_X.transform(X)
        
        for i, target_name in enumerate(target_names):
            # 预测
            y_pred_scaled = self.models[target_name].predict(X_scaled).reshape(-1, 1)
            
            # 反标准化预测结果
            predictions[:, i] = self.scalers_y[target_name].inverse_transform(y_pred_scaled).flatten()
            
        return predictions
    
    def evaluate(self, X, y, target_names=None):
        """
        评估模型性能
        X: 输入特征
        y: 真实目标值
        target_names: 目标变量名称列表
        返回: 评估结果字典
        """
        if target_names is None:
            target_names = list(self.models.keys())
            
        y_pred = self.predict(X, target_names)
        
        results = {
            'r2_scores': [],
            'rmse_values': [],
            'mae_values': [],
            'rpd_values': [],
            'predictions': y_pred,
            'targets': y
        }
        
        for i, target_name in enumerate(target_names):
            y_true = y[:, i]
            y_predicted = y_pred[:, i]
            
            # 计算评估指标
            r2 = r2_score(y_true, y_predicted)
            rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
            mae = mean_absolute_error(y_true, y_predicted)
            
            # 计算RPD
            std_dev = np.std(y_true)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            results['r2_scores'].append(r2)
            results['rmse_values'].append(rmse)
            results['mae_values'].append(mae)
            results['rpd_values'].append(rpd)
        
        return results

def find_optimal_components(X_train, y_train, target_names, max_components=30, cv=5, logger=None):
    """
    使用交叉验证找到最佳的PLSR组件数
    """
    from sklearn.model_selection import GridSearchCV
    
    if logger:
        logger.info("开始寻找最佳PLSR组件数...")
    
    # 为每个目标变量找到最佳组件数
    best_components = {}
    
    for i, target_name in enumerate(target_names):
        y_train_i = y_train[:, i].reshape(-1, 1)  # 当前目标变量
        
        # 使用网格搜索找到最佳组件数
        param_grid = {'n_components': list(range(1, min(max_components + 1, X_train.shape[1], X_train.shape[0])))}
        
        # 创建并拟合网格搜索
        plsr = PLSRegression()
        grid_search = GridSearchCV(plsr, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train_i)
        
        # 获取最佳组件数
        best_n_components = grid_search.best_params_['n_components']
        best_components[target_name] = best_n_components
        
        if logger:
            logger.info(f"目标变量 {target_name} 的最佳组件数: {best_n_components}")
    
    # 计算平均最佳组件数
    avg_best_components = int(np.mean(list(best_components.values())))
    
    if logger:
        logger.info(f"所有目标变量的平均最佳组件数: {avg_best_components}")
        logger.info(f"将使用 {avg_best_components} 作为PLSR模型的组件数")
    
    return avg_best_components, best_components
