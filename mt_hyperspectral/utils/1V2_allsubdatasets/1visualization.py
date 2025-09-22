import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import torch
from torch.utils.data import DataLoader
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #e:\pycharm_program\MT_hyperspectral_inversion
sys.path.append(root_dir)
from utils.plot_setting import setfig

def plot_predictions(true_values, pred_values, param_names, save_dir='results/simple_nn_allp'):
    """绘制散点图函数，对比预测值和真实值"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, param in enumerate(param_names):
        setfig(column=1, x=2.5, y=2.8)
        plt.scatter(true_values[:, i], pred_values[:, i], alpha=0.6)
        
        # 添加参考线
        min_val = min(true_values[:, i].min(), pred_values[:, i].min())
        max_val = max(true_values[:, i].max(), pred_values[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 计算并显示R²
        from sklearn.metrics import r2_score
        r2 = r2_score(true_values[:, i], pred_values[:, i])
        plt.title(f'{param} Prediction (R² = {r2:.3f})')
        plt.xlabel(f'True {param}')
        plt.ylabel(f'Predicted {param}')
        # plt.grid(True, alpha=0.3)
        
        # 保存图像
        # plt.savefig(f'{save_dir}/{param}_prediction.png', dpi=300)
        plt.savefig(f"{save_dir}/{param}.pdf", format='pdf', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

def visualize_feature_importance(results_dir, feature_names, pca=None):
    """可视化并分析特征重要性"""
    os.makedirs(results_dir, exist_ok=True)
    
    # 如果有PCA组件，分析组件对原始特征的影响
    if pca is not None and hasattr(pca, 'components_'):
        # 计算每个原始特征在PCA中的总体重要性
        feature_importance = np.abs(pca.components_).sum(axis=0)
        
        # 排序并找出最重要的特征
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = sorted_idx[:20]  # 获取前20个最重要的特征
        
        # 创建特征重要性条形图
        setfig(column=1, x=2.7, y=2.1)
        plt.bar(range(20), feature_importance[top_features])
        if feature_names is not None and len(feature_names) == len(feature_importance):
            # plt.xticks(range(20), [feature_names[i] for i in top_features], rotation=90, ha='right')
            plt.xticks(range(20), [feature_names[i] for i in top_features], rotation=45)
        else:
            plt.xticks(range(20), [f'Feature {i}' for i in top_features], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Top 20 Important Features')
        # plt.savefig(f'{results_dir}/feature_importance.png', dpi=300)
        plt.savefig(f'{results_dir}/feature_importance.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()
        
        # 保存最重要的特征
        if feature_names is not None:
            important_features = pd.DataFrame({
                'Feature': [feature_names[i] for i in sorted_idx],
                'Importance': feature_importance[sorted_idx]
            })
            important_features.to_csv(f'{results_dir}/important_features.csv', index=False)
    
    # 打印分析结果
    print("\n特征重要性分析已保存至", results_dir)

def plot_correlation_matrix(df, target_cols, save_path):
    """绘制目标变量之间的相关性矩阵"""
    corr_matrix = df[target_cols].corr()
    # plt.figure(figsize=(10, 8))
    setfig(column=1, x=4, y=3)
    # 修改annot_kws参数，设置字体大小为6pt
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 6})
    plt.title('Correlation Matrix of Target Variables')
    # plt.savefig(save_path, dpi=300)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

def create_prediction_plot_with_r2(model, target_idx, test_dataset, train_dataset=None, 
                                 target_name=None, device=None, save_dir=None, 
                                 test_r2=None, train_r2=None, format='PNG', plot_train=True):
    """创建预测结果的散点图，显示测试集和可选的训练集的R²分数
    
    参数:
        model: 训练好的模型
        target_idx: 目标变量的索引
        test_dataset: 测试数据集
        train_dataset: 训练数据集，可选
        target_name: 目标变量名称
        device: 计算设备
        save_dir: 保存图表的目录
        test_r2: 测试集的R²分数，如果已知
        train_r2: 训练集的R²分数，如果已知
        format: 保存图表的格式（PNG或PDF）
        plot_train: 是否绘制训练集数据
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # 处理测试集数据
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            target = y[:, target_idx].unsqueeze(1)
            output = model(X)
            test_preds.append(output.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    test_preds = np.vstack(test_preds).flatten()
    test_targets = np.vstack(test_targets).flatten()
    
    # 如果未提供R²分数，计算它 (使用标准化数据计算R²)
    if test_r2 is None:
        test_r2 = r2_score(test_targets, test_preds)
    
    # 反标准化测试集数据
    if hasattr(test_dataset, 'y_mean') and hasattr(test_dataset, 'y_std'):
        # 使用数据集中的均值和标准差进行反标准化
        test_preds_orig = test_preds * test_dataset.y_std[target_idx] + test_dataset.y_mean[target_idx]
        test_targets_orig = test_targets * test_dataset.y_std[target_idx] + test_dataset.y_mean[target_idx]
    else:
        # 如果数据集没有提供均值和标准差，使用原始数据
        print("警告: 未找到均值和标准差，使用标准化后的数据绘图")
        test_preds_orig = test_preds
        test_targets_orig = test_targets
    
    # 处理训练集数据（如果提供且plot_train=True）
    train_preds = None
    train_targets = None
    train_preds_orig = None
    train_targets_orig = None
    
    if train_dataset is not None and plot_train:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        train_preds = []
        train_targets = []
        
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                target = y[:, target_idx].unsqueeze(1)
                output = model(X)
                train_preds.append(output.cpu().numpy())
                train_targets.append(target.cpu().numpy())
        
        train_preds = np.vstack(train_preds).flatten()
        train_targets = np.vstack(train_targets).flatten()
        
        # 如果未提供R²分数，计算它 (使用标准化数据计算R²)
        if train_r2 is None and train_preds is not None:
            train_r2 = r2_score(train_targets, train_preds)
            
        # 反标准化训练集数据
        if hasattr(train_dataset, 'y_mean') and hasattr(train_dataset, 'y_std'):
            train_preds_orig = train_preds * train_dataset.y_std[target_idx] + train_dataset.y_mean[target_idx]
            train_targets_orig = train_targets * train_dataset.y_std[target_idx] + train_dataset.y_mean[target_idx]
        else:
            train_preds_orig = train_preds
            train_targets_orig = train_targets
    
    # 创建预测图 (使用反标准化后的数据)
    # plt.figure(figsize=(8, 6))
    setfig(column=1, x=2.5, y=2.8)
    
    # 绘制测试集点 (使用反标准化数据)
    plt.scatter(test_targets_orig, test_preds_orig, color='blue', alpha=0.7, s=8, marker='o', edgecolors='none', label=f'Test R² = {test_r2:.3f}')
    
    # 绘制训练集点 (使用反标准化数据, 如果可用且启用)
    if train_preds_orig is not None and plot_train:
        plt.scatter(train_targets_orig, train_preds_orig, color='green', alpha=0.3, s=8, marker='o', edgecolors='none', label=f'Train R² = {train_r2:.3f}')
    
    # 添加完美预测线 (使用反标准化数据的范围)
    min_val = min(np.min(test_targets_orig), np.min(test_preds_orig))
    max_val = max(np.max(test_targets_orig), np.max(test_preds_orig))
    if train_preds_orig is not None and plot_train:
        min_val = min(min_val, np.min(train_targets_orig), np.min(train_preds_orig))
        max_val = max(max_val, np.max(train_targets_orig), np.max(train_preds_orig))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加标签和标题 (标明使用了反标准化数据)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{target_name} Prediction Results' if target_name else 'Prediction Results')
    plt.legend(prop={'size':6, 'family': 'Arial'},frameon=False) 
    
    # 保存图表
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        save_name = f"{target_name}_predictions" if target_name else "predictions"
        if format.upper() == 'PDF':
            plt.savefig(f"{save_dir}/{save_name}.pdf", format='pdf', bbox_inches='tight', transparent=True, dpi=300)
        else:
            plt.savefig(f"{save_dir}/{save_name}.png", bbox_inches='tight', dpi=300)
    
    plt.close()

def plot_evaluation_summary(results, save_dir, format='PNG', include_train=True):
    """为所有目标变量绘制评估结果汇总
    
    参数:
        results: 评估结果列表，每个元素包含一个目标变量的评估指标
        save_dir: 保存图表的目录
        format: 保存图表的格式（PNG或PDF）
        include_train: 是否包含训练集结果
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 提取目标变量名称和测试集R²分数
    targets = [result['target'] for result in results]
    test_r2s = [result['test_r2'] for result in results]
    
    if include_train:
        train_r2s = [result['train_r2'] for result in results]
        val_r2s = [result['val_r2'] for result in results]
    
    # 将结果整理为DataFrame
    if include_train:
        data = {
            'Target': targets,
            'Train R²': train_r2s,
            'Validation R²': val_r2s,
            'Test R²': test_r2s
        }
    else:
        data = {
            'Target': targets,
            'Test R²': test_r2s
        }
    
    results_df = pd.DataFrame(data)
    
    # 按测试集R²值排序
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    # 创建竖直条形图
    setfig(column=1, x=5, y=2.8)
    
    # 创建X轴位置
    x = np.arange(len(targets))
    width = 0.25 if include_train else 0.6
    
    # 绘制竖直条形图
    if include_train:
        plt.bar(x - width, results_df['Train R²'], width, label='Training', color='green', alpha=0.7)
        plt.bar(x, results_df['Validation R²'], width, label='Validation', color='orange', alpha=0.7)
        plt.bar(x + width, results_df['Test R²'], width, label='Testing', color='blue', alpha=0.7)
    else:
        plt.bar(x, results_df['Test R²'], width, label='Testing', color='blue', alpha=0.8)
    
    # 添加标签和标题
    plt.xticks(x, results_df['Target'], rotation=45, ha='right')
    plt.ylabel('R² Score')
    plt.title('Evaluation Results for Each Target Variable')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False)
    
    # 为每个条形添加数值标签
    if include_train:
        for i, (train_r2, val_r2, test_r2) in enumerate(zip(results_df['Train R²'], results_df['Validation R²'], results_df['Test R²'])):
            plt.text(i - width, train_r2 + 0.01, f'{train_r2:.3f}', ha='center', va='bottom', fontsize=5, fontname='Arial', rotation=90)
            plt.text(i, val_r2 + 0.01, f'{val_r2:.3f}', ha='center', va='bottom', fontsize=5, fontname='Arial', rotation=90)
            plt.text(i + width, test_r2 + 0.01, f'{test_r2:.3f}', ha='center', va='bottom', fontsize=5, fontname='Arial', rotation=90)
    else:
        for i, test_r2 in enumerate(results_df['Test R²']):
            plt.text(i, test_r2 + 0.01, f'{test_r2:.3f}', ha='center', va='bottom', fontsize=5, fontname='Arial')
    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    
    if format.upper() == 'PDF':
        plt.savefig(f"{save_dir}/evaluation_summary.pdf", format='pdf', bbox_inches='tight', transparent=True, dpi=300)
    else:
        plt.savefig(f"{save_dir}/evaluation_summary.png", bbox_inches='tight', dpi=300)
    
    plt.close()
    
    # 保存汇总结果为CSV
    results_df.to_csv(f"{save_dir}/evaluation_summary.csv", index=False)
    print(f"评估汇总已保存至: {save_dir}/evaluation_summary.csv")

def create_prediction_plot_from_loader(model, idx, test_loader, target, device, output_dir, 
                                       test_r2, y_mean=None, y_std=None, format='PDF'):
    """
    使用DataLoader创建预测图，避免依赖dataset的内部属性，并进行反标准化处理。
    
    参数:
        model: 已训练的模型
        idx: 当目标变量有多个时，指定当前目标的索引
        test_loader: 测试数据的DataLoader
        target: 目标变量名称
        device: 计算设备 (CPU或GPU)
        output_dir: 输出目录
        test_r2: 测试集R²分数
        y_mean: 目标变量的均值，用于反标准化
        y_std: 目标变量的标准差，用于反标准化
        format: 输出图像格式
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.plot_setting import setfig
    
    # 创建图表目录
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 收集预测结果
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            try:
                # 将数据移至设备
                batch_data = batch_data.to(device)
                # 获取预测值
                outputs = model(batch_data)
                
                # 提取当前目标变量的真实值
                if batch_targets.shape[1] > 1:  # 多目标变量情况
                    batch_targets = batch_targets[:, idx].unsqueeze(1)
                    
                # 立即将预测转移到CPU，减少GPU内存占用
                batch_preds = outputs.cpu().numpy().flatten()
                batch_targets_np = batch_targets.cpu().numpy().flatten()
                
                # 添加到结果列表
                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets_np)
                
                # 显式清理GPU内存
                if device.type == 'cuda':
                    del outputs
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    print(f"CUDA错误处理: {str(e)}")
                    
                    # 如果CUDA错误，切换到CPU处理
                    model_cpu = model.cpu()
                    batch_outputs = model_cpu(batch_data.cpu())
                    model.to(device)  # 处理完后切回GPU
                    
                    # 提取目标变量
                    if batch_targets.shape[1] > 1:
                        batch_targets = batch_targets[:, idx].unsqueeze(1)
                        
                    # 添加到结果列表
                    all_preds.extend(batch_outputs.numpy().flatten())
                    all_targets.extend(batch_targets.numpy().flatten())
                else:
                    raise  # 重新抛出非CUDA错误
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 进行反标准化处理
    if y_mean is not None and y_std is not None:
        try:
            # 尝试作为数组访问
            if isinstance(y_mean, (np.ndarray, list)) and len(y_mean) > idx:
                y_mean_val = y_mean[idx]
                y_std_val = y_std[idx]
            else:
                # 是标量或者索引超出范围，直接使用
                y_mean_val = y_mean
                y_std_val = y_std
            
            # 反标准化
            print(f"为{target}图表进行反标准化处理: mean={y_mean_val}, std={y_std_val}")
            all_preds_orig = all_preds * y_std_val + y_mean_val
            all_targets_orig = all_targets * y_std_val + y_mean_val
            
            # 使用反标准化的数据
            plot_preds = all_preds_orig
            plot_targets = all_targets_orig
            units_label = "(原始单位)"
            is_denormalized = True
        except (TypeError, IndexError) as e:
            print(f"反标准化处理出错: {str(e)}，使用标准化数据绘图")
            plot_preds = all_preds
            plot_targets = all_targets
            units_label = "(标准化单位)"
            is_denormalized = False
    else:
        print("未提供标准化参数，使用原始数据绘图")
        plot_preds = all_preds
        plot_targets = all_targets
        units_label = ""
        is_denormalized = False
    
    # 绘制散点图
    setfig(column=1, x=2.5, y=2.8)
    plt.scatter(plot_targets, plot_preds, color='blue', alpha=0.7, s=8, marker='o', edgecolors='none',
               label=f"Test R² = {test_r2:.3f}")
    
    # 绘制完美预测线
    if len(plot_targets) > 0:
        min_val = min(np.min(plot_targets), np.min(plot_preds))
        max_val = max(np.max(plot_targets), np.max(plot_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"{target} Predictions Results")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend(prop={'size':6, 'family': 'Arial'}, frameon=False) 
    # plt.grid(True, alpha=0.3)
    
    # 使用明确且安全的文件路径
    suffix = "_orig" if is_denormalized else ""
    safe_filename = f"{target.replace('-', '_')}_predictions.pdf"
    plot_path = os.path.join(plot_dir, safe_filename)
    plt.savefig(plot_path, format=format.lower(), bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    
    return plot_path

def create_feature_comparison_plot(feature_comparison_df, output_dir, format='pdf'):
    """
    创建特征比较条形图，比较不同特征配置间的R²分数
    
    参数:
        feature_comparison_df: 包含比较数据的DataFrame
        output_dir: 输出目录路径
        format: 输出图表格式，默认为pdf
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.plot_setting import setfig
    
    # 只显示原始参数，排除'Average'行
    plot_df = feature_comparison_df[feature_comparison_df['Parameter'] != 'Average'].copy()
    
    # 为每个参数创建分组条形图
    fig = setfig(column=1, x=5, y=2.8)
    ax = plt.gca()  # 获取当前的坐标轴对象
    
    # 获取所有R2列
    r2_cols = [col for col in plot_df.columns if col.endswith('_R2')]
    
    # 排序，根据平均R2值
    plot_df['avg_r2'] = plot_df[r2_cols].mean(axis=1)
    plot_df = plot_df.sort_values('avg_r2', ascending=False)
    
    # 绘制分组条形图
    x = np.arange(len(plot_df))
    width = 0.8 / len(r2_cols)  # 分配宽度
    rects_list = []
    for i, col in enumerate(r2_cols):
        rects = ax.bar(x + (i - len(r2_cols)/2 + 0.5) * width, 
                      plot_df[col], width, 
                      label=col.replace('_R2', ''))
        rects_list.append(rects)
    
    # 添加标签和标题
    ax.set_ylabel('R² Score')
    ax.set_title('Comparison of R² Scores Across Different Feature Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Parameter'], rotation=45, ha='right')
    
    # 添加图例
    ax.legend(prop={'size':6, 'family': 'Arial'}, 
             borderpad=0.1, handlelength=0.5, labelspacing=0.3, frameon=False)
    
    # 添加R²=0.5的参考线
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='R²=0.5')
    
    # # 更新图例，确保包含参考线
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels, loc='best', 
    #          prop={'size':6, 'family': 'Arial'}, 
    #          borderpad=0.1, handlelength=0.5, labelspacing=0.3, frameon=False)
    
    # 增加图表边距，为图例留出更多空间
    plt.subplots_adjust(right=0.85)  # 减小right值会在右侧留出更多空间给图例
    
    # 保存图像
    output_path = os.path.join(output_dir, 'feature_method_comparison_all.pdf')
    plt.savefig(output_path, format=format.lower(), 
               transparent=True, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    return output_path
