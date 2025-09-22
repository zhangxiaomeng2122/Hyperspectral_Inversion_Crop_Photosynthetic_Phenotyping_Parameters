Pixel-level High-throughput Estimation of Crop Photosynthetic Phenotyping Parameters Using a Multi-task Deep Learning Framework

## Description

### 📁 数据目录

```
data/
├── Rice_subsets/          # 水稻反射率数据集
└── Tomato_subsets/        # 番茄反射率数据集
```

### 📁 核心模块

```
mt_hyperspectral/
├── data/                  # 数据处理模块
│   ├── dataset.py        # 数据集定义和加载
│   └── preprocessing.py  # 数据预处理工具
├── models/               # 模型设计模块  
│   ├── DNN_ensemble.py   # 深度神经网络集成模型
│   ├── base_models.py    # 基础模型定义
│   └── architectures/    # 模型架构实现
└── training/             # 模型训练模块
    ├── trainer.py        # 训练器实现
    ├── loss_functions.py # 损失函数定义
    └── optimization.py   # 优化策略
```

### 📁 工具模块

```
utils/
├── plot_setting.py       # 绘图基础配置
├── metrics.py           # 评估指标计算
├── visualization.py     # 数据可视化工具
└── io.py               # 输入输出处理
```

### 📁 执行脚本

```
scripts/
├── DNN_baseline_model.py           # DNN基线模型执行脚本
├── transfer_learning_cross-species.py  # 跨物种迁移学习脚本
├── model_evaluation.py             # 模型评估脚本
└── data_analysis.py                # 数据分析脚本
```

### 详细功能说明

#### 🔬 数据模块 (`data/`)
- **水稻数据集**: 包含水稻在不同生长阶段的高光谱反射率数据
- **番茄数据集**: 包含番茄叶片的高光谱反射率测量数据

#### 🧠 模型模块 (`mt_hyperspectral/`)
- **数据处理**: 高光谱数据预处理、特征提取和数据增强
- **模型设计**: 深度学习模型架构，包括DNN、CNN、Transformer等
- **训练模块**: 模型训练策略、优化算法和验证流程

#### 🛠️ 工具模块 (`utils/`)
- **绘图工具**: 统一的图表样式设置和可视化函数
- **评估指标**: R²、RMSE、RPD等模型性能评估指标
- **数据处理**: 文件读写、数据格式转换等通用工具

#### ⚡ 执行脚本 (`scripts/`)
- **基线模型**: DNN基础模型的训练和评估
- **迁移学习**: 跨作物物种的知识迁移实验
- **模型评估**: 预训练模型的性能测试和对比分析
- **数据分析**: 数据集统计分析和特征可视化

### 使用流程

1. **数据准备**: 将高光谱数据放置在 `data/` 对应子目录中
2. **模型训练**: 使用 `scripts/` 中的脚本进行模型训练
3. **结果分析**: 利用 `utils/` 中的工具进行结果可视化和分析
4. **模型部署**: 基于训练好的模型进行预测和应用

1、Specifically, we developed **MTI-Net**, a lightweight multi-task inversion network that simultaneously retrieves multiple PPPs in rice and tomato。

2、A complementary **differentiable band selection module (DRS-Net)** further reduced spectral dimensionality to 40 informative bands while maintaining retrieval accuracy above 0.75, underscoring its potential for sensor design.

3、**ML_baseline_model** are including PLSR、XGBoost、RF、SVR、MPLSR. DL_baseline_model including single-output model (**DNN_baseline_model** )and mutil-output baseline model (**MDNN_baseline_model**)

4、We evaluated transferability between tomato and rice by freezing the encoder layers and fine-tuning only the decoder in **MT_evaluate_full_small_sample.py**.

5、 To further assess generalization, we conducted a cross-validation experiment (**MT_evaluate_full_small_sample.py**) where models trained on pixel-level spectra were tested on canopy-level averaged spectra.
