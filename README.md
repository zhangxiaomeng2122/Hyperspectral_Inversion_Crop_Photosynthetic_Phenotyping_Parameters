Pixel-level High-throughput Estimation of Crop Photosynthetic Phenotyping Parameters Using a Multi-task Deep Learning Framework

## Description

### 📁 Data Directory

```
data/
├── Rice_subsets/          # Rice reflectance dataset
└── Tomato_subsets/        # Tomato reflectance dataset
```

### 📁 Core Modules

```
mt_hyperspectral/
├── data/                  # Data processing module
│   ├── dataset.py        # Dataset definition and loading
│   └── datasplit.py      # datasplit
├── models/               # Model design module  
│   ├── DNN_ensemble.py   # single-output Deep neural network ensemble
│   ├── MDNN_model.py     # mutil-output Deep neural network ensemble
│   ├── ML_baseline.py    # Base model definitions, including PLSR、XGBoost、RF、SVR、MPLSR.
│   └── MTI_model.py      # our mutil-output inversion model
└── training/             # Training module
    ├── trainer.py        # Training pipeline
```

### 📁 Utility Modules

```
utils/
├── plot_setting.py       # Plotting configuration
├── baseset.py            # logger and seed set
```

### 📁 Execution Scripts

```
scripts/
├── DNN_baseline_model.py           # DNN基线模型执行脚本
├── transfer_learning_cross-species.py  # 跨物种迁移学习脚本
├── model_evaluation.py             # 模型评估脚本
└── data_analysis.py                # 数据分析脚本
```

### Detailed Functionality

#### 🔬 数据模块 (`data/`)
- **Rice dataset**: Hyperspectral reflectance data of rice across different growth stages.
- **Tomato dataset**: Hyperspectral reflectance measurements of tomato leaves.

🧠 模型模块 (`mt_hyperspectral/`)

- **Data processing**: Hyperspectral preprocessing, feature extraction, and data augmentation.
- **Model design**: Deep learning architectures including DNN, CNN, and Transformer.
- **Training module**: Training strategies, optimization algorithms, and validation workflows.

#### 🛠️ Utility Module (`utils/`)

- **Plotting tools**: Standardized chart styles and visualization functions.
- **Evaluation metrics**: R², RMSE, RPD, and other performance indicators.
- **Data utilities**: File I/O, format conversion, and general-purpose tools.

#### ⚡ Execution Scripts (`scripts/`)

- **Baseline models**: Training and evaluation of DNN baseline models.
- **Transfer learning**: Cross-crop knowledge transfer experiments.
- **Model evaluation**: Performance testing and comparative analysis of pretrained models.
- **Data analysis**: Dataset statistics and feature visualization.

### Workflow

1. **Data preparation**: Place hyperspectral datasets into the appropriate subdirectories under `data/`.
2. **Model training**: Train models using the scripts in `scripts/`.
3. **Result analysis**: Use the utilities in `utils/` for visualization and analysis.
4. **Model deployment**: Apply trained models for prediction and applications.

1、Specifically, we developed **MTI-Net**, a lightweight multi-task inversion network that simultaneously retrieves multiple PPPs in rice and tomato。

2、A complementary **differentiable band selection module (DRS-Net)** further reduced spectral dimensionality to 40 informative bands while maintaining retrieval accuracy above 0.75, underscoring its potential for sensor design.

3、**ML_baseline_model** are including PLSR、XGBoost、RF、SVR、MPLSR. DL_baseline_model including single-output model (**DNN_baseline_model** )and mutil-output baseline model (**MDNN_baseline_model**)

4、We evaluated transferability between tomato and rice by freezing the encoder layers and fine-tuning only the decoder in **MT_evaluate_full_small_sample.py**.

5、 To further assess generalization, we conducted a cross-validation experiment (**MT_evaluate_full_small_sample.py**) where models trained on pixel-level spectra were tested on canopy-level averaged spectra.
