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
├── data/                 # Data processing module
│   ├── datasplit.py      # datasplit
│   └── dataset.py        # Dataset definition and loading
├── models/               # Model design module  
│   ├── DNN_ensemble.py   # single-output Deep neural network ensemble
│   ├── MDNN_model.py     # mutil-output Deep neural network ensemble
│   ├── ML_baseline.py    # Base model definitions, including PLSR、XGBoost、RF、SVR、MPLSR.
│   └── MTI_model.py      # our mutil-output inversion model
└── training/             # Training module
    └── trainer.py        # Training pipeline
```

### 📁 Utility Modules

```
utils/
├── baseset.py            # logger and seed set
├── io.py                 # Input/output processing
├── metrics.py            # Evaluation metrics
├── plot_setting.py       # Plotting configuration
└── visualization.py      # Visualization utilities
```

### 📁 Execution Scripts

```
scripts/
├── DNN_baseline_model.py           	# Execution script for the DNN baseline model
├── DRS-Net.py 							# Execution script for the band selection
├── MDNN_baseline_model.py          	# Execution script for the MDNN baseline model
├── ML_baseline_model.py           		# Execution script for the ML baseline model
├── MT_evaluate_full_small_sample.py    # Execution script for cross-validation experiment
├── MTI_Net.py           				# Execution script for the MTI baseline model
└── transfer_learning_cross-species.py  # Cross-species transfer learning

```

### 🧠Detailed Functionality

- **DNN_baseline_model:** deep learning baseline model.
- **DRS-Ne**t: DRS-Net means differentiable band selection module, reduced spectral dimensionality to 40 informative bands while maintaining retrieval accuracy above 0.75, underscoring its potential for sensor design.
- **MDNN_baseline_model**: mutil-output deep learning baseline model.
- **ML_baseline_model**: including PLSR、XGBoost、RF、SVR、MPLSR.
- **MT_evaluate_full_small_sample**: models trained on pixel-level spectra were tested on canopy-level averaged spectra.
- **MTI_Net**:  a lightweight multi-task inversion network that simultaneously retrieves multiple PPPs in rice and tomato.
- **transfer_learning_cross-species**: evaluated transferability between tomato and rice by freezing the encoder layers and fine-tuning only the decoder.

### Workflow

1. **Data preparation**: Place hyperspectral datasets into the appropriate subdirectories under `data/`.
2. **Model training**: Train models using the scripts in `scripts/`.
3. **Result analysis**: Use the utilities in `utils/` for visualization and analysis.
4. **Model deployment**: Apply trained models for prediction and applications.
