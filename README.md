Pixel-level High-throughput Estimation of Crop Photosynthetic Phenotyping Parameters Using a Multi-task Deep Learning Framework

## Description

1、Specifically, we developed **MTI-Net**, a lightweight multi-task inversion network that simultaneously retrieves multiple PPPs in rice and tomato。

2、A complementary **differentiable band selection module (DRS-Net)** further reduced spectral dimensionality to 40 informative bands while maintaining retrieval accuracy above 0.75, underscoring its potential for sensor design.

3、**ML_baseline_model** are including PLSR、XGBoost、RF、SVR、MPLSR. DL_baseline_model including single-output model (**DNN_baseline_model** )and mutil-output baseline model (**MDNN_baseline_model**)

4、We evaluated transferability between tomato and rice by freezing the encoder layers and fine-tuning only the decoder in **MT_evaluate_full_small_sample.py**.

5、 used for 

To further assess generalization, we conducted a cross-validation experiment (**MT_evaluate_full_small_sample.py**) where models trained on pixel-level spectra were tested on canopy-level averaged spectra.
