# Chest X-ray Augmentations
Comparing data augmentation and lung segmentation techniques in Chest X-rays

Datasets not uploaded due to some specific restrictive permissions.

1. After downloading images, resize them and save them as png - \
resize_dicom.py and resize_jpg_png.py

2. Generate PyTorch datasets - datasets.py

3. Trained a lung sementation model from another dataset in segment.ipynb \
The code for training is in segment.ipynb , the model is lung_segment_model_180823.pt \
Created a transform compatible with PyTorch in crop.py

4. Training and testing loops for
- Train-val-test split - train_nocv.py
- k-fold cross-validation - train_cv.py

5. Main training script - **augment.py** \
(import other modules)