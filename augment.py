from datasets import generate_qu, generate_rsna, generate_mimic, generate_ucsd, generate_vindr
from train_nocv import run

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
import albumentations as A
import random
import matplotlib.pyplot as plt
from datetime import datetime

# UNCOMMENT THE ONES TO USE

# Models to test from – For all, use Weights.IMAGENET1K_V1
from torchvision.models import (
    # vgg16_bn, VGG16_BN_Weights,
    # inception_v3, Inception_V3_Weights,
    # resnet50, ResNet50_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    # vit_b_16, ViT_B_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

# FIXED VALUES
SEED = 1999

# MODELS

MODELS = {
    # vgg16_bn: VGG16_BN_Weights.IMAGENET1K_V1,
    # inception_v3: Inception_V3_Weights.IMAGENET1K_V1,
    # resnet50: ResNet50_Weights.IMAGENET1K_V1,
    efficientnet_v2_s: EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    # vit_b_16: ViT_B_16_Weights.IMAGENET1K_V1,
    convnext_tiny: ConvNeXt_Tiny_Weights.IMAGENET1K_V1
}

DATASETS = {
    # TODO add val for rsna, ucsd, vindr
    'qu': generate_qu,
    # 'rsna': generate_rsna,
    # 'mimic': generate_mimic,
    # 'ucsd': generate_ucsd,
    # 'vindr': generate_vindr
}

# Make Albumentations compatible with PyTorch's Compose
class AB:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if self.transform.__class__.__name__ == 'Identity':
            return img
        img = np.array(img)
        augmented = self.transform(image=img)
        img = augmented['image']
        img = Image.fromarray(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

AUGMENTATIONS =[
    nn.Identity(),
    # A.CLAHE(always_apply=True),
    # A.RandomBrightnessContrast(always_apply=True),
    A.GaussianBlur(always_apply=True),
    # A.GaussNoise(always_apply=True),
    A.MotionBlur(always_apply=True),
    A.Downscale(always_apply=True),
    # A.Rotate(always_apply=True),
    # A.HorizontalFlip(always_apply=True),
    A.ElasticTransform(always_apply=True),
    A.GridDistortion(always_apply=True),
    # A.OpticalDistortion(always_apply=True),
]

def generate_compose(transform):
    return T.Compose([
        AB(transform),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

random.seed(SEED)
results_folder = Path('E:/Prut/cxr/src/february/results')

results = []
for data_source, data_function in DATASETS.items():
    print('='*50)
    print(data_source.upper())
    print('='*50)
    
    for model, weights in MODELS.items():
        print('-'*50)
        print(f'Model: {model.__name__}\n')
        print(f'Weights: {weights}')
        print('-'*50)

        for aug in AUGMENTATIONS:
            train_dataset = data_function(generate_compose(nn.Identity()))[0]
            val_dataset = data_function(generate_compose(nn.Identity()))[1]
            test_dataset = data_function(generate_compose(nn.Identity()))[-1]

            # Add augmented dataset to original dataset, if augmenting
            if aug.__class__.__name__ != 'Identity':
                train_dataset = train_dataset + data_function(generate_compose(aug))[0]

            print(f'\nAugmentation: {aug.__class__.__name__}\n')
            train_accuracies, val_accuracies, test_accuracy = run(
                input_model = model,
                input_weights = weights,
                train_dataset = train_dataset,
                val_dataset = val_dataset,
                test_dataset = test_dataset,
                epochs = 20,
                lr = 1e-3
                )

            out_path = results_folder / f'{model.__name__}_{aug.__class__.__name__}_test_acc_{test_accuracy}.csv'
            df = pd.DataFrame([train_accuracies, val_accuracies], index=['train', 'val'], columns=[f'epoch_{i}' for i in range(len(val_accuracies))])
            df.to_csv(out_path, index=False)

            results.append([data_source, model.__name__, aug.__class__.__name__, test_accuracy])

        print('Results so far:', results)

    model_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'Augmentation', 'Test F1 score'])
    model_df.to_csv(results_folder / f'{model.__name__}_results_time_{datetime.now().strftime("%d_%b_%I%p")}.csv', index=False)

results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'Augmentation', 'Test F1 score'])
results_df.to_csv(results_folder / f'overall_results_time_{datetime.now().strftime("%d_%b_%I%p")}.csv', index=False)
print('Complete!')