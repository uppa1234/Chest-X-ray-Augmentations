import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

STATE = 1999
TEST_SIZE = 0.2

# QU
def generate_qu(transform=None):
    qu_f = Path(r'E:\Prut\cxr\data\qatar\Lung_Segmentation_Data\Lung_Segmentation_Data\resized_224\Lung Segmentation Data Binary')

    def qu_target_transform(label):
        # Label is 0, 1 or 2. We want 1->0, 2->1.
        return max(label-1, 0)

    qu_train_dataset = ImageFolder(qu_f / 'Train', transform=transform, target_transform=None)
    qu_val_dataset = ImageFolder(qu_f / 'Val', transform=transform, target_transform=None)
    qu_test_dataset = ImageFolder(qu_f / 'Test', transform=transform, target_transform=None)
    # qu_train_dataset = ImageFolder(qu_f / 'Train', transform=transform, target_transform=qu_target_transform)
    # qu_val_dataset = ImageFolder(qu_f / 'Val', transform=transform, target_transform=qu_target_transform)
    # qu_test_dataset = ImageFolder(qu_f / 'Test', transform=transform, target_transform=qu_target_transform)

    print('Train dataset:', len(qu_train_dataset))
    print('Val dataset:', len(qu_val_dataset))
    print('Test dataset:', len(qu_test_dataset))

    return qu_train_dataset, qu_val_dataset, qu_test_dataset

# MIMIC
class MimicDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Adjusting for the variable number of images per patient
        for _, row in df.iterrows():
            label = row['label']
            subject_id = str(int(row['subject_id']))
            study_id = str(int(row['study_id']))
            for image_path in (root_dir / f'p{subject_id[:2]}/p{subject_id}/s{study_id}').iterdir():
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def generate_mimic(transform=None):

    mimic_f = Path(r'E:\Prut\cxr\data\mimic-cxr\2.0.0\resized_224\files')

    mimic_df = pd.read_csv(r'E:\Prut\cxr\data\mimic-cxr\2.0.0\mimic-cxr-2.0.0-negbio.csv.gz')
    # alternative_mimic_df = pd.read_csv(r'E:\Prut\cxr\data\mimic-cxr\2.0.0\mimic-cxr-2.0.0-chexpert.csv.gz')

    split_df = pd.read_csv(r'E:\Prut\cxr\data\mimic-cxr\2.0.0\mimic-cxr-2.0.0-split.csv.gz')[['study_id', 'subject_id', 'split']]
    mimic_df = mimic_df.merge(split_df, on=['study_id', 'subject_id'], how='left').drop_duplicates()
    assert mimic_df['split'].isna().sum() == 0

    # Define columns which infer presence of pneumonia
    mimic_id_cols = [
        'subject_id',
        'study_id',
        'split'
    ]

    mimic_pneumonia_cols = ['Pneumonia']


    # Combine cols
    mimic_cols = mimic_id_cols + mimic_pneumonia_cols

    # Check that normal and abnormal have no overlap
    assert mimic_df[mimic_df['No Finding'] == 1.][mimic_pneumonia_cols].sum().sum() == 0

    # Target transform
    mimic_df = mimic_df[mimic_cols]
    mimic_df = mimic_df.replace(-1., 0)
    mimic_df = mimic_df[mimic_df[mimic_pneumonia_cols].notna().any(axis=1)]
    mimic_df['label'] = mimic_df[mimic_pneumonia_cols].max(axis=1)

    # View
    mimic_df = mimic_df[mimic_cols + ['label']]

    # Split

    mimic_train_df = mimic_df[mimic_df['split'] == 'train']
    mimic_val_df = mimic_df[mimic_df['split'] == 'validate']
    mimic_test_df = mimic_df[mimic_df['split'] == 'test']

    mimic_train_dataset = MimicDataset(df=mimic_train_df, root_dir=mimic_f, transform=transform)
    mimic_val_dataset = MimicDataset(df=mimic_val_df, root_dir=mimic_f, transform=transform)
    mimic_test_dataset = MimicDataset(df=mimic_test_df, root_dir=mimic_f, transform=transform)


    print('Train dataset:', len(mimic_train_dataset))
    print('Val dataset:', len(mimic_val_dataset))
    print('Test dataset:', len(mimic_test_dataset))

    return mimic_train_dataset, mimic_test_dataset

# RSNA 
class RSNADataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            image_path = (root_dir / row['patientId']).with_suffix('.png')
            label = row['Target']
            self.image_paths.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def generate_rsna(transform=None):
    rsna_f = Path('E:/Prut/cxr/data/rsna/resized_224')
    rsna_train_f = rsna_f / 'stage_2_train_images'

    rsna_df = pd.read_csv('E:/Prut/cxr/data/rsna/stage_2_train_labels.csv')


    rsna_train_df, rsna_test_df = train_test_split(rsna_df, test_size=TEST_SIZE, random_state=STATE, stratify=rsna_df['Target'])

    rsna_train_dataset = RSNADataset(df=rsna_train_df, root_dir=rsna_train_f, transform=transform)
    rsna_test_dataset = RSNADataset(df=rsna_test_df, root_dir=rsna_train_f, transform=transform)

    print('Train dataset:', len(rsna_train_dataset))
    print('Test dataset:', len(rsna_test_dataset))

    return rsna_train_dataset, rsna_test_dataset

# VINDR
class VinDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            image_path = (root_dir / row['image_id']).with_suffix('.png')
            label = row['class_id']
            self.image_paths.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

def generate_vindr(transform=None):

    vindr_f = Path('E:/Prut/cxr/data/vin/resized_224')
    vindr_train_f = vindr_f / 'train'
    vindr_df = pd.read_csv('E:/Prut/cxr/data/vin/train.csv')

    vindr_map = {
        0: 0,
        3: 0,
        10: 0,
        11: 0,
        14: 0,
        4: 1,
        6: 1,
        7: 1,
        8: 1
    }

    # Target Transform
    vindr_df.loc[:, 'class_id'] = vindr_df['class_id'].map(vindr_map) # equals lambda x: dict.get(x, NaN)
    vindr_df = vindr_df.dropna(subset='class_id') # 67914 -> 58591

    vindr_train_df, vindr_test_df = train_test_split(vindr_df, test_size=TEST_SIZE, random_state=STATE, stratify=vindr_df['class_id'])

    vindr_train_dataset = VinDataset(df=vindr_train_df, root_dir=vindr_train_f, transform=transform)
    vindr_test_dataset = VinDataset(df=vindr_test_df, root_dir=vindr_train_f, transform=transform)

    print('Train dataset:', len(vindr_train_dataset))
    print('Test dataset:', len(vindr_test_dataset))

    return vindr_train_dataset, vindr_test_dataset

# UCSD
def generate_ucsd(transform=None):
    ucsd_f = Path('E:/Prut/cxr/data/ucsd3/ZhangLabData/CellData/chest_xray/resized_224')

    ucsd_train_dataset = ImageFolder(ucsd_f / 'Train', transform=transform)
    ucsd_test_dataset = ImageFolder(ucsd_f / 'Test', transform=transform)

    print('Train dataset:', len(ucsd_train_dataset))
    print('Test dataset:', len(ucsd_test_dataset))

    return ucsd_train_dataset, ucsd_test_dataset