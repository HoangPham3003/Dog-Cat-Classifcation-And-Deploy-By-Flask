import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class DogCatDataset(Dataset):
    """
    create dataset
    """

    def __init__(self, dataset_file, transform=None):
        self.dataset = pd.read_csv(dataset_file)
        self.transform = transform
    

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        image_path, label = self.dataset.iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
            image = np.array(image)
        return image, label


class DogCatDataLoader:
    """
    load dataset --> return train and valid dataset
    """

    def __init__(self, dataset_file,
                       batch_size=8,
                       random_seed=42,
                       valid_size=0.2,
                       shuffle=True):
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle
    

    def create_data(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010],)
        tranform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = DogCatDataset(dataset_file=self.dataset_file, transform=tranform) 
        valid_dataset = DogCatDataset(dataset_file=self.dataset_file, transform=tranform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=self.batch_size, sampler=valid_sampler
        )

        return (train_loader, valid_loader)
