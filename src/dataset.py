import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('L')),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
transform_rotate = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('L')),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),    

])


class OriginalDataset(Dataset):
    def __init__(self, data_dir, df,aug=False):
        self.data_dir = data_dir
        self.df = df
#        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        image_file_name = self.df['x'].values[idx]
        x = Image.open(os.path.join(self.data_dir, image_file_name))
        if self.aug:
            x = transform_rotate(x)
        else:
            x = transform(x)

        y = self.df['y'].values[idx]

        return x.float(), torch.tensor(y).float()




