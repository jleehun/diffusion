import torchvision
import torchvision.transforms as T 
from torch.utils.data import Dataset
from PIL import Image

MNIST_STATS = {
    'mean' : [0.1307],
    'std' : [0.3081] 
}

CIFAR10_STATS = {
    'mean' : [0.4914, 0.4822, 0.4465],
    'std' : [0.2023, 0.1994, 0.2010]
}

def make_dataset(data,  data_path, train_augment=False):
    if data=="mnist":

        transform = T.Compose([
                    T.Resize(32),
                    T.ToTensor(), 
                    T.Normalize(MNIST_STATS['mean'], MNIST_STATS['std'])
                ])
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True,download=True, transform=transform)
        valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False,download=True, transform=transform)
    elif data in ['cifar10', 'cifar100']:
        if train_augment:
            train_transform = T.Compose([T.Resize(32), 
                                T.RandomCrop(32, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(), 
                                T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
           
        else:
             train_transform = T.Compose([T.Resize(32), 
                                T.ToTensor(), 
                                T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
        valid_transform = T.Compose([T.Resize(32), 
                            T.ToTensor(), 
                            T.Normalize(CIFAR10_STATS['mean'], CIFAR10_STATS['std'])])
        if data == 'cifar10':
            train_dataset  = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform) 
            valid_dataset  = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=valid_transform) 
            
            
    return train_dataset, valid_dataset


class CustomDataset(Dataset):

    def __init__(self, data_df, transforms):
        image_paths = []
        for idx, row in data_df.iterrows():
            image_path = row["image_path"]

            image_paths.append(image_path)
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return {"input": image}
