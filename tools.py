import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as transforms
import random


def get_free_gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            raise RuntimeError("No GPU found by pynvml.")

        max_memory = 0
        best_gpu = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if memory_info.free > max_memory:
                max_memory = memory_info.free
                best_gpu = i

        return f'cuda:{best_gpu}'

    except (ImportError, RuntimeError):
        # If pynvml not available or no GPUs found
        if torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'

class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, element):
        self.sum +=element
        self.count += 1

    def get_average(self):
        if self.count == 0:
            return 0
        return self.sum / self.count
    

class TrainingMetrics:
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def update(self, loss, outputs, labels):
        self.losses.append(loss.item())

        if labels.dim()>1:
            labels = torch.argmax(labels,dim = 1)

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0) 
        self.accuracies.append(accuracy)

    def get_epoch_stats(self):
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        avg_accuracy = sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0
        return avg_loss, avg_accuracy

    def reset(self):
        self.losses = []
        self.accuracies = []

class TransformedTensorDataset(Dataset):
        def __init__(self, x, y, transform):
            self.x = x
            self.y = y
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            img, label = self.x[index], self.y[index]
            if self.transform:
                img = to_pil_image(img)
                img = self.transform(img)
            return img, label
        
def get_dataset(name='cifar10', data_root = './data/clean', train = True, transform = transforms.Compose([transforms.ToTensor()])):

    if name.lower() not in ['cifar10', 'cifar100', 'svhn']:
        raise ValueError(f"Unsupported dataset: {name}")

    if name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root=data_root, train=train, download=True,
                                   transform=transform)
    elif name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(root=data_root, train=train, download=True,
                                    transform=transform)
    elif name.lower() == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(root=data_root, split=split, download=True,
                                transform=transform)
    return dataset

def load_set(path, transform = None):

    unlearnable_data = torch.load(path)
    unlearnable_x = unlearnable_data['data']
    unlearnable_y = unlearnable_data['labels']

    dataset = TransformedTensorDataset(unlearnable_x, unlearnable_y, transform)
    return dataset

class SplitMergeDataset(Dataset):
    def __init__(self, dataset1, dataset2, ratio=1.0, seed=None):
        assert len(dataset1) == len(dataset2), "Both datasets must have the same length"
        assert 0 < ratio < 1, "Ratio must be between 0 and 1"

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_len = len(dataset1)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

        indices = list(range(self.total_len))
        random.shuffle(indices)

        # Split indices based on the given ratio
        split_point = int(ratio * self.total_len)
        self.indices1 = set(indices[:split_point])
        self.indices2 = set(indices[split_point:])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx in self.indices1:
            img, label = self.dataset1[idx]
            if isinstance(label, int):
                label = torch.tensor(label)
            return img, label
        elif idx in self.indices2:
            img, label = self.dataset2[idx]
            if isinstance(label, int):
                label = torch.tensor(label)
            return img, label
        else:
            raise IndexError(f"Index {idx} is out of valid range")