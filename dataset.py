from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_datasets(num_clients):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    total_size = len(trainset)
    base_size = total_size // num_clients
    remainder = total_size % num_clients
    split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    indices = list(range(total_size))
    trainloaders = []
    start = 0
    for i, split_size in enumerate(split_sizes):
        end = start + split_size
        subset_indices = indices[start:end]
        trainloaders.append(
            DataLoader(Subset(trainset, subset_indices), batch_size=32, shuffle=True)
        )
        start = end
    return trainloaders, DataLoader(testset, batch_size=64)
