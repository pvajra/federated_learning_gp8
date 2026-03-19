from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_datasets(num_clients):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    data_size = len(trainset) // num_clients
    trainloaders = [DataLoader(Subset(trainset, list(range(i * data_size, (i + 1) * data_size))), batch_size=32, shuffle=True) for i in range(num_clients)]
    return trainloaders, DataLoader(testset, batch_size=64)
