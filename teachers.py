
import torch, torchvision, torchvision.transforms as t 
import numpy as np

class mnist_dataset: 
    def __init__(self, N,):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.save_data = bool("NaN")
        self.randmat_file = f"randmat_N0_{self.N}.txt"

    def _get_transforms(self):
        T = t.Compose([
            t.Resize(size=self.side_size),
            t.ToTensor(),
            #t.Normalize((0.1307,), (0.3081,)),
            t.Lambda(lambda x: torch.flatten(x))
        ])
        return T

    def make_data(self, P, P_test, device):
        transform_dataset = self._get_transforms()
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_dataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_dataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000, num_workers=0)

        data, labels = next(iter(trainloader))
        ones = labels == torch.ones_like(labels)
        zeros = labels == torch.zeros_like(labels)
        data = data[ones+zeros][:P]
        labels = labels[ones+zeros][:P]
        
        test_data, test_labels = next(iter(testloader))
        ones = test_labels == torch.ones_like(test_labels)
        zeros = test_labels == torch.zeros_like(test_labels)        
        test_data = test_data[ones+zeros][:P_test]
        test_labels = test_labels[ones+zeros][:P_test]

        epsilon = epsilon = 1e-25
        mean1 = torch.mean(data,axis=0)
        stdev1 = torch.std(data, axis = 0)
        data = torch.where(stdev1 < epsilon, data, (data - mean1) / (stdev1 + epsilon) )
        test_data = torch.where(stdev1 < epsilon, test_data, (test_data - mean1) / (stdev1 + epsilon))

        return data, labels, test_data, test_labels

class cifar_dataset: 
    def __init__(self, N,):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.save_data = bool("NaN")
        self.randmat_file = f"randmat_N0_{self.N}.txt"

    def _get_transforms(self):
        T = t.Compose([
            t.Resize(size=self.side_size),
            t.ToTensor(),
            t.Grayscale(),
            #t.Normalize((0.1307,), (0.3081,)),
            t.Lambda(lambda x: torch.flatten(x))
        ])
        return T

    def make_data(self, P, P_test, device):
        transform_dataset = self._get_transforms()
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_dataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=600000, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_dataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size=60000, num_workers=0)
        
        data, labels = next(iter(trainloader))
        ones = labels == 7*torch.ones_like(labels)
        zeros = labels == 9*torch.ones_like(labels)
        #zeros = labels == torch.zeros_like(labels)
        data = data[ones+zeros][:P]
        labels = labels[ones+zeros][:P]
        
        test_data, test_labels = next(iter(testloader))
        ones = test_labels == 7*torch.ones_like(test_labels)
        zeros = test_labels == 9*torch.ones_like(test_labels)
        #zeros = test_labels == torch.zeros_like(test_labels)  
        test_data = test_data[ones+zeros][:P_test]
        test_labels = test_labels[ones+zeros][:P_test]
        
        epsilon = epsilon = 1e-25
        mean1 = torch.mean(data,axis=0)
        stdev1 = torch.std(data, axis = 0)
        data = torch.where(stdev1 < epsilon, data, (data - mean1) / (stdev1 + epsilon) )
        test_data = torch.where(stdev1 < epsilon, test_data, (test_data - mean1) / (stdev1 + epsilon))

        return data, labels, test_data, test_labels

class random_dataset: 
    def __init__(self, N):
        self.N = N

    def make_data(self,P,P_test, device):
        inputs = torch.randn((P,self.N))
        targets = torch.randn(P)
        test_inputs = torch.randn((P_test,self.N))
        test_targets = torch.randn(P_test)
        return inputs, targets, test_inputs, test_targets