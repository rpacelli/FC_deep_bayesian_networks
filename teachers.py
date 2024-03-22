
import torch, torchvision, torchvision.transforms as t 
import numpy as np


def getTransforms(self):
        T = t.Compose([
            t.Resize(size = self.side_size), 
            t.ToTensor(), 
            t.Grayscale(), 
            #t.Normalize((0.5),(0.24)),
            t.Lambda(lambda x: torch.flatten(x))
        ])
        return T

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
    

class emnist_dataset: 
    def __init__(self, N,  whichTask):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.whichTask = whichTask
    
    def get_dataset_emnist(self,x_train, y_train, x_test, y_test, which_norm = 'norm_1'):
        if self.whichTask not in ["ABEL-CHJS", "ABFL-CHIS"]:
            raise Exception("Sorry, no task found!")
        elif self.whichTask == 'ABEL-CHJS':
            x_train -= x_train.mean()
            x_train /= x_train.std()
            x_train_g1 = np.concatenate([x_train[(y_train == 1)], x_train[(y_train == 2)], x_train[(y_train == 5)], x_train[(y_train == 12)]])
            x_train_g2 = np.concatenate([x_train[(y_train == 3)], x_train[(y_train == 8)], x_train[(y_train == 10)], x_train[(y_train == 19)]])
            y_train_g1 = np.zeros(len(x_train_g1))
            y_train_g2 = np.ones(len(x_train_g2))
            x_train = np.concatenate([x_train_g1, x_train_g2])
            y_train = np.concatenate([y_train_g1, y_train_g2])
            rp = np.random.permutation(len(y_train))
            x_train = x_train[rp[:self.P]]
            y_train = y_train[rp[:self.P]]
            y_train = 2 * y_train - 1
            #test
            x_test -= x_test.mean()
            x_test /= x_test.std()
            x_test_g1 = np.concatenate([x_test[(y_test == 1)], x_test[(y_test == 2)], x_test[(y_test == 5)], x_test[(y_test == 12)]])
            x_test_g2 = np.concatenate([x_test[(y_test == 3)], x_test[(y_test == 8)], x_test[(y_test == 10)], x_test[(y_test == 19)]])
            y_test_g1 = np.zeros(len(x_test_g1))
            y_test_g2 = np.ones(len(x_test_g2))
            x_test = np.concatenate([x_test_g1, x_test_g2])
            y_test = np.concatenate([y_test_g1, y_test_g2])
            y_test = 2 * y_test - 1
            rp_test = np.random.permutation(len(y_test))
        elif self.whichTask == 'ABFL-CHIS':
            x_train -= x_train.mean()
            x_train /= x_train.std()
            x_train_g1 = np.concatenate([x_train[(y_train == 1)], x_train[(y_train == 2)], x_train[(y_train == 6)], x_train[(y_train == 12)]])
            x_train_g2 = np.concatenate([x_train[(y_train == 3)], x_train[(y_train == 8)], x_train[(y_train == 9)], x_train[(y_train == 19)]])
            y_train_g1 = np.zeros(len(x_train_g1))
            y_train_g2 = np.ones(len(x_train_g2))
            x_train = np.concatenate([x_train_g1, x_train_g2])
            y_train = np.concatenate([y_train_g1, y_train_g2])
            rp = np.random.permutation(len(y_train))
            x_train = x_train[rp[:self.P]]
            y_train = y_train[rp[:self.P]]
            y_train = 2 * y_train - 1
            print(y_train)
            x_test -= x_test.mean()
            x_test /= x_test.std()
            x_test_g1 = np.concatenate([x_test[(y_test == 1)], x_test[(y_test == 2)], x_test[(y_test == 6)], x_test[(y_test == 12)]])
            x_test_g2 = np.concatenate([x_test[(y_test == 3)], x_test[(y_test == 8)], x_test[(y_test == 9)], x_test[(y_test == 19)]])
            y_test_g1 = np.zeros(len(x_test_g1))
            y_test_g2 = np.ones(len(x_test_g2))
            x_test = np.concatenate([x_test_g1, x_test_g2])
            y_test = np.concatenate([y_test_g1, y_test_g2])
            y_test = 2 * y_test - 1
            print(y_test)
            rp_test = np.random.permutation(len(y_test))
        x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
        return x_train, y_train, x_test[rp_test[:self.Ptest]], y_test[rp_test[:self.Ptest]]

    def make_data(self, P, Ptest, batch_size):
        self.P = P 
        self.Ptest = Ptest
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = True,  transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0)
        testset = torchvision.datasets.EMNIST(root = './data', download = True, split = "letters", train = False, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, num_workers = 0)
        data, labels = next(iter(trainloader))
        testData, testLabels = next(iter(testloader))
        # Filter train and test datasets
        #data, labels = filter_by_label(trainloader, self.selectedLabels, P, self.whichTask)
        #testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, self.whichTask)
        #data, testData  = normalizeDataset(data, testData)
        data, labels, testData, testLabels = self.get_dataset_emnist(data, labels, testData, testLabels)
        return data, labels.squeeze(), testData, testLabels.squeeze()
