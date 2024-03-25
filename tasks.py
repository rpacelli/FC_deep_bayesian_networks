
import torch, torchvision, torchvision.transforms as t 
import numpy as np

def relu(x):
    return x if x > 0 else 0

def filter_by_label(data_loader, labels, P, whichTask):
    data, target = next(iter(data_loader))
    mask = torch.zeros_like(target, dtype = torch.bool)
    for label in labels:
        mask |=  target  ==  label
    filtered_data = data[mask][P*whichTask:P*whichTask+P]
    filtered_labels = target[mask][P*whichTask:P*whichTask+P]
    zero_one_labels = [0 if x  ==  labels[0] else 1 for x in filtered_labels]
    return filtered_data, torch.tensor(zero_one_labels)

def getTransforms(self):
        T = t.Compose([
            t.Resize(size = self.side_size), 
            t.ToTensor(), 
            t.Grayscale(), 
            #t.Normalize((0.5),(0.24)),
            t.Lambda(lambda x: torch.flatten(x))
        ])
        return T

def normalizeDataset(data, testData):
    mean = data.mean()
    std = data.std()
    data = data - mean
    testData = testData - mean
    data = data / std
    testData = testData /std
    return data, testData


class mnist_dataset: 
    def __init__(self, N, selectedLabels,whichTask):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.whichTask = whichTask
    def make_data(self, P, Ptest, batch_size):#selectedLabels is a list, for example [2, 3] will select the labels 2 and 3 from mnist
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size)
        testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 10000, num_workers = 0)
        #all_data, targets = next(iter(trainloader))
        # Filter train and test datasets
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, self.whichTask)
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, self.whichTask)
        data, testData  = normalizeDataset(data, testData)
        return data, labels.unsqueeze(1), testData, testLabels.unsqueeze(1)
 

class cifar_dataset: 
    def __init__(self, N, selectedLabels, whichTask):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.whichTask = whichTask
    def make_data(self, P, Ptest, batch_size):
        transformDataset = getTransforms(self)
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transformDataset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 0)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transformDataset)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, num_workers = 0)
        #all_data, targets = next(iter(trainloader))
        # Filter train and test datasets
        data, labels = filter_by_label(trainloader, self.selectedLabels, P, self.whichTask)
        testData, testLabels = filter_by_label(testloader, self.selectedLabels, Ptest, self.whichTask)
        data, testData  = normalizeDataset(data, testData)
        return data, labels.unsqueeze(1), testData, testLabels.unsqueeze(1)


class emnist_dataset: 
    def __init__(self, N, selectedLabels, whichTask):
        self.N = N
        self.side_size = int(np.sqrt(self.N))
        self.selectedLabels = selectedLabels
        self.whichTask = whichTask
    
    def get_dataset_emnist(self,x_train, y_train, x_test, y_test):
        if self.whichTask not in ["ABEL-CHJS", "ABFL-CHIS"]:
            raise Exception("Sorry, no task found!")
        elif self.whichTask == 'ABEL-CHJS':
            x_train_g1 = np.concatenate([x_train[(y_train == 1)], x_train[(y_train == 2)], x_train[(y_train == 5)], x_train[(y_train == 12)]])
            x_train_g2 = np.concatenate([x_train[(y_train == 3)], x_train[(y_train == 8)], x_train[(y_train == 10)], x_train[(y_train == 19)]])
            y_train_g1 = np.zeros(len(x_train_g1))
            y_train_g2 = np.ones(len(x_train_g2))
            x_train = np.concatenate([x_train_g1, x_train_g2])
            y_train = np.concatenate([y_train_g1, y_train_g2])
            #rp = np.random.permutation(len(y_train))
            #x_train = x_train[rp[:self.P]]
            #y_train = y_train[rp[:self.P]]
            y_train = 2 * y_train - 1
            x_test_g1 = np.concatenate([x_test[(y_test == 1)], x_test[(y_test == 2)], x_test[(y_test == 5)], x_test[(y_test == 12)]])
            x_test_g2 = np.concatenate([x_test[(y_test == 3)], x_test[(y_test == 8)], x_test[(y_test == 10)], x_test[(y_test == 19)]])
            y_test_g1 = np.zeros(len(x_test_g1))
            y_test_g2 = np.ones(len(x_test_g2))
            x_test = np.concatenate([x_test_g1, x_test_g2])
            y_test = np.concatenate([y_test_g1, y_test_g2])
            y_test = 2 * y_test - 1
        elif self.whichTask == 'ABFL-CHIS':
            x_train_g1 = np.concatenate([x_train[(y_train == 1)], x_train[(y_train == 2)], x_train[(y_train == 6)], x_train[(y_train == 12)]])
            x_train_g2 = np.concatenate([x_train[(y_train == 3)], x_train[(y_train == 8)], x_train[(y_train == 9)], x_train[(y_train == 19)]])
            y_train_g1 = np.zeros(len(x_train_g1))
            y_train_g2 = np.ones(len(x_train_g2))
            x_train = np.concatenate([x_train_g1, x_train_g2])
            y_train = np.concatenate([y_train_g1, y_train_g2])
            y_train = 2 * y_train - 1
            x_test_g1 = np.concatenate([x_test[(y_test == 1)], x_test[(y_test == 2)], x_test[(y_test == 6)], x_test[(y_test == 12)]])
            x_test_g2 = np.concatenate([x_test[(y_test == 3)], x_test[(y_test == 8)], x_test[(y_test == 9)], x_test[(y_test == 19)]])
            y_test_g1 = np.zeros(len(x_test_g1))
            y_test_g2 = np.ones(len(x_test_g2))
            x_test = np.concatenate([x_test_g1, x_test_g2])
            y_test = np.concatenate([y_test_g1, y_test_g2])
            y_test = 2 * y_test - 1
            print(y_test)
        x_train = x_train[:self.P]
        y_train = y_train[:self.P]
        x_test =  x_test[:self.Ptest]
        y_test = y_test[:self.Ptest]
        x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
        x_train -= x_train.mean()
        x_train /= x_train.std()
        x_test -= x_train.mean()
        x_test /= x_train.std()
        return x_train, y_train, x_test, y_test 

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

class random_dataset: 
    def __init__(self, N):
        self.N = N

    def make_data(self, P, Ptest):
        inputs = torch.randn((P, self.N))
        targets = torch.randn(P)
        test_inputs = torch.randn((Ptest, self.N))
        test_targets = torch.randn(Ptest)
        return inputs, targets, test_inputs, test_targets
    
class hmm_dataset:
    def __init__(self, N, D, F, Wt, whichTask, q, eta, rho, actx):
        self.N = N
        self.D = D
        self.side_size = int(np.sqrt(self.N))
        self.save_data = bool("NaN")
        self.randmat_file = f"randmat_N0_{self.N}.txt"
        self.whichTask = whichTask
        self.q = q
        self.eta = eta
        self.rho = rho
        self.actx = actx
        self.Wt = Wt
        self.F = F

    def perturb_teachers(self):
        Wt_perturbed = np.copy(self.Wt)
        n_latent = len(Wt_perturbed)
        #print("latent dimension is", n_latent)
        Wt_perturbed = self.q * Wt_perturbed + np.sqrt(1 - self.q**2) * np.random.normal(0., 1., size = (n_latent))
        return Wt_perturbed
    
    def perturb_features(self):
        F_perturbed = np.copy(self.F)
        n_latent, n = np.shape(F_perturbed)
        n_rows = int(self.rho*n_latent)
        F_perturbed = self.eta * F_perturbed + np.sqrt(1 - self.eta**2) * np.random.normal(0, 1., size = (n_latent, n))
        F_perturbed[:n_rows] = np.random.normal(0, 1., size = (n_rows, n))
        return F_perturbed

    def get_hmm(self, P, P_test):
        if self.whichTask == 'target':
            Wt = self.perturb_teachers()
            F = self.perturb_features()
        else: 
            Wt = self.Wt
            F = self.F
        c_train = np.random.normal(0., 1., size = (P, self.D))
        c_test = np.random.normal(0., 1., size = (P_test, self.D))
        if self.actx == 'tanh':
            x_train = np.tanh(c_train@F/np.sqrt(self.D))
            x_test = np.tanh(c_test@F/np.sqrt(self.D))
        if self.actx == 'relu':
            x_train = relu(c_train@F/np.sqrt(self.D))
            x_test = relu(c_test@F/np.sqrt(self.D))
        y_train = np.sign(c_train @ Wt)
        y_test = np.sign(c_test @ Wt)
        x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
        return x_train, y_train, x_test, y_test

    def make_data(self, P, P_test, batch_size):

        if self.whichTask not in ["source", "target"]:
            raise Exception("Sorry, no task found!")
        else:
            data, labels, test_data, test_labels = self.get_hmm(P, P_test)
        #print("the dimension of data is ", len(data[0]))
        return data, labels, test_data, test_labels