import os, os.path,time,argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("teacher_type", help="choose a teacher type: mnist, cifar10, random", type=str)
    # Net arguments
    parser.add_argument("-L", help="number of hidden layers", type=int, default=1)
    parser.add_argument("-N", "--N", help="size of input data", type=int, default=784)
    parser.add_argument("-N1", "--N1", help="size ofh idden layer(s)", type=int, default=500)    
    parser.add_argument("-act", help="activation function, choose between erf and relu", type=str, default="erf")
    # Learning dynamics arguments
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default=1e-03)
    parser.add_argument("-T", "--T", help="temperature", type=float, default=1e-03)
    parser.add_argument("-lambda1", help="gaussian prior of last layer",type = float, default= 1.)
    parser.add_argument("-lambda0",help="gaussian prior of all other layers", type = float, default= 1.)
    parser.add_argument("-device", "--device",help="choose between cuda and cpu",   type=str, default="cpu")
    parser.add_argument("-epochs", "--epochs", help="number of train epochs", type = int, default = 5000000)
    parser.add_argument("-checkpoint", "--checkpoint", help="# epochs checkpoint", type=int, default=10000)
    parser.add_argument("-R", "--R", help="replica index", type=int, default=1)
    # Data specification
    parser.add_argument("-P", "--P", help="size of training set", type=int, default=500)
    parser.add_argument("-Ptest", "--Ptest", help="# examples in test set", type=int, default=500)    
    # Theory computation
    parser.add_argument("-compute_theory", type = bool, default= False)
    parser.add_argument("-only_theo", type = bool, default= False)
    parser.add_argument("-infwidth", help="compute infinite width theory", type = bool, default= False)

    args = parser.parse_args()
    return args

def train_prep(net, data, labels):
    net.train()
    P = len(data)
    s = np.arange(P)
    np.random.shuffle(s)
    data = data[s]
    labels = labels[s]
    return data, labels

def train(net,data, labels, criterion, optimizer,T,lambda0,lambda1):
    data, labels = train_prep(net, data, labels)
    inputs, targets = data, (labels).unsqueeze(1)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs.float(), targets.float(),net,T,lambda0,lambda1)
    loss.backward()
    optimizer.step()
    return loss.item() 

def test(net, test_data, test_labels, criterion):
        net.eval()
        P_test = len(test_data)
        with torch.no_grad():
                inputs, targets = test_data, (test_labels).unsqueeze(1)
                outputs = net(inputs)
                loss = criterion(outputs, targets) 
        return loss.item()

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.erf(x)

class Norm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, x):
        return x/self.norm

class FCNet: 
    def __init__(self, N0, N1,  L):
        self.N0 = N0 
        self.N1 = N1
        self.L = L
    def Sequential(self, bias, act_func):
        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "erf":
            act = Erf()
        modules = []
        first_layer = nn.Linear(self.N0, self.N1, bias=bias)
        init.normal_(first_layer.weight, std = 1)
        if bias:
            init.constant_(first_layer.bias,0)
        modules.append(Norm(np.sqrt(self.N0)))
        modules.append(first_layer)
        for l in range(self.L-1): 
            modules.append(act)
            modules.append(Norm(np.sqrt(self.N1)))
            layer = nn.Linear(self.N1, self.N1, bias = bias)
            init.normal_(layer.weight, std = 1)
            if bias:
                init.normal_(layer.bias,std = 1)
            modules.append(layer)
        modules.append(act)
        modules.append(Norm(np.sqrt(self.N1)))
        last_layer = nn.Linear(self.N1, 1, bias=bias)  
        init.normal_(last_layer.weight, std = 1) 
        if bias:
                init.normal_(last_layer.bias,std = 1)
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} dense hidden layer(s) of size {self.N1} with {act_func} actviation function', sequential)
        return sequential

def find_device(device):
    try:
        if device == 'cpu':
            raise TypeError
        torch.cuda.is_available() 
        device = 'cuda'	
    except:
        device ='cpu'
        print('\nWorking on', device)
    return device

def cuda_init(net, device):
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        CUDA_LAUNCH_BLOCKING=1

class LangevinOpt(optim.Optimizer):
    def __init__(self, model: nn.Module, lr, temperature):
        defaults = {
            'lr': lr,
            'temperature': temperature
        }
        groups = []
        for i, layer in enumerate(model.children()):
            groups.append({'params': layer.parameters(),
                           'lr': lr,
                           'temperature': temperature})
        super(LangevinOpt, self).__init__(groups, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            learning_rate = group['lr']
            temperature = group['temperature']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                d_p = torch.randn_like(parameter) * (2*learning_rate*temperature)**0.5
                d_p.add_(parameter.grad, alpha=-learning_rate)
                parameter.add_(d_p)

def reg_loss(output, target,net,T,lambda0,lambda1):
    loss = 0
    for i in range(len(net)):
        if i == len(net)-1:
            loss += (0.5*lambda1*T)*(torch.linalg.matrix_norm(net[i].weight)**2)
        else:
            if isinstance(net[i],nn.Linear):
                loss += (0.5*lambda0*T)*(torch.linalg.matrix_norm(net[i].weight)**2)
            
    loss += 0.5*torch.sum((output - target)**2)
    return loss

def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)

def make_folders(root_dir, args):
    #CREATION OF 2ND FOLDER WITH TEACHER & NETWORK SPECIFICATIONS
    first_subdir = root_dir + f'teacher_{args.teacher_type}_net_{args.L}hl_actf_{args.act}/'
    make_directory(first_subdir)
    #CREATION OF 3RD FOLDER WITH RUN SPECIFICATIONS
    attributes_string = f'lr_{args.lr}_T_{args.T}_lambda0_{args.lambda0}_lambda1_{args.lambda1}_{args.L}hl_N_{args.N}_N1_{args.N1}'
    run_folder = first_subdir + attributes_string + '/'
    make_directory(run_folder)
    return first_subdir, run_folder