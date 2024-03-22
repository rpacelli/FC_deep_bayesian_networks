from __future__ import print_function
import numpy as np
import torch
from scipy.optimize import minimize

def k0(x,y,lambda0):
    N0 = len(x)
    return (1/(lambda0*N0)) * np.dot(x,y)

def CorrMat(P,data,lambda0):
    C = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            C[i][j] = k0(data[i], data[j], lambda0)
            C[j][i] = C[i][j]
    return C  

def kmatrix(P,C,kernel,lambda1):
    K = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            K[i][j] = kernel(C[i][i], C[i][j], C[j][j],lambda1)
            K[j][i] = K[i][j]
    return K

def kernel_erf(k0xx,k0xy,k0yy,lambda1):
    return (2/(lambda1*np.pi))*np.arcsin((2*k0xy)/np.sqrt((1+2*k0xx)*(1+2*k0yy)))

def kappa1(u):
    return (1/(2*np.pi))*(u * (np.pi - np.arccos(u))+ np.sqrt(1-u**2))   

def kernel_relu(k0xx, k0xy, k0yy,lambda1):
    if k0xy == k0xx:
        return np.sqrt(k0xx*k0yy)/(lambda1*2)
    else:
        u = k0xy/np.sqrt(k0xx*k0yy)
        kappa = kappa1(u)
        return np.sqrt(k0xx*k0yy)*kappa/(lambda1)

def test_error(data,x,y,labels,lambda1,invK,Qbar,lambda0,kernel, L):
    P = len(data)
    k0xx = k0(x,x,lambda0)
    k0yyvec, Kmu = np.random.randn(P), np.random.randn(P)
    for i in range(P):
        k0xy = k0(x,data[i],lambda0)
        k0yy = k0(data[i],data[i],lambda0)
        k0yyvec[i] = k0yy
        Kmu[i] = kernel(k0xx,k0xy,k0yy,lambda1) if L ==1 else kernel(k0xx,k0xy,k0yy,lambda0)
    k0xx = kernel(k0xx,k0xx,k0xx,lambda1) if L==1 else kernel(k0xx,k0xx,k0xx,lambda0)
    for l in range(L-1):
        if l== L-2:
            Lambda = lambda1
        else:
            Lambda = lambda0 
        for i in range(P):
            k0yy = kernel(k0yyvec[i],k0yyvec[i],k0yyvec[i],Lambda)
            k0yyvec[i] = k0yy
            Kmu[i] = kernel(k0xx,Kmu[i],k0yy,Lambda) 
        k0xx = kernel(k0xx,k0xx,k0xx,Lambda)
    K0_invK = np.matmul(Kmu, invK)
    bias = -np.dot(K0_invK, labels) + y
    var = -Qbar/lambda1*np.dot(K0_invK, Kmu) + k0xx
    pred_err = bias**2 + Qbar/lambda1*var
    return pred_err.item()


def compute_theory(data, labels, test_data, test_labels, args):
    P = len(labels)
    N0 = len(data[0])
    Ptest = len(test_labels)
    data,labels, test_data, test_labels  = data.detach().cpu(),labels.detach().cpu(),test_data.detach().cpu(),test_labels.detach().cpu()
    targets =  torch.tensor(labels, dtype = torch.float64)
    K = CorrMat(P,data,args.lambda0)
    kernel = eval(f"kernel_{args.act}")
    for i in range(args.L):
        K = kmatrix(P,K,kernel,args.lambda1)
    
    Qbar = np.array(1)
    yky = np.array(1.)
    ktensor = torch.from_numpy(K)
    if not args.infwidth:
        [K_eigval, K_eigvec] = np.linalg.eig(K)
        U = K_eigvec
        #print(U)
        Udag = np.transpose(U)
        diag_K = np.diagflat(K_eigval)
        ytilde = np.matmul( Udag, targets.squeeze())
        x0 = 1.0
        bns = ((1e-8,np.inf),)
        params = args.T, args.P,args.N1,args.lambda1,diag_K,K_eigval,ytilde
        res = minimize(S, x0, bounds=bns, tol=1e-20,args = params)
        Qbar = (res.x).item()
    print(f"\nbar Q is {Qbar}")
    invK = np.linalg.inv(Qbar/args.lambda1*K+ (args.T)*np.eye(P))
    yky = np.matmul(np.matmul(np.transpose(labels), invK), labels)
    print(f'\ns /P is {yky/args.P}')
    pred_loss = 0
    for p in range(Ptest):
        x,y = np.array(test_data[p]),np.array(test_labels[p])
        pred_loss += test_error(data, x, y, labels, args.lambda1, invK, Qbar,args.lambda0,kernel,args.L)
    pred_loss = pred_loss/Ptest
    return pred_loss, Qbar, yky.item()

def S(x, *params):
    T, P,N1,lambda1,diag_K,K_eigval,ytilde = params 
    HH = T * np.identity(P) + (x/lambda1) * diag_K
    HH_inv = np.linalg.inv(HH)
    part_1 = -1 + x + np.log(1/x)
    part_2 = (1/N1) * np.sum(np.log(T + x*K_eigval/lambda1))
    part_3 = (1/(lambda1 * N1)) * np.dot(ytilde, np.dot(HH_inv, ytilde))
    return (part_1 + part_2 + part_3)