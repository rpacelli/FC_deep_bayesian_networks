from __future__ import print_function
import numpy as np

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
    var = -np.dot(K0_invK, Kmu) + k0xx
    return bias**2 - Qbar*var

def qbar(labels, invK, N1,lambda1):
    P = len(labels)
    alpha1 = P/N1
    print(alpha1)
    yky = np.matmul(np.matmul(np.transpose(labels), invK), labels)
    print(f'\ny K^(-1) y /P is {yky/P}')
    return ((alpha1-1)-np.sqrt((alpha1-1)**2 + 4*alpha1*yky/P))/2, yky/P

def compute_theory(data, labels, test_data, test_labels, N1, lambda1,lambda0,act,L,infwidth):
    P = len(labels)
    N0 = len(data[0])
    Ptest = len(test_labels)
    data,labels, test_data, test_labels  = data.detach().cpu(),labels.detach().cpu(),test_data.detach().cpu(),test_labels.detach().cpu()
    K = CorrMat(P,data,lambda0)
    if act == "erf":
        kernel = kernel_erf
    else:
        print("here")
        kernel = kernel_relu
    for i in range(L):
        K = kmatrix(P,K,kernel,lambda1)
        np.savetxt( f"kmatrix_layer_{i+1}_P_{P}.txt",K)
    invK = np.linalg.inv(K)
    Qbar = np.array(-1)
    yky = np.array(1.)
    if not infwidth:
        Qbar, yky = qbar(labels, invK, N1, lambda1)
    #Qbar = np.array(-1)
    print(f"\nbar Q is {Qbar}")
    pred_loss = 0
    for p in range(Ptest):
        x,y = np.array(test_data[p]),np.array(test_labels[p])
        pred_loss += test_error(data, x, y, labels, lambda1, invK, Qbar,lambda0,kernel,L).item()
    pred_loss = pred_loss/Ptest
    return pred_loss, Qbar,yky