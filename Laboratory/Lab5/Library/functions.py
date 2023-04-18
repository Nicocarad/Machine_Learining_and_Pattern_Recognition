import numpy as np


def vrow(array):
    return array.reshape((1,array.size))

def vcol(array):
    return array.reshape((array.size, 1))

def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = np.dot(DC, DC.T)/N
    
    return mu, C

def logpdf_GAU_ND_fast(X, mu, C):
    
    X_c = X - mu
    M = X.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (X_c*np.dot(L, X_c)).sum(0)
    
    return const - 0.5 * logdet - 0.5 *v 


