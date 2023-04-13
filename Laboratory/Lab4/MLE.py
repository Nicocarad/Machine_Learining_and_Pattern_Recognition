import numpy as np
import matplotlib.pyplot as plt
from MVG import logpdf_GAU_ND_fast, vcol


def loglikelihood(XND, m_ML, C_ML):
    
    return
    
    
    
    
    
    
if __name__ == '__main__':
    
    data_matrix = np.load("utils/XND.npy")
    N = data_matrix.shape[1] # number of samples
    mu = vcol(data_matrix.mean(1)) # data_matrix.mean(1) return a 1-D array so we must transform it into a column
    DC = data_matrix - mu  # performing broadcast, we are removing from all the data the mean
    C = np.dot(DC, DC.T)/N
    
    print("MEAN:")
    print(mu)
    print("COVARIANCE")
    print(C)