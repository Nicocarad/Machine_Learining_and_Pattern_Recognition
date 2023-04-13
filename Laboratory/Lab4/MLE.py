import numpy as np
import matplotlib.pyplot as plt
from MVG import logpdf_GAU_ND_fast, vcol, vrow, logpdf_GAU_ND


def loglikelihood(XND, m_ML, C_ML):
    
    MVG = logpdf_GAU_ND_fast(XND, m_ML, C_ML)
    ll = np.sum(MVG)
 
    return ll
    
  
    
if __name__ == '__main__':
    
    data_matrix = np.load("utils/XND.npy")
    N = data_matrix.shape[1] # number of samples
    mu = vcol(data_matrix.mean(1)) # data_matrix.mean(1) return a 1-D array so we must transform it into a column
    DC = data_matrix - mu  # performing broadcast, we are removing from all the data the mean
    C = np.dot(DC, DC.T)/N

    ll = loglikelihood(data_matrix, mu, C)
    print(ll)
    
    X1D = np.load('utils/X1D.npy')
    N = X1D.shape[1] # number of samples
    mu = vcol(X1D.mean(1)) # data_matrix.mean(1) return a 1-D array so we must transform it into a column
    DC = X1D - mu  # performing broadcast, we are removing from all the data the mean
    C = np.dot(DC, DC.T)/N

    ll = loglikelihood(X1D, mu, C)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
    plt.show()
    plt.savefig("images/One_dimension_MLE.pdf")
    print(ll)