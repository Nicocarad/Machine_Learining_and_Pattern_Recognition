import numpy as np
import matplotlib.pyplot as plt
from MVG import logpdf_GAU_ND_fast, vcol, vrow, logpdf_GAU_ND


def loglikelihood(XND, m_ML, C_ML):
    
    MVG = logpdf_GAU_ND_fast(XND, m_ML, C_ML)
    ll = np.sum(MVG)
 
    return ll

def compute_loglikelihood(data_matrix):
    
    N = data_matrix.shape[1]
    mu = vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = np.dot(DC, DC.T)/N
    ll = loglikelihood(data_matrix, mu, C)
    
    return ll,mu,C
  
    
if __name__ == '__main__':
    
    #  LIKELIHOOD ESTIMATE FOR A GENERIC MATRIX
    XND = np.load("utils/XND.npy")
    ll1 = compute_loglikelihood(XND)[0]
    print(ll1)
     # LIKELIHOOD FOR ONE DIMENSIONAL SAMPLES
    X1D = np.load('utils/X1D.npy')
    ll2,mu,C = compute_loglikelihood(X1D)
    print(ll2)
    
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))
    plt.show()
    plt.savefig("images/One_dimension_MLE.pdf")
   