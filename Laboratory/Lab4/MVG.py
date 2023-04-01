import numpy as np
import matplotlib.pyplot as plt


def vrow(array):
    return array.reshape((1,array.size))

def vcol(array):
    return array.reshape((array.size, 1))

# X is a column vector
# mu is a colum vector 4x1
# C is the Covariance matrix 4x4
def logpdf_GAU_ND_1Sample(x,mu,C): 
    
    C_inv = np.linalg.inv(C) # compute the inverse of the Covariance matrix
    _,det_log_C = np.linalg.slogdet(C) # compute the determinant of log|C| ( the det is the second parameter of the function)
    M = x.shape[0]
    x_c = x - mu
    MVG = - M * 0.5 * np.log(2*np.pi) - 0.5 * det_log_C - 0.5 * np.dot(x_c.T, np.dot(C_inv,x_c))
   
    return MVG
    

def logpdf_GAU_ND(X, mu, C):
    
    Y = []
    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:, i:i+1], mu, C))
    return np.array(Y).ravel()
    

if __name__ == '__main__':
    
    
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.show()
 
 
    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    print(np.abs(pdfSol - pdfGau).max())