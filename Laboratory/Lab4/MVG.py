import numpy as np
import matplotlib.pyplot as plt


def vrow(array):
    return array.reshape((1,array.size))

def vcol(array):
    return array.reshape((array.size, 1))

def logpdf_GAU_ND(X,mu,C): 
    
    C_inv = np.linalg.inv(C) # compute the inverse of the Covariance matrix
    _,det_log_C = np.linalg.slogdet(C) # compute the determinant og log|C| ( the det is the second parameter of the function)
    M = C.shape[1]
    for i in range (X.shape[1]):
        X_c = X[i] - mu
        MVG = -M*0.5*np.log(2*np.pi)-0.5*np.log(det_log_C)
        temp = np.dot(X_c.T,C_inv)
        temp1 = -0.5*X_c.dot(temp)
        MVG = MVG + temp1
        Y = np.hstack(MVG)
    
    return Y
    
    

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