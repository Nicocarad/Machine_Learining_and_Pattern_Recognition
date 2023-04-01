import numpy as np
import matplotlib.pyplot as plt


def vrow(array):
    return array.reshape((1,array.size))

def vcol(array):
    return array.reshape((array.size, 1))

# X is a column vector
# mu is a colum vector 4x1
# C is the Covariance matrix 4x4
def logpdf_GAU_ND(X,mu,C): 
    
    C_inv = np.linalg.inv(C) # compute the inverse of the Covariance matrix
    _,det_log_C = np.linalg.slogdet(C) # compute the determinant og log|C| ( the det is the second parameter of the function)
    M = X.shape[0]
    X_c = X - mu
    MVG = -M*0.5*np.log(2*np.pi)-0.5*np.log(det_log_C)
    temp = np.dot(C_inv,X_c)
    temp1 = -0.5*(X_c.T).dot(temp)
    MVG = MVG + temp1
   
        
        
    
    return MVG
    
    

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