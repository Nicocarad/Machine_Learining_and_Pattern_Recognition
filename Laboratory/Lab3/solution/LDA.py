import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# This function takes an one-dimensional numpy array, 
# converts it to a vertical column form of shape (n,1) where n is the size of the input array  
def vcol(array):
    return array.reshape((array.size, 1))

# This function takes an one-dimensional numpy array,
# converts it to a horizontal row form of shape (1,n) where n is the size of the input array
def vrow(array):
    return array.reshape((1,array.size))


# this data are imported from the iris dataset of sklearn library

def load_data():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    data = sklearn.datasets.load_iris()['data']
    return data.T, sklearn.datasets.load_iris()['target']


def SbSw (matrix, label):
    Sb = 0 # initialize the between class cov. matrix
    Sw = 0 # initialize the within class cov. matrix
    mu = vcol(matrix.mean(1)) # dataset mean
    N = matrix.shape[1]
    for i in range(label.max()+1):
        D_c = matrix[:, label == i] # filter the matrix data according to the label (0,1,2)
        nc = D_c.shape[1] # number of sample in class "c"
        mu_c = vcol(D_c.mean(1)) # calc a column vector containing the mean of the attributes (sepal-length, petal-width ...) for one class at a time
        Sb = Sb + nc*np.dot((mu_c - mu),(mu_c - mu).T)
        Sw = Sw + nc*Sw_c(D_c) # calculate the within covariance matrix as a weighted sum of the cov matrix of the classes
    Sb = Sb / N
    Sw = Sw / N
        
    return Sb, Sw


# calculate the covariance matrix for a class "c"
def Sw_c(D_c): 
    Sw_c = 0
    nc = D_c.shape[1] 
    mu_c = vcol(D_c.mean(1)) 
    DC = D_c - mu_c  
    Sw_c = np.dot(DC, DC.T)/nc
    return Sw_c

# GENERALIZED EIGENVALUE PROBLEM
def LDA1(matrix,label,m):
    Sb,Sw = SbSw(matrix,label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m] # reverse the eigenvectors and then retrive the first m
    return W

# EIG. PROB. SOLVED BY JOINT DIAGONALIZATION
def LDA2(matrix, label, m):
    Sb,Sw = SbSw(matrix,label)
    #whiten transformation(see slide)
    U_w, s, _ = np.linalg.svd(Sw)
    P1 = np.dot(U_w * vrow(1.0/(s**0.5)), U_w.T)
    # diagonalization of Sb
    SBTilde = np.dot(P1, np.dot(Sb,P1.T))
    U_b,_,_ = np.linalg.svd(SBTilde)
    P2 = U_b[:, 0:m]
    W = np.dot(P1.T, P2)
    return W
    
def draw_scatter(matrix, label):
    
    mask0 = (label == 0)
    mask1 = (label == 1)
    mask2 = (label == 2)

    D0 = matrix[:, mask0] 
    D1 = matrix[:, mask1] 
    D2 = matrix[:, mask2]
    
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], alpha=0.7, marker="o", label='Setosa')
    plt.scatter(D1[0, :], D1[1, :], alpha=0.7, marker="^", label='Versicolor')
    plt.scatter(D2[0, :], D2[1, :], alpha=0.7, marker="s", label='Virginica')

    plt.legend()
    plt.tight_layout()  
    plt.savefig('LDA_scatter_plot.pdf')
    plt.show()


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data()
    
    W1 = LDA1(D,L,2) # m = 2 because at most n_classes - 1
    y1 = np.dot(W1.T, D)
    W2 = LDA2(D,L,2)
    y2 = np.dot(W2.T, D)
    draw_scatter(y1,L)
    draw_scatter(y2,L)
    print(np.linalg.svd(np.hstack([W1, W2]))[1]) # only m=2 value are different from zero so it means that W1 and W2 are linearly dependent
    