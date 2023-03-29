import numpy as np
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
    Sw = 0 # initialize the within class cov.matrix
    mu = vcol(matrix.mean(1)) # dataset mean
    nc = D_c.shape[1]
    N = matrix.shape[1]
    for i in range(label.max()+1):
        D_c = matrix[:, label == i] # filter the matrix data according to the label (0,1,2)
        mu_c = vcol(D_c.mean(1)) # calc a column vector containing the mean of the attributes (sepal-length, petal-width ...) for one class at a time
        Sb = Sb + nc*np.dot((mu_c - mu),(mu_c - mu).T)
    Sb = Sb / N
        
    return Sb



if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data()
    S = SbSw(D,L)
    
    