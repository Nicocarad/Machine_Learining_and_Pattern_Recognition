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

def between_class_matrix(matrix, n_classes, L):
    
    # calculate the dataset mean
    mu_dataset = 0
    N = matrix.shape[1] # number of samples
    for i in range(n_classes+1):
        mu_dataset = mu_dataset + matrix[:, L==i].mean(1)
    mu_dataset = mu_dataset/N
    
    
    
    
    return


    
def class_mean(class_matrix):
    
    nc = class_matrix.shape[1] #number of sample of class "i"
    class_mean = class_matrix.mean(1)
    return class_mean



if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data()
   
    