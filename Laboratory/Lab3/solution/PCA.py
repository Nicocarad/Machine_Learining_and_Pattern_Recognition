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
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target'] # target is the label


# VISUALIZE DATA


def plot_scatter(matrix, label):

    mask0 = (label == 0)
    mask1 = (label == 1)
    mask2 = (label == 2)

    D0 = matrix[:, mask0] 
    D1 = matrix[:, mask1] 
    D2 = matrix[:, mask2]  

    attributes = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for x in range(4):
        for y in range(4):
            if x == y:
                continue
            plt.figure()
            plt.xlabel(attributes[x])
            plt.ylabel(attributes[y])
            plt.scatter(D0[x, :], D0[y, :], alpha=0.7, marker="o", label='Setosa')
            plt.scatter(D1[x, :], D1[y, :], alpha=0.7, marker="^", label='Versicolor')
            plt.scatter(D2[x, :], D2[y, :], alpha=0.7, marker="s", label='Virginica')

            plt.legend()
            plt.tight_layout()  
            plt.savefig('images/scatterPlots/scatter_%s_%s.pdf' %
                        (attributes[x], attributes[y]))
        plt.show()




def PCA(data_matrix):
    
    # CALCULATE MEAN
    sum = 0
    number_of_column = data_matrix.shape[1] # shape[1] takes the dimension "1" so the columns and calculate the total number 
    for i in range(number_of_column):
        
        sum = sum + data_matrix[:, i:i+1] #sum is a new array containing the same number of rows of data_matrix and only one colum containing the sum of the element of each column of data_matrix
        
    mu = sum / float(number_of_column) # the mean vector contains the mean value for each attribute (sepal length (cm),sepal width (cm), petal length (cm), petal width (cm))
    print("This is the mean vector:\n", mu)
    
    # CALCULATE COVARIANCE MATRIX
    C = 0
    for i in range(number_of_column):
        C = C + np.dot(data_matrix[:, i:i+1] - mu, (data_matrix[:, i:i+1] - mu).T)
    C = C / float(number_of_column)
    print("This is the Covariance matrix:\n", C)
    
    
    # COMPUTE THE EIGENVALUES AND EIGENVECTORS OF THE COVARIANCE MATRIX
    s, U = np.linalg.eigh(C)  # s contains the eigenvalues sorted from the smallest to largest and U contains the corresponding eigenvectors
    print("This is the Eigenvalues Matrix:\n", s)
    print("This is the Covariance matrix:\n", C)
    
    



if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data()
   
    PCA(D)