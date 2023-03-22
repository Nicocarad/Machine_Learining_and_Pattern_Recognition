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



def PCA_not_optimized(data_matrix,m):
    
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
    print("This is the Eigenvectors Matrix:\n", U)
    
    # COMPUTE THE P MATRIX
    
    P = U[:, ::-1][:, 0:m]
    
    # PROJECTION OF POINTS
        
    DP = np.dot(P.T, D)
    
    return DP
    
def PCA(data_matrix,m):
    
    N = data_matrix.shape[1] # number of samples
    mu = vcol(data_matrix.mean(1)) # data_matrix.mean(1) return a 1-D array so we must transform it into a column
    DC = data_matrix - mu  # performing broadcast, we are removing from all the data the mean
    C = np.dot(DC, DC.T)/N
    s, U = np.linalg.eigh(C) # compute the eigenvalues and eigenvectors
    P = U[:, ::-1][:, 0:m] # reverse the matrix in order to move leading eigenvectors in the first "m" column
    DP = np.dot(P.T, D) #apply projection of points
    return DP
    
def PCA2(data_matrix,m):
    
    N = data_matrix.shape[1]
    mu = vcol(data_matrix.mean(1))
    DC = data_matrix - mu
    C = np.dot(DC, DC.T)/N
    U,_,_ = np.linalg.svd(C) # compute the eigenvectors by performing the Single Value Decomposition
    P = U[:, 0:m]  # filter all rows and the column from 0 to m-1 ( extract only the first m column)
    DP = np.dot(P.T, D) # projection of data
    return DP  
 
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
    plt.savefig('PCA_scatter_plot.pdf')
    plt.show()
    


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data()
   
   # DP = PCA_not_optimized(D,2)
   # DP = PCA(D,2)
    DP = PCA2(D,2) # reduce the dimension of the dataset to 2 ( currently is 4 because 4 attributes are considered in the iris dataset)
    
    draw_scatter(DP, L)
    
    