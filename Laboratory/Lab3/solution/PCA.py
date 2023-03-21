import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# This function takes an one-dimensional numpy array, 
# converts it to a vertical column form of shape (n,1) where n is the size of the input array  
def vcol(array):
    return array.reshape((array.size, 1))

# This function takes an one-dimensional numpy array,
# converts it to a horizontal row form of shape (1,n) where n is the size of the input array
def vrow(array):
    return array.reshape((1,array.size))



# LOADING DATA #
def load_data(file_name):

    data_list = []
    label_list = []

    # create a dictionary to map the assigned class in the dataset to numerical values
    labels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

# open the file, the "with" sintax also close the file without specifying it
    with open(file_name) as dataset:
        for line in dataset:
            attr = []
            try:
                
                line_list = line.split(",")
                attributes = line_list[0:4]
                name = line_list[-1].strip()
                for i in attributes:
                    
                    attr.append(float(i))
                
                col_array = vcol(np.array(attr))
                data_list.append(col_array)
                
                label_list.append(labels[name])
            except:
                pass

    
    return np.hstack(data_list), np.array(label_list, dtype=np.int32)



# VISUALIZE DATA

def plot_hist(matrix, label):

    
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

    for index in range(4):

        plt.figure(index)
        plt.xlabel(attributes[index])

        plt.hist(D0[index, :], bins=10, density=True,
                 alpha=0.7, label='Setosa')
        plt.hist(D1[index, :], bins=10, density=True,
                 alpha=0.7, label='Versicolor')
        plt.hist(D2[index, :], bins=10, density=True,
                 alpha=0.7, label='Virginica')

        plt.legend()
        plt.tight_layout()  
        plt.savefig('images/histograms/%s.pdf' % attributes[index])

    plt.show()


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





def meanCalc(data_matrix):
   
    mu = 0
    # iterate through each column of the data_matrix
    for i in range(data_matrix.shape[1]):
        # take a slice of the matrix with all rows for the ith column, which is a column vector
        # add the column vector to mu to keep track of the sum of each respective column
        mu = mu + data_matrix[:, i:i+1]
        
    # calculate the mean vector by dividing the sum vector by the number of columns or samples (data_matrix.shape[1])
    mu = mu / float(data_matrix.shape[1]) 
    
    # return the mean vector as output
    return mu


def covarianceCalc(data_matrix,mu):
    C = 0
    for i in range(data_matrix.shape[1]):
        C = C + np.dot(data_matrix[:, i:i+1] - mu, (data_matrix[:, i:i+1] - mu).T)
    C = C / float(data_matrix.shape[1])
    return C





if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data('iris.csv')
    # plot_hist(D, L)
    # plot_scatter(D, L)
    mean = meanCalc(D)
    Cov_matrix = covarianceCalc(D,mean)
    print(mean)
    print(Cov_matrix)