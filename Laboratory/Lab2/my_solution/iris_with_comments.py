import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def row2col(array):
    # takes a array of shape: (1,array.size) and transform it into (array.shape,1)
    return array.reshape((array.size, 1))


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
                # example of line : 6.5,3.0,5.5,1.8,Iris-virginica
                # create a list of attributes separating strings with "," as delimiter
                line_list = line.split(",")
                # create a new list with only the first 4 element of the previous list (aka: filter the class label)
                attributes = line_list[0:4]
                # extract the label from the line (i.e. Iris-setosa) and remove possible leading or trailing spaces
                name = line_list[-1].strip()
                for i in attributes:
                    # cast the string to a float and inserti in the list
                    attr.append(float(i))
                # transofrm a row to column
                col_array = row2col(np.array(attr))
                data_list.append(col_array)
                # insert in label_list the corresponding label of the line according to the dictionary defined above
                label_list.append(labels[name])
            except:
                pass

    # dtype is a optional parameter, used to define the type of the 1-d array
    return np.hstack(data_list), np.array(label_list, dtype=np.int32)
# the function return:
# 1) the concatenation of the element of the list in a horizontal way
# it means that we have a matrix 4x150
# 2) a 1-dimensional array of size 150, containing the assigned label


# VISUALIZE DATA

def plot_hist(matrix, label):

    # mask0 will be the same kind of structure of "label" with value substituted with "True" if the value is = 0 or "False" otherwise
    mask0 = (label == 0)
    mask1 = (label == 1)
    mask2 = (label == 2)

    # D0 is a matrix derived from the whole matrix applying the mask #extract Setosa info
    D0 = matrix[:, mask0]
    D1 = matrix[:, mask1]  # extract Versicolor info
    D2 = matrix[:, mask2]  # extract Virginica info

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
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig('images/histograms/%s.pdf' % attributes[index])

    plt.show()


def plot_scatter(matrix, label):

    mask0 = (label == 0)
    mask1 = (label == 1)
    mask2 = (label == 2)

    D0 = matrix[:, mask0]  # extract Setosa info
    D1 = matrix[:, mask1]  # extract Versicolor info
    D2 = matrix[:, mask2]  # extract Virginica info

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
            plt.scatter(D0[x, :], D0[y, :], marker="o", alpha=0.7, label='Setosa')
            plt.scatter(D1[x, :], D1[y, :], alpha=0.7, marker="^", label='Versicolor')
            plt.scatter(D2[x, :], D2[y, :], alpha=0.7, marker="s", label='Virginica')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig('images/scatterPlots/scatter_%s_%s.pdf' %
                        (attributes[x], attributes[y]))
        plt.show()


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_data('iris.csv')
    plot_hist(D, L)
    plot_scatter(D, L)
