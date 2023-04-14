import numpy as np
import sklearn.datasets

def vcol(array):
    return array.reshape((array.size, 1))


# load iris dataset
def load_iris():
    
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

# divide the dataset in a Training set and Validation Set
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # nTrain = 100
    np.random.seed(seed) # procduce always the same sequence of random values
    idx = np.random.permutation(D.shape[1]) #compute a permutetion of index: [0,1,2,...,149] to [114,32,11,...,35] randomly
    idxTrain = idx[0:nTrain] # assign the first 2/3 of the indexes to the training set
    idxTest = idx[nTrain:] # assign the last 1/3 of the indexes to the test set
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def mean_and_covariance(D,L):
    
    for i in range(L.max()+1):
        D_c = D[:,L == i] 
        N_c = D_c.shape[1]
        mu_c = vcol(D_c.mean(1))
        DC = D_c - mu_c 
        C_c = np.dot(DC, DC.T)/N_c
        print("Mean class:", i)
        print(mu_c) 
        print("Covariance matrix class:", i)
        print(C_c)
        
    
    return 
    
    


if __name__ == '__main__':
    
    D,L = load_iris()
    (DTR, LTR),(DTE,LTE) = split_db_2to1(D,L)
    mean_and_covariance(DTR,LTR)
    
    