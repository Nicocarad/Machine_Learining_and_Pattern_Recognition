import numpy as np
import scipy
import sklearn.datasets
import Library.functions as lib
import Library.GaussClassifier as GAU

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

def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = np.dot(DC, DC.T)/N
    
    return mu, C
    

def GaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = np.exp(lib.logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
    
    #print(S.shape) # check inf score matrix is n_classes*n_test_sample
    
    prior = np.ones(S.shape)/3.0 # create a matrix n_classes*n_test_sample
    # prior = lib.vcol(np.ones(3)/3.0) works too since broadcasting is performed in the following line
    SJoint = S*prior
    SMarginal = lib.vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal
    
    Predicted_labels = np.argmax(SPost,0) # checks value in the column and return the index of the highest ( so the label )
    result = np.array([LTE[i] == Predicted_labels[i] for i in range(len(LTE))]) # create an array of boolean with correct and uncorrect predictions
    
    
    accuracy = 100*(result.sum())/len(LTE) # summing an array of boolean returns the number of true values
    error_rate = 100-accuracy
    print(error_rate)
     
    return 
    
    
def LogGaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []
    

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
    
    prior = np.ones(S.shape)/3.0
    
    logSJoint = S + np.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    
    Predicted_labels = np.argmax(SPost,0) 
    result = np.array([LTE[i] == Predicted_labels[i] for i in range(len(LTE))]) 
    
    
    accuracy = 100*(result.sum())/len(LTE) 
    error_rate = 100-accuracy
    print(error_rate)
    
    return
    

if __name__ == '__main__':
    
    D,L = load_iris()
    (DTR, LTR),(DTE,LTE) = split_db_2to1(D,L)
    LogGaussianClassifier(DTR,LTR,DTE,LTE)
    
    
    