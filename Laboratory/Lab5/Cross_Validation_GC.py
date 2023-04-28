import numpy as np
import scipy
import sklearn.datasets
import Library.functions as lib


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
    mu = lib.vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = np.dot(DC, DC.T)/N
    
    return mu, C

def LogGaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []
    

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
    
    prior = np.ones(S.shape)/3.0
    
    print("CHECKS: \n")
    
    logSJoint = S + np.log(prior)
    logJoint = np.load("utils\logSJoint_MVG.npy")
    print((logSJoint-logJoint).max())
    
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logMarginal = np.load("utils\logMarginal_MVG.npy")
    print((logSMarginal-logMarginal).max())
    
    logSPost = logSJoint - logSMarginal
    logPosterior = np.load("utils\logPosterior_MVG.npy")
    print((logSPost - logPosterior).max())
    
    SPost = np.exp(logSPost)
    Posterior = np.load("utils\Posterior_MVG.npy")
    print((SPost-Posterior).max())
    
    
    Predicted_labels = np.argmax(SPost,0) 
    error_rate = lib.acc_err_evaluate(Predicted_labels,LTE)[1]
    print("Error rate (LogGaussianClassifier): ",error_rate)
    print("\n")
    
    return
    
    
def NaiveBayes_GaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []
    

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        identity = np.identity(C.shape[0])
        C = C*identity
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
    
    prior = np.ones(S.shape)/3.0
    
    print("CHECKS: \n")
    
    logSJoint = S + np.log(prior)
    logJoint = np.load("utils\logSJoint_NaiveBayes.npy")
    print((logSJoint-logJoint).max())
    
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logMarginal = np.load("utils\logMarginal_NaiveBayes.npy")
    print((logSMarginal-logMarginal).max())
    
    logSPost = logSJoint - logSMarginal
    logPosterior = np.load("utils\logPosterior_NaiveBayes.npy")
    print((logSPost - logPosterior).max())
    
    SPost = np.exp(logSPost)
    Posterior = np.load("utils\Posterior_NaiveBayes.npy")
    print((SPost-Posterior).max())
    
    
    Predicted_labels = np.argmax(SPost,0) 
    error_rate = lib.acc_err_evaluate(Predicted_labels,LTE)[1]
    print("Error rate (Naive Bayes GaussianClassifier): ",error_rate)
    print("\n")
    
    return



if __name__ == '__main__':
    
    D,L = load_iris()
    #DTR: Training data
    #LTR: Training labels
    #DTE: Evaluation data
    #LTE: Evaluation labels
    (DTR, LTR),(DTE,LTE) = split_db_2to1(D,L)
    