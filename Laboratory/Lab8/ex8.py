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

def acc_err_evaluate(Predicted_labels,Real_labels):
    

    result = np.array(Real_labels == Predicted_labels) # create an array of boolean with correct and uncorrect predictions
    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    err = 100-acc
    
    return acc,err

def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = lib.vcol(data_matrix.mean(1)) 
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
    
    #print(S.shape) # checks if score matrix is n_classes*n_test_sample
    
    prior = np.ones(S.shape)/3.0 # create a matrix n_classes*n_test_sample
    # prior = lib.vcol(np.ones(3)/3.0) works too since broadcasting is performed in the following line
    
  
    SJoint = S*prior
    
  
    SMarginal = lib.vrow(SJoint.sum(0))
    
    SPost = SJoint/SMarginal
 
    Predicted_labels = np.argmax(SPost,0) # checks value in the column and return the index of the highest ( so the label )
     
    return Predicted_labels 


def TiedGaussianClassifier(DTR,LTR,DTE,LTE):
    
    # Calculate the Tied Covariance Matrix
    C_star = 0 
    N = DTR.shape[1]
    for i in range(LTR.max()+1): 
        D_c = DTR[:, LTR == i] 
        nc = D_c.shape[1] 
        C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
    C_star = C_star / N
      
    
    # Apply Gaussian Classifier
    S = []
    
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu = mean_and_covariance(D_c)[0]
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C_star)
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
        
    
    prior = np.ones(S.shape)/3.0
    
    
    logSJoint = S + np.log(prior)
    
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))

    
    logSPost = logSJoint - logSMarginal
    
    SPost = np.exp(logSPost)
    
    Predicted_labels = np.argmax(SPost,0) 
    
    return Predicted_labels


def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = np.zeros((K,K), dtype=int)
    
    for i in range (len(Real)):
        real_idx = Real[i]
        pred_idx = Predicted[i]
        confMatrix[pred_idx,real_idx] += 1
        
    
    return confMatrix


            
    




if __name__ == '__main__':
    
    D,L = load_iris()
    #DTR: Training data
    #LTR: Training labels
    #DTE: Evaluation data
    #LTE: Evaluation labels
    (DTR, LTR),(DTE,LTE) = split_db_2to1(D,L)
    
    print("Iris Dataset confusion matrix with Gaussian Classifier: \n")
    print(confusionMatrix(LTE,GaussianClassifier(DTR,LTR,DTE,LTE)))
    print()
    print("Iris Dataset confusion matrix with Tied Gaussian Classifier: \n")
    print(confusionMatrix(LTE,TiedGaussianClassifier(DTR,LTR,DTE,LTE)))
    print()
    SPost = np.load("Data/commedia_ll.npy")
    Predicted_labels = np.argmax(SPost,0)
    Real_labels = np.load("Data/commedia_labels.npy")
    print("Divina Commedia Dataset confusion matrix with Multinomial Classifier: \n")
    print(confusionMatrix(Real_labels,Predicted_labels))