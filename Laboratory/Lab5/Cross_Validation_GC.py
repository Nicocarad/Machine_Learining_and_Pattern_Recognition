import numpy as np
import scipy
import sklearn.datasets
import Library.functions as lib


# load iris dataset
def load_iris():
    
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = lib.vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = np.dot(DC, DC.T)/N
    
    return mu, C

def evaluate_accuracy(Posterior_prob,Real_labels):
    
    Predicted_labels = np.argmax(Posterior_prob,0)
    result = np.array([Real_labels[i] == Predicted_labels[i] for i in range(len(Real_labels))]) # create an array of boolean with correct and uncorrect predictions

    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    
    return acc



def GaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = np.exp(lib.logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(lib.vrow(f_conditional))
    S = np.vstack(S)
    

    prior = np.ones(S.shape)/3.0  
    SJoint = S*prior
    SMarginal = lib.vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal
     
    return SPost
   
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
    
    return SPost
       
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
    
   
    logSJoint = S + np.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    
    return SPost

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
    
    return SPost
    
def Tied_NaiveBayes_GaussianClassifier(DTR,LTR,DTE,LTE):
    
    # Calculate the Tied Covariance Matrix
    C_star = 0 
    N = DTR.shape[1]
    for i in range(LTR.max()+1): 
        D_c = DTR[:, LTR == i] 
        nc = D_c.shape[1] 
        C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
    C_star = C_star / N
    
    # Diagonalize the covariance matrix
    identity = np.identity(C_star.shape[0])
    C_star = C_star*identity
    
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
    
    
    return  SPost


def kfold(model,k,D,L):
    

    SPost_partial = []
    folds = []
    
    
     # Create a list with indices of the Label vector in a random order
    np.random.seed(0)
    idx = np.random.permutation(D.shape[1])
    
    Label = L[idx] # randomize the vector of Real_labels in the same way
    
    #idx = np.arange(150)
    print(idx)
    
    fold_size = D.shape[1] // k

    # Divide indices in k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])
    

    # If the nuber of samples is not divisible by K, add the leavings samples in the last fold
    if D.shape[1] % k != 0:
        folds[-1] = np.concatenate((folds[-1], idx[k * fold_size:]))
        
    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as validation fold
        validation_indices = folds[i]
        DTE = D[:,validation_indices] 
        LTE = L[validation_indices]
        # Use the leaving folds as Training Set
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        DTR = D[:,train_indices] 
        LTR = L[train_indices]
        # Append in the list the Scores (posterior probabilities) for the samples of the training fold 
        Spost = model(DTR, LTR, DTE, LTE)
        SPost_partial.append(Spost)
        
          
    SPost = np.hstack(SPost_partial) 
    acc = evaluate_accuracy(SPost,Label)  
    
    return acc


if __name__ == '__main__':
    
    D,L = load_iris()
    
    #DTR: Training data
    #LTR: Training labels
    #DTE: Evaluation data
    #LTE: Evaluation labels
    print("GaussianClassifier Error Rate:", round(100-kfold(GaussianClassifier,3,D,L),1), "% \n")
    print("LogGaussianClassifier Error Rate:", round(100-kfold(LogGaussianClassifier,3,D,L),1), "% \n")
    print("NaiveBayes_GaussianClassifier Error Rate:", round(100-kfold(NaiveBayes_GaussianClassifier,3,D,L),1), "% \n")
    print("TiedGaussianClassifier Error Rate:", 100-kfold(TiedGaussianClassifier,3,D,L), "% \n")
    print("Tied_NaiveBayes_GaussianClassifier Error Rate:", 100-kfold(Tied_NaiveBayes_GaussianClassifier,3,D,L), "% \n")
    #J = np.load("utils/LOO_logSJoint_TiedNaiveBayes.npy")
    
    
    
    
    
    
    
    
    
    