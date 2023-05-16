import numpy as np
import scipy.optimize as opt
import scipy



def vcol(array):
    return array.reshape((array.size, 1))

def load_iris():
    
    import sklearn.datasets
    
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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

def reshapeMat(vet,split):
    
       # Vector dimension
       N = len(vet)
       # Number of column in the matrix
       num_column = N // split
       
       # reshape the vector as a matrix K x num_column
       return vet[:num_column*N].reshape(split, num_column)
    
def acc_err_evaluate(Predicted_labels,Real_labels):
    
        result = np.array(Real_labels == Predicted_labels) # create an array of boolean with correct and uncorrect predictions

        acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
        err = 100-acc
    
        return acc,err
    
def tMatrix(L):
    T = np.zeros((L.max()+1, L.shape[0]))
    for i, k in enumerate(L):
        T[k, i] = 1

    return T   

def yLogMatrix(scoreMatrix):
    logsum = scipy.special.logsumexp(scoreMatrix,axis=0)
    return scoreMatrix - logsum
    
class logRegClass():
    
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        
    def logreg_obj(self, v):
        
    
       K = LTE.max()+1
       T = tMatrix(LTR)
       D = DTR.shape[0]
       W = np.zeros((D,K))
       b = np.zeros((K,1))
       x = self.DTR
       
       
       b = vcol(v[-K:])
       W = reshapeMat((v[0:-K]),DTR.shape[0])
       
       S = np.dot(W.T, x) + b  #matrix of scores 
       logY = yLogMatrix(S)
       normalizer = self.l * 0.5 * (W*W).sum() #calculate the normalization term
       
       
       loss_funct = 1/LTR.shape[0]*(T*logY).sum()
       
       J = normalizer - loss_funct
       
       return J
       
                          

if __name__ == '__main__':


    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    K = LTE.max()+1 # number of classes

    W_opt = np.zeros((DTR.shape[0],K))
    
    x0 = np.zeros(DTR.shape[0]*K + K)

    
    lambdaVector = np.array([1.E-6, 1.E-3, 1.E-1, 1])
     
    
    print('{:<10s}  {:<10s}  {:<10s}'.format('Lambda', 'J(w*,b*)', 'Error rate %'))
    for l in lambdaVector:
        
        logRegObj = logRegClass(DTR,LTR,l)
        
        x,f,_ = opt.fmin_l_bfgs_b(logRegObj.logreg_obj,x0,approx_grad = True)
        
        b_opt = vcol(x[-K:])

        W_opt = reshapeMat(x[0:-K],DTR.shape[0])
        # evaluate the score array using w_opt and b_opt obtained by the training set and apply them on the evaluation/test set
        S = np.dot(W_opt.T,DTE) + b_opt
        
        Predicted_labels = np.argmax(S,0)
        
    
        err = acc_err_evaluate(Predicted_labels,LTE)[1]

        print('{:<10}  {:<10.6f}  {:<10.1f}'.format(l, f, err))
       