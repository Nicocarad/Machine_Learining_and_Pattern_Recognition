import numpy as np



def load_iris_binary():
    
    import sklearn.datasets
    
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
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



class logRegClass():
    
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        
    def logreg_obj(self, v):
        
        w, b = v[0:-1], v[-1] 
        z = 2*self.LTR -1 #remap the label from {0,1} to {-1,1}
        x = self.DTR
        n = self.DTR.shape[1]
        
        normalization = -self.l*0.5*np.linalg.norm(w)
        
        expo = -z * (w.T.dot(x) + b)
        
        loss_funct = np.logaddexp(0,expo)/n
        
        J = normalization + loss_funct
        
        return J
        
    
    
    
if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    



    
