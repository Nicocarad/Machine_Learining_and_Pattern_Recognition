import numpy as np
import scipy.optimize as opt



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

def acc_err_evaluate(Predicted_labels,Real_labels):
    
    result = np.array(Real_labels == Predicted_labels) # create an array of boolean with correct and uncorrect predictions

    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    err = 100-acc
    
    return acc,err

class logRegClass():
    
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        
    def logreg_obj(self, v):
        
        
        # v is an array [DTR.shape[0]+1] 
        # the initial value of v is the starting point "x0" passed by the user to the function "fmin_l_bfgs_b"
        # at each iteration following the first one, v is passed directly by of "fmin_l_bfgs_b"
        # v corresponds to the "evaluated" minimum at each iteration of the gradient descending algorithm
        
        w, b = v[0:-1], v[-1] 

        z = 2*self.LTR -1 #remap the label from {0,1} to {-1,1}
        x = self.DTR
        
        normalizer = self.l * 0.5 * np.linalg.norm(w)**2 #calculate the normalization term
        
        expo = -z * (w.T.dot(x) + b) # expo is a vector containing the result of e^(-z_i(w.T * x_i + b)) for each sample
          
        loss_funct = np.logaddexp(0,expo).mean() #np.logaddexp(0,expo) is a vector containing the result of log(1+expo[i]) for each samples
        # .mean() sums all the logarithms and than divides by the number of elements
        

        return normalizer + loss_funct
        
        
    
    
if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    # x0 is the initial point for the gradient descending algorithm
    # x0 has 4 components because the function is four-dimensional
    # the function is four-dimensional because attributes have 4 attributes
    x0 = np.zeros(DTR.shape[0] + 1)
    
    lambdaVector = np.array([1.E-6, 1.E-3, 1.E-1, 1])
    
    print('{:<10s}  {:<10s}  {:<10s}'.format('Lambda', 'J(w*,b*)', 'Error rate %'))
    print()
    for l in lambdaVector:
        
        logRegObj = logRegClass(DTR,LTR,l)
        
        x,f,_ = opt.fmin_l_bfgs_b(logRegObj.logreg_obj,x0,approx_grad = True,factr=10000000.0, maxfun=20000)
        
        # retreive from the estimated point of the minimum the four informations: 3 info for parameter "w" and one for parameter "b"
        w_opt,b_opt = x[0:-1], x[-1]
    
        # evaluate the score array using w_opt and b_opt obtained by the training set and apply them on the evaluation/test set
        S = np.dot(w_opt.T,DTE) + b_opt
    
        Predicted_Labels = (S > 0).astype(int)
    
        acc,err = acc_err_evaluate(Predicted_Labels,LTE)

        print('{:<10}  {:<10.6f}  {:<10.2f}'.format(l, round(f,6), round(err,1)))
       

    

    


    
