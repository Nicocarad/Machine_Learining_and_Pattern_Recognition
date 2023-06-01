import numpy
import scipy 


def mCol(v):
    return v.reshape((v.size, 1))


def mRow(array):
    return array.reshape((1,array.size))

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]# We remove setosa from D
    L = L[L!=0]# We remove setosa from L
    L[L==2] = 0# We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # nTrain = 100
    numpy.random.seed(seed) # procduce always the same sequence of random values
    idx = numpy.random.permutation(D.shape[1]) #compute a permutetion of index: [0,1,2,...,149] to [114,32,11,...,35] randomly
    idxTrain = idx[0:nTrain] # assign the first 2/3 of the indexes to the training set
    idxTest = idx[nTrain:] # assign the last 1/3 of the indexes to the test set
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def acc_err_evaluate(Predicted_labels,Real_labels):
    
    result = numpy.array(Real_labels == Predicted_labels) # create an array of boolean with correct and uncorrect predictions

    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    err = 100-acc
    
    return acc,err


def compute_lagrangian_wrapper(H_hat):

    def compute_lagrangian(alpha):
        
        ones =  numpy.ones(alpha.size) # 66,
        term1 = numpy.dot(H_hat,alpha)
        term2 = numpy.dot(alpha.T,term1)
        J_hat_D = 0.5*term2 - numpy.dot(alpha.T , mCol(ones))# 1x1
        L_hat_D_gradient = numpy.dot(H_hat, alpha) - ones # 66x1
        
        return J_hat_D, L_hat_D_gradient.flatten() # 66, 
   
    
    return compute_lagrangian




def polynomial_SVM(K,const,deg,C,DTR,LTR,DTE):
    
        
    #Compute H_hat
    Z = numpy.zeros(LTR.shape)
    Z = 2*LTR -1
    
    poly_kern_DTR = (numpy.dot(DTR.T,DTR) + const)** deg + K**2
    H_hat = mCol(Z) * mRow(Z) * poly_kern_DTR
    #Compute Lagrangian
    compute_lagr= compute_lagrangian_wrapper(H_hat)
    
    
    # Use scipy.optimize.fmin_l_bfgs_b
    x0=numpy.zeros(LTR.size) #alpha
    
    bounds_list = [(0,C)] * LTR.size
    (alpha_star,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, bounds=bounds_list, factr=1.0)
    
    poly_kern_DTE = (numpy.dot(DTR.T, DTE) + const)** deg + K ** 2
    scores = numpy.sum(numpy.dot(alpha_star * mRow(Z), poly_kern_DTE), axis=0)
    
    dual_obj = f
    
    print("DUAL LOSS: ", -dual_obj)
    
    
    return scores
    

if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    print(" K = 0.0 | Poly (d=2, c=0)")   
    S = polynomial_SVM(0.0,0,2,1.0,DTR,LTR,DTE)
    Predicted_Labels = (S > 0).astype(int)
    acc, err = acc_err_evaluate(Predicted_Labels, LTE)
        
    print("ERROR RATE: ", round(err, 1), "%\n")
    
    
    print(" K = 1.0 | Poly (d=2, c=0)")   
    S = polynomial_SVM(1.0,0,2,1.0,DTR,LTR,DTE)
    Predicted_Labels = (S > 0).astype(int)
    acc, err = acc_err_evaluate(Predicted_Labels, LTE)
        
    print("ERROR RATE: ", round(err, 1), "%\n")
    
    print(" K = 0.0 | Poly (d=2, c=1)")   
    S = polynomial_SVM(0.0,1,2,1.0,DTR,LTR,DTE)
    Predicted_Labels = (S > 0).astype(int)
    acc, err = acc_err_evaluate(Predicted_Labels, LTE)
        
    print("ERROR RATE: ", round(err, 1), "%\n")
    
    print(" K = 1.0 | Poly (d=2, c=1)")   
    S = polynomial_SVM(1.0,1,2,1.0,DTR,LTR,DTE)
    Predicted_Labels = (S > 0).astype(int)
    acc, err = acc_err_evaluate(Predicted_Labels, LTE)
        
    print("ERROR RATE: ", round(err, 1), "%\n")