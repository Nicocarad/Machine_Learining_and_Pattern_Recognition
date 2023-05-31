import numpy
import scipy 


def mCol(v):
    return v.reshape((v.size, 1))

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

        ones = numpy.ones(alpha.size) # 66,
        J_hat_D = 0.5*( numpy.linalg.multi_dot([alpha.T, H_hat, alpha]) ) - numpy.dot(alpha.T , mCol(ones))# 1x1
        L_hat_D_gradient= numpy.dot(H_hat, alpha) - ones # 66x1
        
        return J_hat_D, L_hat_D_gradient.flatten() # 66, 
   
    
    return compute_lagrangian



def compute_duality_gap(w_hat_star, C, Z, D_hat, dual_obj):

    term1 = 0.5 * numpy.sum(w_hat_star ** 2)   
    term2 = 1 - ( Z.T * numpy.dot(w_hat_star.T, D_hat) )
    term3= C * numpy.sum(numpy.maximum(0, term2))
    primal_obj = term1 + term3
    
    duality_gap = primal_obj + dual_obj
    
    return primal_obj,numpy.abs(duality_gap)

def linear_SVM(K,C,DTR,LTR):
    
    #Creating D_hat= [xi, k] with k=1
    ones_array = numpy.ones((K, DTR.shape[1]))
    D_hat = numpy.concatenate((DTR, ones_array), axis=0)
    
    #Compute H_hat
    G_hat = numpy.dot(D_hat.T,D_hat)
    Z = 2*LTR -1
    Z = mCol(Z)
    H_hat= Z * Z.T * G_hat # For each row i and column j of H_hat, take the dot product of the ith row of Z with the jth column of Z.T. This gives a scalar value.

    #Compute Lagrangian
    compute_lagr= compute_lagrangian_wrapper(H_hat)
    
    
    # Use scipy.optimize.fmin_l_bfgs_b
    x0=numpy.zeros(LTR.size) #alpha
    
    bounds_list = [(0,C)] * LTR.size
    (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, bounds=bounds_list, factr=1.0)
    

    # From Dual solution to Primal solution
    w_hat_star = numpy.sum( mCol(x) * mCol(Z) * D_hat.T,  axis=0 )

    
    # Extract terms and compute scores
    w_star = w_hat_star[0:-1] 
    b_star = w_hat_star[-1] 
    
    scores = numpy.dot(mCol(w_star).T, DTE) + b_star
    
    
    primal_obj,duality_gap = compute_duality_gap(w_hat_star, C, Z, D_hat,f)
    dual_obj = f
    
   
    print("PRIMAL LOSS: ", primal_obj)
    print("DUAL LOSS: ", -dual_obj)
    print("DUALITY GAP: ", duality_gap)
    
    return scores
    

if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    K_values = [1, 1, 1, 10]
    C_values = [0.1, 1.0, 10.0, 0.1]
    
    for i in range(len(K_values)):
        K = K_values[i]
        C = C_values[i]
        print(">>>> TEST {} <<<<".format(i+1))
        print("K:", K)
        print("C:", C)
        
        S = linear_SVM(K, C, DTR, LTR)
        Predicted_Labels = (S > 0).astype(int)
        acc, err = acc_err_evaluate(Predicted_Labels, LTE)
        
        print("ERROR RATE: ", round(err, 1), "%")
        
        
    
     

    
