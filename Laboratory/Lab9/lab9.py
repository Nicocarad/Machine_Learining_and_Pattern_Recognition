import numpy
import string
import scipy.special
import itertools
import sys
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize

def mRow(v):
    return v.reshape((1,v.size)) #sizes gives the number of elements in the matrix/array
def mCol(v):
    return v.reshape((v.size, 1))

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]# We remove setosa from D
    L = L[L!=0]# We remove setosa from L
    L[L==2] = 0# We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2tol(D,L, seed=0):
    nTrain=int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    idxTrain=idx[0:nTrain]
    idxTest=idx[nTrain:]

    #Training Data
    DTR = D[:, idxTrain]
    #Evaluation Data
    DTE = D[:, idxTest]
    #Training Labels
    LTR = L[idxTrain]
    #Evaluation Labels
    LTE = L[idxTest]

    return [(DTR, LTR), (DTE,LTE)]

def compute_lagrangian_wrapper(H):

    def compute_lagrangian(alpha):

        elle = numpy.ones(alpha.size) # 66,
        L_hat_D=0.5*( numpy.linalg.multi_dot([alpha.T, H, alpha]) ) - numpy.dot(alpha.T , mCol(elle))# 1x1
        L_hat_D_gradient= numpy.dot(H, alpha)-elle # 66x1
        
        return L_hat_D, L_hat_D_gradient.flatten() # 66, 
   
    
    return compute_lagrangian


def compute_accuracy_error(predicted_labels, LTE):
    good_predictions = (predicted_labels == LTE) #array with True when predicted_labels[i] == LTE[i]    
    num_corrected_predictions =(good_predictions==True).sum()
    tot_predictions = predicted_labels.size
    accuracy= num_corrected_predictions /tot_predictions
    error = (tot_predictions - num_corrected_predictions ) /tot_predictions

    return (accuracy, error)
    

def compute_primal_objective(w_hat_star, C, Z, D_hat):

    w_hat_star = mCol(w_hat_star)
    Z = mRow(Z)
    fun1= 0.5 * (w_hat_star*w_hat_star).sum()   
    fun2 = Z* numpy.dot(w_hat_star.T, D_hat)
    fun3 = 1- fun2
    zeros = numpy.zeros(fun3.shape)
    sommatoria = numpy.maximum(zeros, fun3)
    fun4= numpy.sum(sommatoria)
    fun5= C*fun4
    ris = fun1 +fun5
    return ris
    


if __name__ == '__main__':
    D, L = load_iris_binary()
    # D= Data -> matrix 100data*4 attributes ----100-----
    #                                        |          |
    #                                        4          |
    #                                        |----------|
    # L=label-> row of 100 labels (1 per data)  1= iris versicolor, 0= iris virginica
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D, L)
    # samples for trainining and  sample for evaluation  
    # DTR: Training Data
    # DTE: Evaluation Data
    # LTR: Training Labels
    # LTE: Evaluation Labels
    K=1
    k_values= numpy.ones([1,DTR.shape[1]]) *K
    #Creating D_hat= [xi, k] with k=1
    D_hat = numpy.vstack((DTR, k_values))
    #Creating H_hat
    # 1) creating G_hat through numpy.dot and broadcasting
    G_hat= numpy.dot(D_hat.T, D_hat)
    
    # 2)vector of the classes labels (-1/+1)
    Z = numpy.copy(LTR)
    Z[Z == 0] = -1
    Z= mCol(Z)

    
    # 3) multiply G_hat for ZiZj operating broadcasting
    H_hat= Z * Z.T * G_hat

    # Calculate L_hat_D and its gradient DUAL SOLUTION
    compute_lagr= compute_lagrangian_wrapper(H_hat)

    # Use scipy.optimize.fmin_l_bfgs_b
    C=0.1
    x0=numpy.zeros(LTR.size) #alpha
    
    bounds_list = [(0,C)] * LTR.size
    (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)
    
    # From the dual solution obtain the primal one
  
    # EVALUATION!

    sommatoria = mCol(x) * mCol(Z) * D_hat.T
    w_hat_star = numpy.sum( sommatoria,  axis=0 ) 
    w_star = w_hat_star[0:-1] 
    b_star = w_hat_star[-1] 
    
    scores = numpy.dot(mCol(w_star).T, DTE) + b_star
    #Assign a pattern comparing with threshold 0
    predicted_labels = 1*(scores > 0 )
    accuracy, error = compute_accuracy_error(predicted_labels, mRow(LTE))
    
    #Compute the primal objective
    primal_obj= compute_primal_objective(w_hat_star, C, Z, D_hat)
    
    # Compute the duality gap
    dual_obj = f #he one found with scipy.optimize
    
    duality_gap= primal_obj + dual_obj
    print(">>>> LINEAR SVM <<<<")
    print("K:", K)
    print("C:", C)
    print("PRIMAL LOSS: ", primal_obj)
    print("DUAL LOSS: ", -dual_obj)
    print("DUALITY GAP: ", duality_gap)
    print("ERROR RATE: ", error*100, "%")



   
    