from cmath import exp
import numpy 
import sklearn
import scipy.optimize
import scipy.special
from sklearn.datasets import load_files


def mcol(vec):
    return vec.reshape((vec.shape[0],1))

def mrow(vec):
    return vec.reshape((1,vec.shape[0]))

def logreg_obj_wrap(DTR,LTR,l):
    T = tMatrix(LTR)
    K = LTR.max() + 1 
    D = DTR.shape[0]
    def logreg_obj(V):
        W = numpy.zeros((D,K))
        B = numpy.zeros((K,1))
        for i in range(K):
            W[:,i:i+1] += mcol(V[i*D:i*D+D])
        B = mcol(V[D*K:])
        S = scoreMatrix(DTR,W,B)
        Y_log = yLogMatrix(S)
        return l/2*(W*W).sum() + -1/LTR.shape[0]*(T*Y_log).sum()
    return logreg_obj;

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L                
        

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def scoreMatrix(DTR,W,B):
    return numpy.dot(W.T,DTR) + B

def yLogMatrix(scoreMatrix):
    logsum = scipy.special.logsumexp(scoreMatrix,axis=0)
    return scoreMatrix - logsum

#K = 3
def tMatrix(L):
    T = numpy.zeros((3,L.shape[0]))
    for k in range(3):
        for i in range(L.shape[0]):
            if L[i] == k:
                T[k,i] = 1
    return T


D,L = load_iris()

(DTR,LTR), (DTE,LTE) = split_db_2to1(D,L)


obj = logreg_obj_wrap(DTR,LTR,0.000001)
(x,f,d) = scipy.optimize.fmin_l_bfgs_b(obj,numpy.zeros((DTR.shape[0]*3+3)), approx_grad = True)
W = numpy.zeros((DTR.shape[0],3))
for i in range(3):
    W[:,i:i+1] += mcol(x[i*DTR.shape[0]:i*DTR.shape[0]+DTR.shape[0]])
B = mcol(x[DTR.shape[0]*3:])
S = scoreMatrix(DTE,W,B)


err = 0;
for i in range(LTE.shape[0]):
    idx = numpy.argmax(mrow(S[:,i:i+1]))
    if LTE[i] != idx:
        err+=1;

print(err/LTE.shape[0])
    

