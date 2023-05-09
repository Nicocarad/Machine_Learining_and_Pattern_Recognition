import scipy.optimize as spo
import sklearn.datasets
import numpy as np

# def f(a):
#     y, z = a
#     return (y + 3)**2 + npy.sin(y) + (z + 1)**2


# for val in spo.fmin_l_bfgs_b(func=f, x0=npy.array([0,0]), approx_grad=True):
#     print(val)

class logRegClass():

    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l

    def logRegObj(self, v):
        # Function 2 implementation
        w, b = v[0:-1], v[-1]

        # Remap
        z = 2 * self.LTR - 1
        expo = -z * (w.T.dot(self.DTR) + b)
        normalizer = self.l * (w * w).sum() / 2

        return normalizer + np.log1p(np.exp(expo)).mean()
    

    def setLambda(self, lamb):
        self.l = lamb


def load_iris_binary():
    D = sklearn.datasets.load_iris()['data'].T
    L = sklearn.datasets.load_iris()['target']

    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)

    print(D.shape)

    return (D, L)



def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


if __name__ == "__main__":

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    logReg = logRegClass(DTR, LTR, 1.E-3)

    lambdaVector = np.array([0, 1.E-6, 1.E-3, 1])
    
    for l in lambdaVector:

        logReg.setLambda(l)

        x, f, d = spo.fmin_l_bfgs_b(func=logReg.logRegObj, x0=np.zeros(DTR.shape[0] + 1), approx_grad=True, factr=5000, maxfun=20000)
        
        S = (np.dot(x[0:-1].T, DTE) + x[-1])

        LP = (S > 0).astype(int) 

        accuracy =  (LTE == LP).mean()
        errorRate = (1 - accuracy) *100

        print(f"Function calls {d['funcalls']}")
        print(f"{l} | %.8f | %.2f" % (f, errorRate))
