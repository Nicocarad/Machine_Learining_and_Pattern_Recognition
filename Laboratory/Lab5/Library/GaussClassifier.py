import numpy as np
import scipy
import Library.functions as funct


def LogGaussianClassifier(DTR,LTR,DTE,LTE):
    
    S = []
    

    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = funct.mean_and_covariance(D_c)
        f_conditional = funct.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(funct.vrow(f_conditional))
    S = np.vstack(S)
    
    prior = np.ones(S.shape)/3.0
    
    logSJoint = S + np.log(prior)
    logSMarginal = funct.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    
    Predicted_labels = np.argmax(SPost,0) 
    result = np.array([LTE[i] == Predicted_labels[i] for i in range(len(LTE))]) 
    
    
    accuracy = 100*(result.sum())/len(LTE) 
    error_rate = 100-accuracy
    print(error_rate)
    
    return


