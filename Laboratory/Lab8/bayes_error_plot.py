import numpy as np
import matplotlib.pyplot 


def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = np.zeros((K,K), dtype=int)
    
    for i,j in zip(Predicted,Real):
        confMatrix[i, j] += 1
    
    return confMatrix

def optimalBinaryBayesDecision(llr,pi,Cf_n,Cf_p):
    
    t = -np.log(pi*Cf_n/((1-pi)*Cf_p)) # threshold
    Predicted = (llr > t).astype(int)
    return Predicted

def DCF(pi,C_fn,C_fp,confMatrix,type):
    
    # TN = confMatrix[0, 0]
    # FN = confMatrix[0, 1]
    # FP = confMatrix[1, 0]
    # TP = confMatrix[1, 1]
    
    (TN, FN), (FP, TP) = confMatrix
    
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    
    DCFu = pi*C_fn*FNR + (1-pi)*C_fp*FPR
    
    if type == "un-normalized":
        return DCFu
    elif type == "normalized":
        Bdummy = min(pi * C_fn, (1 - pi) * C_fp)
        DCFn = DCFu/Bdummy
        return DCFn
    else:
        raise ValueError('type must be either "un-normalized" or "normalized"')
    
def min_DCF(pi, C_fn, C_fp,LTE,scores):
    
    t = np.concatenate([scores, [-np.inf, np.inf]])
    t.sort()

    result = []
    for i in range(len(t)):
        
        Predicted = (scores > t[i]).astype(int)
        conf_matrix = confusionMatrix(LTE, Predicted)
        result.append(DCF(pi,C_fn,C_fp,conf_matrix,"normalized"))
        
    return min(result)

if __name__ == '__main__':
    
    
    llr = np.load("Data/commedia_llr_infpar.npy.npy")
    LTE = np.load("Data/commedia_labels_infpar.npy")
    
    
    effPriorLogOdds = np.linspace(-3, 3, 21)
    
    dcf = []
    mindcf = []

    for p in effPriorLogOdds:
        pi = 1/(1+np.exp(-p))
        predicted_labels = optimalBinaryBayesDecision(llr, pi, 1, 1)
        conf_matrix = confusionMatrix(LTE, predicted_labels)
        dcf.append(DCF(pi,1,1,conf_matrix,"normalized"))
        mindcf.append(min_DCF(pi,1,1,LTE,llr))

        
        
    matplotlib.pyplot.plot(effPriorLogOdds, dcf, label="DCF", color="r")
    matplotlib.pyplot.plot(effPriorLogOdds, mindcf, label="min DCF", color="b")
    matplotlib.pyplot.ylim([0, 1.1])
    matplotlib.pyplot.xlim([-3, 3])
    matplotlib.pyplot.xlabel('prior log-odds')
    matplotlib.pyplot.ylabel('DCF value')
    matplotlib.pyplot.title('Bayes error plot')
    matplotlib.pyplot.savefig("Bayes_error_plot.pdf")
    matplotlib.pyplot.legend(loc = "lower left")
    matplotlib.pyplot.show()