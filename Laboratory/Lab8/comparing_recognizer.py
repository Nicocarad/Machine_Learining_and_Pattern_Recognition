import numpy as np

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
    
    llr = np.load("Data/commedia_llr_infpar.npy")
    LTE = np.load("Data/commedia_labels_infpar.npy")
    llr_e1 = np.load("Data/commedia_llr_infpar_eps1.npy")
    LTE_e1 = np.load("Data/commedia_labels_infpar_eps1.npy")

    parameter_combinations = [
    (0.5, 1, 1),
    (0.8, 1, 1),
    (0.5, 10, 1),
    (0.8, 1, 10)
]

    print("{:<10}{:<20}{:<20}".format("", "epsilon = 0.001", "epsilon = 1.0"))
    for pi, C_fn, C_fp in parameter_combinations:
        print("{:<15}{:<15}{:<15}{:<15}{:<15}".format("({},{},{})".format(pi, C_fn, C_fp), "", "", "", ""))
        
        # Compute confusion matrix and DCF values for original data
        predicted = optimalBinaryBayesDecision(llr, pi, C_fn, C_fp)
        conf_matrix = confusionMatrix(LTE, predicted)
        dcf_normalized = DCF(pi, C_fn, C_fp, conf_matrix, "normalized")
        min_dcf = min_DCF(pi, C_fn, C_fp, LTE, llr)

        # Compute confusion matrix and DCF values for perturbed data
        predicted_e1 = optimalBinaryBayesDecision(llr_e1, pi, C_fn, C_fp)
        conf_matrix_e1 = confusionMatrix(LTE_e1, predicted_e1)
        dcf_normalized_e1 = DCF(pi, C_fn, C_fp, conf_matrix_e1, "normalized")
        min_dcf_e1 = min_DCF(pi, C_fn, C_fp, LTE_e1, llr_e1)

        # Print the results in three columns
        print("{:<15}{:<15.3f}{:<15.3f}{:<15s}{:<15s}".format("DCF", dcf_normalized, dcf_normalized_e1, "", ""))
        print("{:<15}{:<15.3f}{:<15.3f}{:<15s}{:<15s}".format("min DCF", min_dcf, min_dcf_e1, "", ""))



