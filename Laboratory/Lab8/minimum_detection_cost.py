import numpy as np

def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = np.zeros((K,K), dtype=int)
    
    for i,j in zip(Predicted,Real):
        confMatrix[i, j] += 1
    
    return confMatrix


def DCF(pi,C_fn,C_fp,confMatrix,type):
    
    
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
    
    parameter_combinations = [
    (0.5, 1, 1),
    (0.8, 1, 1),
    (0.5, 10, 1),
    (0.8, 1, 10)
]

    result = []
    for pi1, C_fn, C_fp in parameter_combinations:
        min_dcf = min_DCF(pi1,C_fn,C_fp,LTE,llr)
        result.append((pi1, C_fn, C_fp, round(min_dcf, 3)))
        

    print(" (π1,Cfn,Cfp) | min DCF ")
    print(f" {parameter_combinations[0]} | {result[0][3]} ")
    print(f" {parameter_combinations[1]} | {result[1][3]} ")
    print(f" {parameter_combinations[2]} | {result[2][3]} ")
    print(f" {parameter_combinations[3]} | {result[3][3]} \n")