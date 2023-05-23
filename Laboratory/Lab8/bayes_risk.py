import numpy as np

def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = np.zeros((K,K), dtype=int)

    for i,j in zip(Predicted.astype(int),Real.astype(int)):
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
    
 
    
if __name__ == '__main__':
    
    
    llr = np.load("Data/commedia_llr_infpar.npy")
    LTE = np.load("Data/commedia_labels_infpar.npy")
    
    parameter_combinations = [
    (0.5, 1, 1),
    (0.8, 1, 1),
    (0.5, 10, 1),
    (0.8, 1, 10)
]

# calculate DCFu for each combination and store the results in a list of tuples
    result1 = []
    result2 = []
    for pi1, C_fn, C_fp in parameter_combinations:
        predicted_labels = optimalBinaryBayesDecision(llr, pi1, C_fn, C_fp)
        conf_matrix = confusionMatrix(LTE, predicted_labels)
        dcfu = DCF(pi1, C_fn, C_fp, conf_matrix,"un-normalized")
        dcf = DCF(pi1,C_fn,C_fp,conf_matrix,"normalized")
        result1.append((pi1, C_fn, C_fp, round(dcfu, 3)))
        result2.append((pi1, C_fn, C_fp, round(dcf, 3)))
        

# print the results in a table with labels
    print(" (π1,Cfn,Cfp) | DCFu (B) ")
    print(f" {parameter_combinations[0]} | {result1[0][3]} ")
    print(f" {parameter_combinations[1]} | {result1[1][3]} ")
    print(f" {parameter_combinations[2]} | {result1[2][3]} ")
    print(f" {parameter_combinations[3]} | {result1[3][3]} \n")
    
    print(" (π1,Cfn,Cfp) | DCF ")
    print(f" {parameter_combinations[0]} | {result2[0][3]} ")
    print(f" {parameter_combinations[1]} | {result2[1][3]} ")
    print(f" {parameter_combinations[2]} | {result2[2][3]} ")
    print(f" {parameter_combinations[3]} | {result2[3][3]} \n")
    
    