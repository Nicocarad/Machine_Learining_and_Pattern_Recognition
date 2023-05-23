import numpy as np
import matplotlib.pyplot as plt



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



def plot_ROC(scores,LTE):
    
    # Concatenate scores with -inf and +inf
    thresholds = np.concatenate([scores, [-np.inf, np.inf]])
    # Sort the thresholds
    thresholds.sort()

    fpr_list = []
    tpr_list = []

    for t in thresholds:
    # Compute confusion matrix for current threshold
        predicted = (scores > t).astype(int)
        conf_matrix = confusionMatrix(LTE, predicted)

    # TN = confMatrix[0, 0]
    # FN = confMatrix[0, 1]
    # FP = confMatrix[1, 0]
    # TP = confMatrix[1, 1]
    
    # Extract false negative rate (FNR) and false positive rate (FPR)
        (TN, FN), (FP, TP) = conf_matrix
    
        fnr = FN / (FN + TP)
        fpr = FP / (FP + TN)

    # Compute true positive rate (TPR)
        tpr = 1 - fnr

    # Append FPR and TPR to lists
        fpr_list.append(fpr)
        tpr_list.append(tpr)

# Plot ROC curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0,1.0])
    plt.plot(fpr_list, tpr_list)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC.pdf")
    plt.show()
    

if __name__ == '__main__':
    
    
    llr = np.load("Data/commedia_llr_infpar.npy")
    LTE = np.load("Data/commedia_labels_infpar.npy")
    
    plot_ROC(llr,LTE)