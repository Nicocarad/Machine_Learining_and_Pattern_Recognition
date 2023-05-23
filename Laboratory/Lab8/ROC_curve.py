import numpy as np




def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = np.zeros((K,K), dtype=int)

    for i,j in zip(Predicted.astype(int),Real.astype(int)):
        confMatrix[i, j] += 1

    # Print confusion matrix with labels
    print("Confusion Matrix:")
    print("{:>15}{:^10}{:^10}".format("", "Class 0", "Class 1"))
    print("{:<15}{:^10}{:^10}".format("Prediction 0", confMatrix[0,0], confMatrix[0,1]))
    print("{:<15}{:^10}{:^10}".format("Prediction 1", confMatrix[1,0], confMatrix[1,1]))

    return confMatrix

def optimalBinaryBayesDecision(llr,pi,Cf_n,Cf_p):
    
    t = -np.log(pi*Cf_n/((1-pi)*Cf_p)) # threshold
    Predicted = (llr > t).astype(int)
    return Predicted