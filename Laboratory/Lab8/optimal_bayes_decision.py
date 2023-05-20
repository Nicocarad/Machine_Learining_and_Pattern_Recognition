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



if __name__ == '__main__':
    llr = np.load("Data/commedia_llr_infpar.npy")
    LTE = np.load("Data/commedia_labels_infpar.npy")

    # Test 1
    Predicted_label = optimalBinaryBayesDecision(llr,0.5,1,1)
    print("Test 1 - pi=0.5, Cf_n=1, Cf_p=1")
    confusionMatrix(LTE,Predicted_label)

    # Test 2
    Predicted_label = optimalBinaryBayesDecision(llr,0.8,1,1)
    print("\nTest 2 - pi=0.8, Cf_n=1, Cf_p=1")
    confusionMatrix(LTE,Predicted_label)

    # Test 3
    Predicted_label = optimalBinaryBayesDecision(llr,0.5,10,1)
    print("\nTest 3 - pi=0.5, Cf_n=10, Cf_p=1")
    confusionMatrix(LTE,Predicted_label)

    # Test 4
    Predicted_label = optimalBinaryBayesDecision(llr,0.8,1,10)
    print("\nTest 4 - pi=0.8, Cf_n=1, Cf_p=10")
    confusionMatrix(LTE,Predicted_label)
    