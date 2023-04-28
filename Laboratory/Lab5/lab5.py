#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.datasets
import numpy
import scipy

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

def GAU_pdf(x, mu, var):
    
    return 1/(numpy.sqrt(2*numpy.pi*var))*numpy.exp(-((x-mu)**2)/(2*var))


def GAU_logpdf(x, mu, var):
    
    return -0.5*numpy.log(2*numpy.pi) - 0.5*numpy.log(var) - (x-mu)**2/(2*var)

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)
    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()

def vcol(x):
    return x.reshape(x.shape[0], 1)


# In[2]:


D, L = load_iris()

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)


# In[3]:


def MVG_classifier(DTR, LTR, DTE, LTE):
   DTR_c = [DTR[:, LTR == i] for i in range(3)]
   m_c = []
   s_c = []
   for d in DTR_c:
      m_c.append(d.mean(1))
      s_c.append(numpy.cov(d, bias=True))

   S = numpy.zeros((3, DTE.shape[1]))

   for i in range(3):
      for j, sample in enumerate(DTE.T):
         sample = vcol(sample)
         S[i, j] = numpy.exp(logpdf_GAU_ND(sample, vcol(m_c[i]), s_c[i]))

   SJoint = 1/3*S
   SSum = SJoint.sum(axis=0)
   SPost = SJoint/SSum

   Predictions = SPost.argmax(axis=0) == LTE
   Predicted = Predictions.sum()
   NotPredicted = Predictions.size - Predicted
   acc = Predicted/Predictions.size
   

   return Predicted, DTE.shape[1]


# In[4]:


MVG_classifier(DTR, LTR, DTE, LTE)


# ## Inference Chain in the log-domain

# In[5]:


def MVG_log(DTR, LTR, DTE, LTE):

    DTR_c = [DTR[:, LTR == i] for i in range(3)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(numpy.cov(d, bias=True))

    S = numpy.zeros((3, DTE.shape[1]))

    for i in range(3):
        for j, sample in enumerate(DTE.T):
            sample = vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, vcol(m_c[i]), s_c[i])

    SJoint = numpy.log(1/3) + S
    SSum = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size

    return Predicted, DTE.shape[1]


# In[6]:


MVG_log(DTR, LTR, DTE, LTE)


# ## Naive Bayes Gaussian Classifier

# In[7]:


def NaiveBayesGaussianClassifier(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(3)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(numpy.cov(d, bias=True)*numpy.identity(d.shape[0]))
    
    S = numpy.zeros((3, DTE.shape[1]))

    for i in range(3):
        for j, sample in enumerate(DTE.T):
            sample = vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, vcol(m_c[i]), s_c[i])
    
    SJoint = numpy.log(1/3) + S
    SSum = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size

    return Predicted, DTE.shape[1]


# In[8]:


NaiveBayesGaussianClassifier(DTR, LTR, DTE, LTE)


# ## Tied Covariance Gaussian Classifier

# In[9]:


def TiedCovarianceGaussianClassifier(DTR, LTR, DTE, LTE):
    DTR_c = [DTR[:, LTR == i] for i in range(3)]
    m_c = []
    s_c = []
    for d in DTR_c:
        m_c.append(d.mean(1))
        s_c.append(numpy.cov(d, bias=True))

    Sstar = 0
    for i in range(3):
        Sstar += DTR_c[i].shape[1]*s_c[i]
    Sstar /= DTR.shape[1]

    S = numpy.zeros((3, DTE.shape[1]))

    for i in range(3):
        for j, sample in enumerate(DTE.T):
            sample = vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, vcol(m_c[i]), Sstar)

    SJoint = numpy.log(1/3) + S
    SSum = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    return Predicted, DTE.shape[1]


# In[10]:


TiedCovarianceGaussianClassifier(DTR, LTR, DTE, LTE)


# ## K-fold cross validation

# In[11]:


K = 150
N = int(D.shape[1]/K)
classifiers = [(MVG_log, "Multivariate Gaussian Classifier"), (NaiveBayesGaussianClassifier, "Naive Bayes"), (TiedCovarianceGaussianClassifier, "Tied Covariance")]

for j, (c, cstring) in enumerate(classifiers):
    nWrongPrediction = 0
    numpy.random.seed(j)
    indexes = numpy.random.permutation(D.shape[1])
    for i in range(K):

        idxTest = indexes[i*N:(i+1)*N]

        if i > 0:
            idxTrainLeft = indexes[0:i*N]
        elif (i+1) < K:
            idxTrainRight = indexes[(i+1)*N:]

        if i == 0:
            idxTrain = idxTrainRight
        elif i == K-1:
            idxTrain = idxTrainLeft
        else:
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
        
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        nCorrectPrediction, nSamples = c(DTR, LTR, DTE, LTE)
        nWrongPrediction += nSamples - nCorrectPrediction

    errorRate = nWrongPrediction/D.shape[1]
    accuracy = 1 - errorRate
    print(f"{cstring} results:\nAccuracy: {round(accuracy*100, 1)}%\nError rate: {round(errorRate*100, 1)}%\n") 


# In[ ]:




