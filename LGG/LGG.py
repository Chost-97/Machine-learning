import pandas as pd
import numpy as np
import math
import statistics as st
import sklearn.metrics
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
import sklearn.model_selection
from warnings import simplefilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import cluster



# future warnings ignoring from sklearn
simplefilter(action='ignore', category=FutureWarning)

# reading the dataset from spam.data 
x=pd.read_csv("spambase.data",names=range(57),dtype=np.float)
y=pd.read_csv("spambase.data",names=[57])

# dropping null values from dataset
x.dropna()
y.dropna()

# converting to numpy arrays for further operations
X=x.values.astype(float)
Y=y.values


# Setting number of bins using sturgs rule 
s=1+(math.log(len(X),2))
print("\nNumber of bins:",round(s))

# Discetizer object creation 
d = KBinsDiscretizer(n_bins=round(s), encode='ordinal', strategy='uniform')


# Hypothesis space creation
H=[]
for i in range(57):
    H.append([])


''' Algorithm 4.1 implementation with initializating the Hypothesis space with the first instance from dataset 
    and then implementing algorithm 4.3 in the LGG for finding the conjuntions of internal disjunction of the literals and returning 
    the hypothesis space with the conjugates '''
def LGG(H,X):
    #Initilization of LGG with first instance from dataset
    for j in range(len(X[0])):
        H[j].append(X[0][j])
    while(j<len(X)):
        H=LGG_conj_ID(H,X[j])
        j+=1
    return(H)

''' Algorithm 4.2 implementation '''
def LGG_conj(H,X):
    for i in range(len(H)):
        if(H[i]!=[]):
            if(X[i] not in H[i]):
                H[i]=[]
    return(H)

''' Algorithm 4.3 implementation '''
def LGG_conj_ID(H,X):
    for i in range(len(H)):
            if(X[i] not in H[i]):
                H[i].append(X[i])
    return(H)


# Prediction list creation
Ypre=[]

# testing of data
def test(X,y):
    for i in range(len(X)):
        count=0
        for j in range(len(X[i])):
            # test for hypothesis space
            if(X[i][j] in H[j]):
                count+=1
        if(count==len(X[i])):
            Ypre.append(1)
        else: Ypre.append(0)

# training of dataset
def train(X,y):
    T=[]
    for i in range(len(X)):
            if(y[i]==1):
                T.append(X[i])
    # creation of LGG
    return(LGG(H,T)) 

# Hypothesis space caliculation
def Hypothesis(H):
    m=1
    for i in range(len(H)):
        m*=len(H[i])
    print("\nThe size of hypothesis space:",m)


# Statistics(Accuracy and F-measures) defenition
def stat(x):
    acc=0
    total=0
    for i in range(2):
        acc+=x[i][i]
        for j in range(2):
            total+=x[i][j]
    acc=acc/total
    print("\nContingency Matrix:\n", x)
    f_measure=((2*x[0][0])/((2*x[0][0])+x[1][0]+x[0][1]))
    print("\nAccuracy: ",acc)
    print("\nF-measure",f_measure,"\n")



# spliting of dataset into train and test datasets
X_train1, X_test1, y_train, y_test = train_test_split(X, Y, test_size=0.2 , random_state=10)

# Printing number of spam mails in the created train dataset
count=0
for i in range(len(y_train)):
    if(y_train[i]==1):
        count+=1
print("\nThe percentage of the spam mails in the training dataset:",round((count/1813)*100))
#Standardization of the train data
X_train2=preprocessing.scale(X_train1)

# Discretization of train data using Kbinsdiscretizer for descretization of data using discretizer
d.fit(X_train2)
X_train=d.transform(X_train2)

#Standardization of the test data
X_test2=preprocessing.scale(X_test1)
# Discretization of test data using Kbinsdiscretizer for descretization of data using discretizer
d.fit(X_test2)
X_test=d.transform(X_test2)



# training data for training dataset and creating Hypothesis space
H=train(X_train,y_train)

# testing data for testing dataset
test(X_test,y_test)
print("\nHYPOTHESIS SPACE GENERATED\n",H,"\n")

#Predicted Hypothesis space caliculation
#print("\n Predicted Hypotheis space : ",pow(2,pow(10,57)),"\n") # too big to print leading to memeory error

#Generated Hypothesis space
Hypothesis(H)

# Creation of contingency matrix
sp=cluster.contingency_matrix(y_test, Ypre, eps=None, sparse=False)


# accuracy and F-measure presentation using contingency matrix
stat(sp)
