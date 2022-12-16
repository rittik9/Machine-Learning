STUDENT_NAME = 'Rittik Panda' #Put your name
STUDENT_ROLLNO = 'MT2022090' #Put your roll number
CODE_COMPLETE = True 
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
#@PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS 
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this

## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
#@PROTECTED_1_END

X_test1=X_test

X_train=X_train/255
X_test1=X_test1/255

X_train,X_test,y_train,y_test = model_selection.train_test_split(X_train,y_train,test_size=0.2,random_state=55 )

def ReLU(sign):
    return np.maximum(sign, 0)

def softmax(data):
    e=np.exp(data-np.max(data))
    return e/e.sum(axis=0)

def ReLU_deriv(Z):
    return Z > 0

def predict(A2):
    return np.argmax(A2, 0)

def accuracy(pred, Y):
    print(pred, Y)
    return np.sum(pred == Y) / Y.size


def model_train(X, Y, learning_rate, epochs,W_l1,b_l1,W_l2,b_l2):
    
    for i in range(epochs):
        
        #forward propagation
        sign1 = W_l1.dot(X) + b_l1
        A1 = ReLU(sign1)
        sign2 = W_l2.dot(A1) + b_l2
        A2 = softmax(sign2)
    
        #backward propagation
        size=X.shape[0]
        encoded_Y = np.zeros((Y.size, Y.max() + 1))
        encoded_Y[np.arange(Y.size), Y] = 1
        encoded_Y = encoded_Y.T
        Z2_der = A2 - encoded_Y
        W2_der = 1 / size * Z2_der.dot(A1.T)
        b2_der = 1 / size * np.sum(Z2_der)
        Z1_der = W_l2.T.dot(Z2_der) * ReLU_deriv(sign1)
        W1_der = 1 / size * Z1_der.dot(X.T)
        b1_der = 1 / size * np.sum(Z1_der)
        
        #update parameters
        W_l1 = W_l1 - learning_rate * W1_der
        b_l1 = b_l1 - learning_rate * b1_der    
        W_l2 = W_l2 - learning_rate * W2_der  
        b_l2 = b_l2 - learning_rate * b2_der 
        
        
        
        #checking accuracy
        if i % 10 == 0:
            print("Epoch ", i)
            pred = predict(A2)
            print(accuracy(pred, Y))
            
    return W_l1, b_l1, W_l2, b_l2

#weight initialization
W_l1 = np.random.rand(102, 784) - 0.5
b_l1 = np.random.rand(102, 1) - 0.5
W_l2 = np.random.rand(10, 102) - 0.5
b_l2 = np.random.rand(10, 1) - 0.5


W_l1, b_l1, W_l2, b_l2 = model_train(X_train.T, y_train.T, 0.02,1500,W_l1, b_l1, W_l2, b_l2)

#check accuracy
sign1 = W_l1.dot(X_test.T) + b_l1
A1 = ReLU(sign1)
sign2 = W_l2.dot(A1) + b_l2
A2 = softmax(sign2)

pred = predict(A2)
c=accuracy(pred, y_test)
print(c)

sign1 = W_l1.dot(X_test1.T) + b_l1
A1 = ReLU(sign1)
sign2 = W_l2.dot(A1) + b_l2
A2 = softmax(sign2)
submissions_df = predict(A2)
submissions_df = pd.DataFrame(submissions_df, columns =['label'])

#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END