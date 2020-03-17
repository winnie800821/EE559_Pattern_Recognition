import numpy as np
np.set_printoptions(threshold=np.inf)
import math
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from collections import Counter
import random
# Use scikit-learn in Python
#(a)
training = np.loadtxt(open("wine_train.csv"), delimiter=",")
training_label=training[:,len(training[0])-1]
Original_traindata=np.zeros((len(training),len(training[0]-1)))
Original_traindata=training[:,0:len(training[0])-1]
scaler = StandardScaler() #initialize object
scaler.fit(Original_traindata) #get the mean and standard deviation
Normalized_traindata=np.zeros((np.shape(Original_traindata)))
Normalized_traindata=scaler.transform(Original_traindata)

#(b)
print ('The mean of each feature in original training data is ',np.mean(Original_traindata,axis=0))
print ('The standard deviation of each feature in original training data is ',np.std(Original_traindata,axis=0))
print ('The mean of each feature in normalized training data is ',np.mean(Normalized_traindata,axis=0))
print ('The standard deviation of each feature in normalized training data is ',np.std(Normalized_traindata,axis=0))

testing = np.loadtxt(open("wine_test.csv"), delimiter=",")
testing_label=testing[:,len(testing[0])-1]
Original_testdata=np.zeros((len(testing),len(testing[0]-1)))
Original_testdata=testing[:,0:len(testing[0])-1]
Normalized_testdata=np.zeros((np.shape(Original_testdata)))
Normalized_testdata=scaler.transform(Original_testdata)


##Solution for (d)
#############two features accuracy for training data
model_twofeature=Perceptron(tol=1e-3)

twofeature_traindata=np.zeros((len(training),2))
twofeature_traindata=Normalized_traindata[:,0:2]
model_twofeature.fit(twofeature_traindata,training_label) #get the mean and standard deviation
twofeature_predict_label_train=model_twofeature.predict(twofeature_traindata)
twofeature_accuracy_train=accuracy_score(training_label,twofeature_predict_label_train)
print('The accuracy for the first two features in training data is ',twofeature_accuracy_train)
#############two features accuracy for testing data
twofeature_testdata=np.zeros((len(testing),2))
twofeature_testdata=Normalized_testdata[:,0:2]
twofeature_predict_label_test=model_twofeature.predict(twofeature_testdata)
twofeature_accuracy_test=accuracy_score(testing_label,twofeature_predict_label_test)
print ('The accuracy for the first two features in testing data is ',twofeature_accuracy_test)
print('The resulting 3 weight vectors for using the first two features are ',model_twofeature.coef_)


#############13 features accuracy for training data
model_allfeature=Perceptron(tol=1e-3)
model_allfeature.fit(Normalized_traindata,training_label) #get the mean and standard deviation
allfeature_predict_label_train=model_allfeature.predict(Normalized_traindata)
allfeature_accuracy_train=accuracy_score(training_label,allfeature_predict_label_train)
print('The accuracy for all features in training data is ',allfeature_accuracy_train)
#############13 features accuracy for testing data
allfeature_predict_label_test=model_allfeature.predict(Normalized_testdata)
allfeature_accuracy_test=accuracy_score(testing_label,allfeature_predict_label_test)
print ('The accuracy for all features in testing data is ',allfeature_accuracy_test)
print('The resulting 3 weight vectors for using all features are ',model_allfeature.coef_)
#print('Ｗ0 for using all features is ',model_allfeature.intercept_)

##Solution for (e)
##Generate the random initial weight vectors for 100 times
#####two features
for i in range(0,100):
    initial_2feature_weight_train=np.random.randn(3,2) #set initial weight vector
    changeinitial_2features_model=Perceptron(tol=1e-3)
    changeinitial_2features_model.fit(twofeature_traindata,training_label,coef_init=initial_2feature_weight_train) #get the mean and standard deviation
    changeinitial_2feature_label_train=changeinitial_2features_model.predict(twofeature_traindata)
    changeinitial_2feature_accuracy_train=accuracy_score(training_label,changeinitial_2feature_label_train)
    changeinitial_2feature_label_test=changeinitial_2features_model.predict(twofeature_testdata)
    changeinitial_2feature_accuracy_test = accuracy_score(testing_label, changeinitial_2feature_label_test)
    if i==0:
        accuracy_array_2featuretrain=(changeinitial_2feature_accuracy_train)
        finalW_array_2featuretrain=(initial_2feature_weight_train)
        accuracy_array_2featuretest=(changeinitial_2feature_accuracy_test)
    else:
        accuracy_array_2featuretrain=np.r_[accuracy_array_2featuretrain,changeinitial_2feature_accuracy_train]
        finalW_array_2featuretrain=np.r_[finalW_array_2featuretrain,initial_2feature_weight_train]
        accuracy_array_2featuretest=np.r_[accuracy_array_2featuretest,changeinitial_2feature_accuracy_test]

best_accuracy_2featuretrain=max(accuracy_array_2featuretrain)
best_index=[]
random_best_index=[]
for i in range (0,len(accuracy_array_2featuretrain)):  #to check how many times the max accuracy happened in the array
    if accuracy_array_2featuretrain[i]==best_accuracy_2featuretrain:
        best_index=np.r_[best_index,i]
if len(best_index)>=2: #if the max accuracy happened more than once
    random_best_index=random.sample(list(best_index),1)#we need to choose the max accuracy index randomly
    b=int(random_best_index[0])
else:     #if the max accuracy happened only one time
    random_best_index=best_index[0]  #save the index info
    b = int(random_best_index)

index_bestaccuracy_2featuretrain=np.argmax(accuracy_array_2featuretrain)
paired_accuracy_2featuretest=accuracy_array_2featuretest[b]

print ('The best accuracy for two features in training data in 100 times run with different initial w vector is',best_accuracy_2featuretrain,'. \nAnd the paired final vectors are',finalW_array_2featuretrain[b*3:(b+1)*3])
print ('The best accuracy result in training data happened in',b+1 ,'th time and the paired accuracy for two features in testing data is',paired_accuracy_2featuretest)

#####all features
for i in range(0,100):
    initial_allfeature_weight_train=np.random.randn(3,13) #set initial weight vector
    changeinitial_allfeatures_model=Perceptron(tol=1e-3)
    changeinitial_allfeatures_model.fit(Normalized_traindata,training_label,coef_init=initial_allfeature_weight_train) #get the mean and standard deviation
    changeinitial_allfeature_label_train=changeinitial_allfeatures_model.predict(Normalized_traindata)
    changeinitial_allfeature_accuracy_train=accuracy_score(training_label,changeinitial_allfeature_label_train)
    changeinitial_allfeature_label_test=changeinitial_allfeatures_model.predict(Normalized_testdata)
    changeinitial_allfeature_accuracy_test = accuracy_score(testing_label, changeinitial_allfeature_label_test)
    if i==0:
        accuracy_array_allfeaturetrain=(changeinitial_allfeature_accuracy_train)
        finalW_array_allfeaturetrain=(initial_allfeature_weight_train)
        accuracy_array_allfeaturetest=(changeinitial_allfeature_accuracy_test)
    else:
        accuracy_array_allfeaturetrain=np.r_[accuracy_array_allfeaturetrain,changeinitial_allfeature_accuracy_train]
        finalW_array_allfeaturetrain=np.r_[finalW_array_allfeaturetrain,initial_allfeature_weight_train]
        accuracy_array_allfeaturetest=np.r_[accuracy_array_allfeaturetest,changeinitial_allfeature_accuracy_test]

best_accuracy_allfeaturetrain=max(accuracy_array_allfeaturetrain)
best_index=[]
random_best_index=[]
for i in range (0,len(accuracy_array_allfeaturetrain)):  #to check how many times the max accuracy happened in the array
    if accuracy_array_allfeaturetrain[i]==best_accuracy_allfeaturetrain:
        best_index=np.r_[best_index,i]
if len(best_index)>=2: #if the max accuracy happened more than once
    random_best_index=random.sample(list(best_index),1)#we need to choose the max accuracy index randomly
    a = int(random_best_index[0])
else:     #if the max accuracy happened only one time
    random_best_index=best_index[0]  #save the index info
    a = int(random_best_index)

best_accuracy_allfeature_test=accuracy_array_allfeaturetest[a]
print ('The best accuracy for all features in training data in 100 times run with different initial w vector is',best_accuracy_allfeaturetrain,'. \nAnd the paired final vectors are',finalW_array_allfeaturetrain[a*3:(a+1)*3])
print ('The paired accuracy for all features in testing data is',best_accuracy_allfeature_test,'happened in',a+1,'th time run')

#Parts (g)-(j) below use MSE (pseudo-inverse version) classification.
# scikit-learn implementation.

#(g)(g) For this part use unnormalized data.
# Run the pseudoinverse classifier, and report the classification accuracy on the test data, for the first 2 features and for all 13 features.
Original_2feature_traindata=np.zeros((len(training),2))
Original_2feature_traindata=training[:,0:2]

Original_2feature_testdata=np.zeros((len(testing),2))
Original_2feature_testdata=testing[:,0:2]

from sklearn.linear_model import LinearRegression
class MSE_binary(LinearRegression):
    def __init__(self):
        print("Calling_newly_created_MSE_binary_function...")
        super(MSE_binary, self).__init__()
    def predict(self, X):
        thr=0.5 #may vary depending on how you defind b in Xw=b
        y = self._decision_function(X)
        y_binary = np.zeros(y.shape)
        for i in range(y.shape[0]): #the number of samples
            if(y[i] >= thr):
                y_binary[i] = 1
            else:
                y_binary[i] = 0
        return y_binary

from sklearn.multiclass import OneVsRestClassifier
binary_model=MSE_binary()


#report the MSE(pseudo-inverse version) classification accuracy for the first 2 features on the test data
MSEmodel=OneVsRestClassifier(binary_model)
MSEmodel.fit(Original_2feature_traindata,training_label)

MSE_2feature_accuracy_train=MSEmodel.score(Original_2feature_traindata,training_label)
MSE_2feature_accuracy_test=MSEmodel.score(Original_2feature_testdata,testing_label)
print ('The MSE(pseudo-inverse version) classification accuracy for the first 2 features on original training data is',MSE_2feature_accuracy_train)
print ('The MSE(pseudo-inverse version) classification accuracy for the first 2 features on original testing data is',MSE_2feature_accuracy_test)
MSEmodel=OneVsRestClassifier(binary_model)
MSEmodel.fit(Original_traindata,training_label)

MSE_13feature_accuracy_train = MSEmodel.score(Original_traindata, training_label)
MSE_13feature_accuracy_test = MSEmodel.score(Original_testdata, testing_label)
print ('The MSE(pseudo-inverse version) classification accuracy for 13 features on original training data is',MSE_13feature_accuracy_train)
print ('The MSE(pseudo-inverse version) classification accuracy for 13 features on original testing data is',MSE_13feature_accuracy_test)



#(h)using standardized data to repeat (g)
#2 feature

MSEmodel=OneVsRestClassifier(binary_model)
MSEmodel.fit(twofeature_traindata,training_label)

MSE_2feature_accuracy_train=MSEmodel.score(twofeature_traindata,training_label)
MSE_2feature_accuracy_test=MSEmodel.score(twofeature_testdata,testing_label)
print ('The MSE(pseudo-inverse version) classification accuracy for the first 2 features on standardized training data is',MSE_2feature_accuracy_train)
print ('The MSE(pseudo-inverse version) classification accuracy for the first 2 features on standardized testing data is',MSE_2feature_accuracy_test)

MSEmodel=OneVsRestClassifier(binary_model)
MSEmodel.fit(Normalized_traindata,training_label)
MSE_13feature_accuracy_train=MSEmodel.score(Normalized_traindata,training_label)
MSE_13feature_accuracy_test=MSEmodel.score(Normalized_testdata,testing_label)
print ('The MSE(pseudo-inverse version) classification accuracy for 13 features on standardized training data is',MSE_13feature_accuracy_train)
print ('The MSE(pseudo-inverse version) classification accuracy for 13 features on standardized testing data is',MSE_13feature_accuracy_test)