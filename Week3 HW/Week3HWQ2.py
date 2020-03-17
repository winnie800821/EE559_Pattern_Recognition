#answer for Week3 HW question2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import plotDecBoundaries as pDB
import  plotDecBoundaries_revised as pDBR


training = np.loadtxt(open("wine_train.csv"), delimiter=",")

label_train = training[:,13]
#put class 1 into class1 ; put class2 and class3 into class2
label_train_class1=np.zeros(len(label_train))
label_train_class1[training[:,13]!=1]=2
label_train_class1[training[:,13]==1]=1
#put class 2 into class1 ; put class1 and class3 into class2
label_train_class2=np.zeros(len(label_train))
label_train_class2[training[:,13]!=2]=2
label_train_class2[training[:,13]==2]=1
#put class 3 into class1 ; put class1 and class3 into class2
label_train_class3=np.zeros(len(label_train))
label_train_class3[training[:,13]!=3]=2
label_train_class3[training[:,13]==3]=1

X1total=np.sum(training[label_train == 1,0])
Y1total=np.sum(training[label_train == 1,1])
X2total=np.sum(training[label_train == 2,0])
Y2total=np.sum(training[label_train == 2,1])
X3total=np.sum(training[label_train == 3,0])
Y3total=np.sum(training[label_train == 3,1])
Numberclass1=np.count_nonzero(label_train == 1)
Numberclass2=np.count_nonzero(label_train == 2)
Numberclass3=np.count_nonzero(label_train == 3)

####calculate mean of x1 and not x1(x2+x3)
X1mean = X1total/Numberclass1
Y1mean = Y1total/Numberclass1

notX1mean = (X2total+X3total) /(Numberclass2+Numberclass3)
notY1mean = (Y2total+Y3total) /(Numberclass2+Numberclass3)

sample_mean1=([X1mean,Y1mean],[notX1mean,notY1mean])

print('Sample mean of [Xclass1,Yclass1],[Xclass2&3,Yclass2&3] is ',(sample_mean1))

####plot x1 and not x1
pDB.plotDecBoundaries(training, label_train_class1, sample_mean1)
####calculate mean of x2 and not x2(x1+x3)
X2mean = X2total/Numberclass2
Y2mean = Y2total/Numberclass2

notX2mean = (X1total+X3total) /(Numberclass1+Numberclass3)
notY2mean = (Y1total+Y3total) /(Numberclass1+Numberclass3)

sample_mean2=([X2mean,Y2mean],[notX2mean,notY2mean])
print('Sample mean of [Xclass2,Yclass2],[Xclass1&3,Yclass1&3] is ',(sample_mean2))

####plot x2 and not x2
pDB.plotDecBoundaries(training, label_train_class2, sample_mean2)
####calculate mean of x3 and not x3(x2+x3)
X3mean = X3total/Numberclass3
Y3mean = Y3total/Numberclass3

notX3mean = (X2total+X1total) /(Numberclass2+Numberclass1)
notY3mean = (Y2total+Y1total) /(Numberclass2+Numberclass1)
sample_mean3=([X3mean, Y3mean],[notX3mean, notY3mean])
print('Sample mean of [Xclass3,Yclass3],[Xclass1&2,Yclass1&2] is ',(sample_mean3))

####plot x3 and not x3
pDB.plotDecBoundaries(training, label_train_class3, sample_mean3)


#############################training data
trainaccuracy=0
for x in range(len(label_train)):
    training_d1=math.sqrt((training[x,0]-X1mean)**2+(training[x,1]-Y1mean)**2)
    training_d23=math.sqrt((training[x,0]-notX1mean)**2+(training[x,1]-notY1mean)**2)
    training_d2=math.sqrt((training[x,0]-X2mean)**2+(training[x,1]-Y2mean)**2)
    training_d13=math.sqrt((training[x,0]-notX2mean)**2+(training[x,1]-notY2mean)**2)
    training_d3=math.sqrt((training[x,0]-X3mean)**2+(training[x,1]-Y3mean)**2)
    training_d12=math.sqrt((training[x,0]-notX3mean)**2+(training[x,1]-notY3mean)**2)
    if label_train[x]==1:
        if training_d1<=training_d23 and training_d13<=training_d2 and training_d12<=training_d3:
            trainaccuracy=trainaccuracy+1
    elif label_train[x]==2:
        if training_d2<=training_d13 and training_d12<=training_d3 and training_d23<=training_d1:
            trainaccuracy = trainaccuracy + 1
    elif label_train[x]==3:
        if training_d3<=training_d12 and training_d13<=training_d2 and training_d23<=training_d1:
            trainaccuracy=trainaccuracy+1

training_accuracyrate=trainaccuracy/(len(label_train))
print('The classification of ',(trainaccuracy),'points in training data are correct. Classification accuracy of training set is :',(training_accuracyrate)) #The accuracy of training data

###############testing
testing = np.loadtxt(open("wine_test.csv"), delimiter=",")
label_test=testing [:,13]
testaccuracy=0
for x in range(len(label_test)):
    testing_d1=math.sqrt((testing[x,0]-X1mean)**2+(testing[x,1]-Y1mean)**2)
    testing_d23=math.sqrt((testing[x,0]-notX1mean)**2+(testing[x,1]-notY1mean)**2)
    testing_d2=math.sqrt((testing[x,0]-X2mean)**2+(testing[x,1]-Y2mean)**2)
    testing_d13=math.sqrt((testing[x,0]-notX2mean)**2+(testing[x,1]-notY2mean)**2)
    testing_d3=math.sqrt((testing[x,0]-X3mean)**2+(testing[x,1]-Y3mean)**2)
    testing_d12=math.sqrt((testing[x,0]-notX3mean)**2+(testing[x,1]-notY3mean)**2)
    if label_test[x]==1:
        if testing_d1<=testing_d23 and testing_d13<=testing_d2 and testing_d12<=testing_d3:
            testaccuracy=testaccuracy+1
    elif label_test[x]==2:
        if testing_d2<=testing_d13 and testing_d12<=testing_d3 and testing_d23<=testing_d1:
            testaccuracy = testaccuracy + 1
    elif label_test[x]==3:
        if testing_d3<=testing_d12 and testing_d13<=testing_d2 and testing_d23<=testing_d1:
            testaccuracy=testaccuracy+1

testing_accuracyrate=testaccuracy/(len(label_test))
print('The classification of ',(testaccuracy),'points in testing data are correct. Classification accuracy of testing set is :',(testing_accuracyrate)) #The accuracy of training data

sample_mean=([X1mean,Y1mean],[notX1mean,notY1mean],[X2mean,Y2mean],[notX2mean,notY2mean],[X3mean,Y3mean],[notX3mean,notY3mean])

#2(c)Show the classification and indeterminant region on one plot
pDBR.plotDecBoundaries_revised(training, label_train, sample_mean)

