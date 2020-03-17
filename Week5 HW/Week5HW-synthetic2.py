import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
np.set_printoptions(threshold=np.inf)
import  plotDecBoundaries_revised as pDBR

training_data = np.loadtxt(open("synthetic2_train.csv"),delimiter=",")
data_column=len(training_data[0])
data_row=len(training_data)
training=np.zeros([data_row,data_column])
for i in range(len(training)):
    if training_data[i,2]==2:
            training[i,0] = (-1)
            training[i,1]=(-1)*training_data[i,0]
            training[i,2] = (-1) * training_data[i, 1]

    else:
        training[i, 0] = 1
        training[i, 1] = training_data[i, 0]
        training[i, 2] = training_data[i, 1]



np.random.shuffle(training)
data_w=[0.1,0.1,0.1]
data_k=[0.1,0.1,0.1]

W_modifytimes=0
for epoch in range(1,10000):
    errorpoint = 0
    for i in range(data_row):
        a=np.inner(data_k,training[i])
        if a<0:
            errorpoint = errorpoint + 1
            data_k[0]=data_k[0]+training[i][0]
            data_k[1] = data_k[1] + training[i][1]
            data_k[2] = data_k[2] + training[i][2]
            W_modifytimes = W_modifytimes + 1
            data_w = np.insert(data_w, len(data_w), data_k, axis=0)
    if errorpoint==0:
#        print('epoch is',(epoch))
        break
    if epoch==1000:
#        print('epoch is 1000', (epoch))
        break
size=len(data_w)/3
data_w=np.reshape(data_w,(int(size),3))


#print(data_w)
#print(W_modifytimes)
final_w=data_w[len(data_w)-1]
print(final_w)
err=0
for i in range(data_row):
    innervalue=np.inner(final_w,training[i])
    if innervalue<0:
        err=err+1
errorrate=err/len(training)
print("The error rate of of training data %.5f" % errorrate)

label_train=training_data[:,2]
pDBR.plotDecBoundaries_revised(training_data, label_train, final_w)


##############testing
testing_data = np.loadtxt(open("synthetic2_train.csv"),delimiter=",")

testing=np.zeros([len(testing_data),len(testing_data[0])])
for i in range(len(testing_data)):
    if testing_data[i,2]==2:
        testing[i,0] = -1
        testing[i,1]=(-1)*testing_data[i,0]
        testing[i,2] = (-1) * testing_data[i,1]

    else:
        testing[i, 0] = 1
        testing[i, 1] = testing_data[i, 0]
        testing[i, 2] = testing_data[i, 1]

error_test=0
for i in range(len(testing_data)):
     test_inner=np.inner(final_w,testing[i])
     if test_inner<0:
         error_test=error_test+1

errrate_test=error_test/len(testing)
print("The error rate of testing data is %.5f" % errrate_test)
