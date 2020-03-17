#answer for question(d)and(e)
#find the best feature from the error rate
#after we know the best pair of figures,
#plot 2D classification figure and compute the error rate (training data and testing data)

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from prettytable import PrettyTable


training = np.loadtxt(open("wine_train.csv"), delimiter=",")
label_train = training[:,13]
testing = np.loadtxt(open("wine_test.csv"), delimiter=",")
label_test=testing [:,13]
f= PrettyTable(["feature 1","feature 2","training err%","testing err%"])
trainerrratelist=[]
#in order to get the lowest error rate and std of error% in training
testerrratelist=[]
#in order to get the std of error% in testing
item=0
#the index of the table


for n in range(13):
    for k in range(n+1,13):
       X1mean=np.sum(training[label_train==1,n])/np.count_nonzero(label_train==1)
       Y1mean=np.sum(training[label_train==1,k])/np.count_nonzero(label_train==1)

       X2mean=np.sum(training[label_train==2,n])/np.count_nonzero(label_train==2)
       Y2mean=np.sum(training[label_train==2,k])/np.count_nonzero(label_train==2)

       X3mean=np.sum(training[label_train==3,n])/np.count_nonzero(label_train==3)
       Y3mean=np.sum(training[label_train==3,k])/np.count_nonzero(label_train==3)

       item=item+1
#############################training data
       class1original=training[label_train==1,:]
       listcountclass1=[]
       for x in range(np.count_nonzero(label_train==1)):
           d1=math.sqrt((class1original[x,n]-X1mean)**2+(class1original[x,k]-Y1mean)**2)
           d2=math.sqrt((class1original[x,n]-X2mean)**2+(class1original[x,k]-Y2mean)**2)
           d3=math.sqrt((class1original[x,n]-X3mean)**2+(class1original[x,k]-Y3mean)**2)
           if d1<=d2 and d1<=d3:
            listcountclass1.insert(x,0)
            ##when we insert 0 means correct
           else:
                listcountclass1.insert(x,1)
 ##when we insert 1 means wrong
       class1err=np.sum(listcountclass1) #how many errors in class1

       class2original=training[label_train==2,:]
       listcountclass2=[]
       for x in range(np.count_nonzero(label_train==2)):
            d1=math.sqrt((class2original[x,n]-X1mean)**2+(class2original[x,k]-Y1mean)**2)
            d2=math.sqrt((class2original[x,n]-X2mean)**2+(class2original[x,k]-Y2mean)**2)
            d3=math.sqrt((class2original[x,n]-X3mean)**2+(class2original[x,k]-Y3mean)**2)
            if d1>=d2 and d3>=d2:
                listcountclass2.insert(x,0)
                 ##when we insert 0 means correct
            else:
                listcountclass2.insert(x,1)
            ##when we insert 1 means wrong
       class2err=np.sum(listcountclass2)
       #how many errors in class2


       class3original=training[label_train==3,:]
       listcountclass3=[]
       for x in range(np.count_nonzero(label_train==3)):
            d1=math.sqrt((class3original[x,n]-X1mean)**2+(class3original[x,k]-Y1mean)**2)
            d2=math.sqrt((class3original[x,n]-X2mean)**2+(class3original[x,k]-Y2mean)**2)
            d3=math.sqrt((class3original[x,n]-X3mean)**2+(class3original[x,k]-Y3mean)**2)
            if d1>=d3 and d2>=d3:
              listcountclass3.insert(x,0)
            ##when we insert 0 means correct
            else:
                listcountclass3.insert(x,1)
            ##when we insert 1 means wrong
       class3err=np.sum(listcountclass3) #how many errors in class3
       errorrate=(class1err+class2err+class3err)/(len(class1original)+len(class2original)+len(class3original))

       trainerrratelist.insert(item,errorrate)


#############################testing data
       class1originaltest=testing[label_test==1,:]
       listcountclass1test=[]
       for x in range(np.count_nonzero(label_test==1)):
            d1test=math.sqrt((class1originaltest[x,n]-X1mean)**2+(class1originaltest[x,k]-Y1mean)**2)
            d2test=math.sqrt((class1originaltest[x,n]-X2mean)**2+(class1originaltest[x,k]-Y2mean)**2)
            d3test=math.sqrt((class1originaltest[x,n]-X3mean)**2+(class1originaltest[x,k]-Y3mean)**2)
            if d1test<=d2test and d1test<=d3test:
                listcountclass1test.insert(x,0)
             ##when we insert 0 means correct
            else:
                listcountclass1test.insert(x,1)
            ##when we insert 1 means wrong
       class1errtest=np.sum(listcountclass1test) #how many errors in class1 in testing data


       class2originaltest=testing[label_test==2,:]
       listcountclass2test=[]
       for x in range(np.count_nonzero(label_test==2)):
            d1test=math.sqrt((class2originaltest[x,n]-X1mean)**2+(class2originaltest[x,k]-Y1mean)**2)
            d2test=math.sqrt((class2originaltest[x,n]-X2mean)**2+(class2originaltest[x,k]-Y2mean)**2)
            d3test=math.sqrt((class2originaltest[x,n]-X3mean)**2+(class2originaltest[x,k]-Y3mean)**2)
            if d1test>=d2test and d3test>=d2test:
              listcountclass2test.insert(x,0)
         ##when we insert 0 means correct
            else:
                listcountclass2test.insert(x,1)
   ##when we insert 1 means wrong
       class2errtest=np.sum(listcountclass2test) #how many errors in class2

       class3originaltest=testing[label_test==3,:]
       listcountclass3test=[]
       for x in range(np.count_nonzero(label_test==3)):
            d1test=math.sqrt((class3originaltest[x,n]-X1mean)**2+(class3originaltest[x,k]-Y1mean)**2)
            d2test=math.sqrt((class3originaltest[x,n]-X2mean)**2+(class3originaltest[x,k]-Y2mean)**2)
            d3test=math.sqrt((class3originaltest[x,n]-X3mean)**2+(class3originaltest[x,k]-Y3mean)**2)
            if d1test>=d3test and d2test>=d3test:
               listcountclass3test.insert(x,0)
                 ##when we insert 0 means correct
            else:
                 listcountclass3test.insert(x,1)
            ##when we insert 1 means wrong
       class3errtest=np.sum(listcountclass3test) #how many errors in class2
       errorratetest=(class1errtest+class2errtest+class3errtest)/(len(class1originaltest)+len(class2originaltest)+len(class3originaltest))
       testerrratelist.insert(item,errorratetest)
       f.add_row([n+1,k+1,errorrate,errorratetest])

print(f)
print("The minimum error rate in training data is",(min(trainerrratelist)))
print(f[trainerrratelist.index(min(trainerrratelist))])
print("The standard deviation of error rate in training data is",(np.std(trainerrratelist)))
print("The standard deviation of error rate in testing data is",(np.std(testerrratelist)))
print("The mean of training data is",(np.sum(trainerrratelist)/78))
print("The mean of testing data is",(np.sum(testerrratelist)/78))





