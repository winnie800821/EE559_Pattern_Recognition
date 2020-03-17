import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist

training = np.loadtxt(open("wine_train.csv"), delimiter=",")

label_train = training[:,13]
X1mean=np.sum(training[label_train==1,0])/np.count_nonzero(label_train==1)
Y1mean=np.sum(training[label_train==1,11])/np.count_nonzero(label_train==1)
print('In wine_train,class1 Xmean is ',(X1mean))
print('In wine_train,class1 Ymean is ',(Y1mean))
plt.scatter(training[label_train==1,0],training[label_train==1,11],marker="^",c="r",label="class1")
plt.scatter(X1mean,Y1mean,marker="s",label="class1 mean")

X2mean=np.sum(training[label_train==2,0])/np.count_nonzero(label_train==2)
Y2mean=np.sum(training[label_train==2,11])/np.count_nonzero(label_train==2)
print('In wine_train,class2 Xmean is ',(X2mean))
print('In wine_train,class2 Ymean is ',(Y2mean))
plt.scatter(training[label_train==2,0],training[label_train==2,11],marker="o",c="g",label="class2")
plt.scatter(X2mean,Y2mean,marker=(5, 2),c="g",label="class2 mean")

X3mean=np.sum(training[label_train==3,0])/np.count_nonzero(label_train==3)
Y3mean=np.sum(training[label_train==3,11])/np.count_nonzero(label_train==3)
print('In wine_train,class3 Xmean is ',(X3mean))
print('In wine_train,class3 Ymean is ',(Y3mean))
plt.scatter(training[label_train==3,0],training[label_train==3,11],marker="+",c="b",label="class3")
plt.scatter(X3mean,Y3mean,marker="1",c="b",label="class3 mean")

plt.legend(loc='lower right')

sample_mean=([X1mean,Y1mean],[X2mean,Y2mean],[X3mean,Y3mean])
plt.show()


#def plotDecBoundaries(training, label_train, sample_mean):



nclass = max(np.unique(label_train))

# Set the feature range for ploting
max_x = np.ceil(max(training[:, 0])) + 1
min_x = np.floor(min(training[:, 0])) - 1
max_y = np.ceil(max(training[:, 11])) + 1
min_y = np.floor(min(training[:, 11])) - 1

xrange = (min_x, max_x)
yrange = (min_y, max_y)

# step size for how finely you want to visualize the decision boundary.
inc = 0.005

# generate grid coordinates. this will be the basis of the decision
# boundary visualization.
(x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc), np.arange(yrange[0], yrange[1] + inc / 100, inc))

# size of the (x, y) image, which will also be the size of the
# decision boundary image that is used as the plot background.
image_size = x.shape
xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  # make (x,y) pairs as a bunch of row vectors.

# distance measure evaluations for each (x,y) pair.
dist_mat = cdist(xy, sample_mean)
pred_label = np.argmin(dist_mat, axis=1)

# reshape the idx (which contains the class label) into an image.
decisionmap = pred_label.reshape(image_size, order='F')

# show the image, give each coordinate a color according to its class label
plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

# plot the class training data.
plt.plot(training[label_train == 1, 0], training[label_train == 1, 11], 'rx')
plt.plot(training[label_train == 2, 0], training[label_train == 2, 11], 'go')
if nclass == 3:
    plt.plot(training[label_train == 3, 0], training[label_train == 3, 11], 'b*')

# include legend for training data
if nclass == 3:
    l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
else:
    l = plt.legend(('Class 1', 'Class 2'), loc=2)
plt.gca().add_artist(l)

# plot the class mean vector.
m1, = plt.plot(sample_mean[0][0], sample_mean[0][1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
m2, = plt.plot(sample_mean[1][0], sample_mean[1][1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
if nclass == 3:
    m3, = plt.plot(sample_mean[2][0], sample_mean[2][1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

# include legend for class mean vector
if nclass == 3:
    l1 = plt.legend([m1, m2, m3], ['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
else:
    l1 = plt.legend([m1, m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

plt.gca().add_artist(l1)
plt.show()



#############################training data
class1original=training[label_train==1,:]
listcountclass1=[]
for x in range(np.count_nonzero(label_train==1)):
 d1=math.sqrt((class1original[x,0]-X1mean)**2+(class1original[x,11]-Y1mean)**2)
 d2=math.sqrt((class1original[x,0]-X2mean)**2+(class1original[x,11]-Y2mean)**2)
 d3=math.sqrt((class1original[x,0]-X3mean)**2+(class1original[x,11]-Y3mean)**2)
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
 d1=math.sqrt((class2original[x,0]-X1mean)**2+(class2original[x,11]-Y1mean)**2)
 d2=math.sqrt((class2original[x,0]-X2mean)**2+(class2original[x,11]-Y2mean)**2)
 d3=math.sqrt((class2original[x,0]-X3mean)**2+(class2original[x,11]-Y3mean)**2)
 if d1>=d2 and d3>=d2:
     listcountclass2.insert(x,0)
   ##when we insert 0 means correct
 else:
    listcountclass2.insert(x,1)
   ##when we insert 1 means wrong

class2err=np.sum(listcountclass2) #how many errors in class2


class3original=training[label_train==3,:]
listcountclass3=[]
for x in range(np.count_nonzero(label_train==3)):
 d1=math.sqrt((class3original[x,0]-X1mean)**2+(class3original[x,11]-Y1mean)**2)
 d2=math.sqrt((class3original[x,0]-X2mean)**2+(class3original[x,11]-Y2mean)**2)
 d3=math.sqrt((class3original[x,0]-X3mean)**2+(class3original[x,11]-Y3mean)**2)
 if d1>=d3 and d2>=d3:
     listcountclass3.insert(x,0)
   ##when we insert 0 means correct
 else:
    listcountclass3.insert(x,1)
   ##when we insert 1 means wrong


class3err=np.sum(listcountclass3) #how many errors in class3


errorrate=(class1err+class2err+class3err)/(len(class1original)+len(class2original)+len(class3original))
print('There are ',(class1err+class2err+class3err),'points misclassified in training dataset.')
print('error rate of training data is :',(errorrate)) #The error rate of training data



#############################testing data
testing = np.loadtxt(open("wine_test.csv"), delimiter=",")
label_test=testing [:,13]
class1originaltest=testing[label_test==1,:]
listcountclass1test=[]
for x in range(np.count_nonzero(label_test==1)):
 d1test=math.sqrt((class1originaltest[x,0]-X1mean)**2+(class1originaltest[x,11]-Y1mean)**2)
 d2test=math.sqrt((class1originaltest[x,0]-X2mean)**2+(class1originaltest[x,11]-Y2mean)**2)
 d3test=math.sqrt((class1originaltest[x,0]-X3mean)**2+(class1originaltest[x,11]-Y3mean)**2)
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
 d1test=math.sqrt((class2originaltest[x,0]-X1mean)**2+(class2originaltest[x,11]-Y1mean)**2)
 d2test=math.sqrt((class2originaltest[x,0]-X2mean)**2+(class2originaltest[x,11]-Y2mean)**2)
 d3test=math.sqrt((class2originaltest[x,0]-X3mean)**2+(class2originaltest[x,11]-Y3mean)**2)
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
 d1test=math.sqrt((class3originaltest[x,0]-X1mean)**2+(class3originaltest[x,11]-Y1mean)**2)
 d2test=math.sqrt((class3originaltest[x,0]-X2mean)**2+(class3originaltest[x,11]-Y2mean)**2)
 d3test=math.sqrt((class3originaltest[x,0]-X3mean)**2+(class3originaltest[x,11]-Y3mean)**2)
 if d1test>=d3test and d2test>=d3test:
     listcountclass3test.insert(x,0)
   ##when we insert 0 means correct
 else:
    listcountclass3test.insert(x,1)
   ##when we insert 1 means wrong
class3errtest=np.sum(listcountclass3test) #how many errors in class2



errorratetest=(class1errtest+class2errtest+class3errtest)/(len(class1originaltest)+len(class2originaltest)+len(class3originaltest))
print('error rate of testing data is :',(errorratetest)) #The error rate of testing data








