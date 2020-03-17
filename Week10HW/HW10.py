from sklearn.svm import SVC
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotSVMBoundaries as pSVMB


#question1
#(a)
print("Question 1 (a)")
training = np.loadtxt(open("HW10_1_csv/train_x.csv"), delimiter=",")
training_label=np.loadtxt(open("HW10_1_csv/train_y.csv"), delimiter=",")

##C=1
SVC_Classifier_C1=SVC(kernel='linear')
SVC_Classifier_C1.fit(training,training_label) #get the mean and standard deviation
SVC_C1_predict_label=SVC_Classifier_C1.predict(training)
SVC_Classifier_C1_accuracy=accuracy_score(training_label,SVC_C1_predict_label)
print('The accuracy for c=1 in SVM is ',SVC_Classifier_C1_accuracy)
pSVMB.plotSVMBoundaries(training, training_label, SVC_Classifier_C1)
##C=100
SVC_Classifier_C100=SVC(kernel='linear',C=100)
SVC_Classifier_C100.fit(training,training_label) #get the mean and standard deviation
SVC_C100_predict_label=SVC_Classifier_C100.predict(training)
SVC_Classifier_C100_accuracy=accuracy_score(training_label,SVC_C100_predict_label)
print('The accuracy for c=100 in SVM is ',SVC_Classifier_C100_accuracy)
pSVMB.plotSVMBoundaries(training, training_label, SVC_Classifier_C100)

#(b)
print("Question 1 (b)")
Support_vector_C100=SVC_Classifier_C100.support_vectors_
pSVMB.plotSVMBoundaries(training, training_label, SVC_Classifier_C100,Support_vector_C100)
W=SVC_Classifier_C100.coef_
W0=SVC_Classifier_C100.intercept_
print("W0=",W0[0])
print("W1=",W[0][0])
print("W2=",W[0][1])
print("The decision boundary equation is (",W[0][0],")X1+(",W[0][1],")X2+(",W0[0],")=0")

#(c)
print("Question 1 (c)")
print("The pairs of support vectors are",Support_vector_C100)
SV1=Support_vector_C100[0]
SV2=Support_vector_C100[1]
SV3=Support_vector_C100[2]

g=[0,0,0]
for i in range (0,3):
    g[i]=Support_vector_C100[i][0]*W[0][0]+Support_vector_C100[i][1]*W[0][1]+W0[0]
    print("g(X) of support vector",i+1,"when C=100 is",g[i])

##test the conjecture C=10000000
print("test the conjecture C=10000000")
SVC_Classifier_C10000000=SVC(kernel='linear',C=10000000)
SVC_Classifier_C10000000.fit(training,training_label) #get the mean and standard deviation
SVC_C10000000_predict_label=SVC_Classifier_C10000000.predict(training)
W_test=SVC_Classifier_C10000000.coef_
W0_test=SVC_Classifier_C10000000.intercept_
Support_vector_C10000000=SVC_Classifier_C10000000.support_vectors_
SV1_1=Support_vector_C10000000[0]
SV2_1=Support_vector_C10000000[1]
SV3_1=Support_vector_C10000000[2]

a=[0,0,0]
for i in range (0,3):
    a[i]=Support_vector_C10000000[i][0]*W_test[0][0]+Support_vector_C10000000[i][1]*W_test[0][1]+W0_test[0]
    print("g(X) of support vector",i+1,"when C=10000000 is",a[i])



#(d)
print("Question 1 (d)")
training2 = np.loadtxt(open("HW10_2_csv/train_x.csv"), delimiter=",")
training_label2=np.loadtxt(open("HW10_2_csv/train_y.csv"), delimiter=",")
#C=50
rbf_C50 = SVC(kernel='rbf',C=50,gamma='auto')
rbf_C50.fit(training2, training_label2)
rbf_C50_predict_label=rbf_C50.predict(training2)
rbf_C50_accuracy=accuracy_score(training_label2,rbf_C50_predict_label)
print('When C=50, the accuracy is',rbf_C50_accuracy)
pSVMB.plotSVMBoundaries(training2, training_label2, rbf_C50)
#C=5000
rbf_C5000 = SVC(kernel='rbf',C=5000,gamma='auto')
rbf_C5000.fit(training2, training_label2)
rbf_C5000_predict_label=rbf_C5000.predict(training2)
rbf_C5000_accuracy=accuracy_score(training_label2,rbf_C5000_predict_label)
print('When C=5000, the accuracy is',rbf_C5000_accuracy)
pSVMB.plotSVMBoundaries(training2, training_label2, rbf_C5000)

#(e)
print("Question 1 (e)")
#gamma=10
rbf_r10 = SVC(kernel='rbf',gamma=10)
rbf_r10.fit(training2, training_label2)
rbf_r10_predict_label=rbf_r10.predict(training2)
rbf_r10_accuracy=accuracy_score(training_label2,rbf_r10_predict_label)
print('When gamma=10, the accuracy is',rbf_r10_accuracy)
pSVMB.plotSVMBoundaries(training2, training_label2, rbf_r10)
#gamma=50
rbf_r50 = SVC(kernel='rbf',gamma=50)
rbf_r50.fit(training2, training_label2)
rbf_r50_predict_label=rbf_r50.predict(training2)
rbf_r50_accuracy=accuracy_score(training_label2,rbf_r50_predict_label)
print('When gamma=50, the accuracy is',rbf_r50_accuracy)
pSVMB.plotSVMBoundaries(training2, training_label2, rbf_r50)
#gamma=50
rbf_r500 = SVC(kernel='rbf',gamma=500)
rbf_r500.fit(training2, training_label2)
rbf_r500_predict_label=rbf_r500.predict(training2)
rbf_r500_accuracy=accuracy_score(training_label2,rbf_r500_predict_label)
print('When gamma=500, the accuracy is',rbf_r500_accuracy)
pSVMB.plotSVMBoundaries(training2, training_label2, rbf_r500)


print("Question 2 (a)")
wine_training = np.loadtxt(open("wine_csv/feature_train.csv"), delimiter=",")
wine_training_label=np.loadtxt(open("wine_csv/label_train.csv"), delimiter=",")
wine_traindata=np.zeros([wine_training_label.shape[0],2])
wine_traindata=wine_training[:,0:2]

CV=StratifiedKFold(n_splits=5,shuffle=True)
CV_rbf=SVC(kernel='rbf', gamma=1,C=1)
acc_CV=cross_val_score(CV_rbf,wine_traindata,wine_training_label,cv=CV)
print("When r=1 and C=1, the average cross-validation accuracy is %.3f"%(np.mean(acc_CV)))

#(b)
print("Question 2 (b)")
C_range = np.logspace(-3, 3, 50)
gamma_range = np.logspace(-3, 3, 50)
ACC=np.zeros([len(gamma_range),len(C_range)])
DEV=np.zeros([len(gamma_range),len(C_range)])
CV=StratifiedKFold(n_splits=5,shuffle=True)
for a in range(len(gamma_range)):
    for b in range(len(C_range)):
        CV_rbf = SVC(kernel='rbf', gamma=gamma_range[a], C=C_range[b])
        acc_CV=cross_val_score(CV_rbf,wine_traindata,wine_training_label,cv=CV)
        ACC[a][b]=np.mean(acc_CV)
        DEV[a][b]=np.std(acc_CV,ddof=1)

max_acc_b= np.amax(ACC)
max_index_b=np.where(ACC==np.max(ACC))

if np.size(max_index_b) != 2:
    y = len(max_index_b[0])
    compare_std=([])
    compare_std = np.zeros(y)
    for w in range(y):
        compare_std[w] = DEV[max_index_b[0][w], max_index_b[1][w]]
    min_std_index = np.where(compare_std == np.min(compare_std))
    m = max_index_b[0][min_std_index[0]]
    n = max_index_b[1][min_std_index[0]]
    max_index_b = ([])
    max_index_b = ([m, n])


print("The max mean accuracy is %.3f and the unbiased standard deviation is %.3f" %(max_acc_b,DEV[max_index_b[0],max_index_b[1]]))
print("and it happened when gamma=%.3f,C=%.3f" %(gamma_range[max_index_b[0]],C_range[max_index_b[1]]))

#Visualize ACC
imgplot = plt.imshow(ACC)
plt.colorbar()
plt.show()


print("Question 2 (c)")
ACC_20t=np.zeros([50,50,20],dtype=np.float64)
DEV_20t=np.zeros([50,50,20],dtype=np.float64)
ACC=([])
DEV=([])
ACC=np.zeros([len(gamma_range),len(C_range)])
DEV=np.zeros([len(gamma_range),len(C_range)])
best_r=0
best_C=0
final_ACC=np.zeros([50,50])
final_DEV=np.zeros([50,50])
print("(i)")
for k in range(20):
    CV = StratifiedKFold(n_splits=5, shuffle=True)
    ACC=np.zeros_like(ACC)
    DEV=np.zeros_like(DEV)
    for a in range(len(gamma_range)):
        for b in range(len(C_range)):
            CV_rbf = SVC(kernel='rbf', gamma=gamma_range[a], C=C_range[b])
            acc_CV=cross_val_score(CV_rbf,wine_traindata,wine_training_label,cv=CV)
            ACC[a][b]=np.mean(acc_CV)
            DEV[a][b]=np.std(acc_CV,ddof=1)
            ACC_20t[a][b][k] = np.mean(acc_CV)
            DEV_20t[a][b][k] = np.std(acc_CV,ddof=1)
#(i)report on the 20 chosen pairs of [γ, C].
    max_index_perT=np.where(ACC==np.amax(ACC))
    min_std_index = ([])
    compare_std = ([])
    p = len(max_index_perT[0])
    compare_std = np.zeros(p)
    if np.size(max_index_perT) != 2:
        for w in range(p):
            compare_std[w] = DEV[max_index_perT[0][w], max_index_perT[1][w]]
        min_std_index = np.where(compare_std == np.amin(compare_std))
        if np.size(min_std_index)!=1:
            min_std_index=np.amin(min_std_index)
        m = max_index_perT[0][min_std_index]
        n = max_index_perT[1][min_std_index]
        max_index_perT = ([])
        max_index_perT = ([m, n])
    best_r=gamma_range[max_index_perT[0]]
    best_C=C_range[max_index_perT[1]]
    std= final_DEV[max_index_perT[0],max_index_perT[1]]
    print("%d. (r,C)= (%.3f,%.3f)"%(k+1,best_r,best_C))

acc_t=np.zeros(20)#store acc in different T but same r,c
for a in range(50):
    for b in range(50):
        acc_t=np.zeros_like(acc_t)
        for k in range(20):
            acc_t[k]=ACC_20t[a][b][k]
        final_ACC[a][b]=np.mean(acc_t)
        final_DEV[a][b] = np.std(acc_t, ddof=1)

max_index=([])
max_index = np.where(final_ACC == np.amax(final_ACC))
#Sometimes, the results of best acc are the same.
#So we choose the one with smallest std

if np.size(max_index)!=2:
    p=len(max_index[0])
    compare_std=([])
    compare_std = np.zeros(p)
    for w in range(p):
        compare_std[w]=final_DEV[max_index[0][w],max_index[1][w]]
    min_std=np.amin(compare_std)
    min_std_index=np.where(compare_std==min_std)
    a=max_index[0][min_std_index[0]]
    b=max_index[1][min_std_index[0]]
    max_index=([])
    max_index=([a,b])

final_r=gamma_range[max_index[0]]
final_C=C_range[max_index[1]]
final_std_value= final_DEV[max_index[0],max_index[1]]
final_acc_value=final_ACC[max_index[0],max_index[1]]

print("(ii)")
print("The the final chosen best values for (r,C) is ( %.3f,%.3f )"%(final_r,final_C))
print("mean cross-validation accuracy : %.3f" %(final_acc_value))
print("standard deviation : %f" %(final_std_value))

print("Question 2 (d)")
wine_test = np.loadtxt(open("wine_csv/feature_test.csv"), delimiter=",")
wine_test_label=np.loadtxt(open("wine_csv/label_test.csv"), delimiter=",")
wine_testdata=np.zeros(([len(wine_test_label),2]))
wine_testdata=wine_test[:,0:2]

wine_model = SVC(kernel='rbf',gamma=final_r,C=final_C)
wine_model.fit(wine_traindata, wine_training_label)
wine_test_model_predict_label=wine_model.predict(wine_testdata)
wine_test_accuracy=accuracy_score(wine_test_label,wine_test_model_predict_label)

print('When we use the chosen (r,C), the accuracy is %.3f'%(wine_test_accuracy))
number_std=(final_acc_value-wine_test_accuracy)/(final_std_value)
print("The estimate is approximately %.3f standard deviation of the mean cross-validation accuracy from (c)"%(number_std))