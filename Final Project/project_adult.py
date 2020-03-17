from sklearn.svm import SVC
import numpy as np
from collections import Counter
import itertools
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.inf)
from sklearn import preprocessing
from sklearn.preprocessing import scale
from imblearn.over_sampling import SMOTE
from sklearn import svm, datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from imblearn.combine import SMOTEENN
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.neighbors import KNeighborsClassifier
train_data = pd.read_csv('adult.train_SMALLER.csv')
test_data = pd.read_csv('adult_test.csv')
def processing(data,trainornot):
    #change the name of the columns
    data.columns=['age','workclass','fnlwgt','education','education-num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','work_hours','native_country','label']
#remove the column of "capital-gain" and "capital-loss" because most of the data in the column are missing.
#    data.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)
#remove the column of 'education' because it's positive correlation to the next column 'education-num'
#    data.drop('education', axis=1, inplace=True)
    data['label'] = data['label'].str.replace('<=50K.', '0', case=False)
    data['label'] = data['label'].str.replace('>50K.', '1', case=False)
    data['label'] = data['label'].str.replace('<=50K', '0', case=False)
    data['label'] = data['label'].str.replace('>50K', '1', case=False)

    # dealing with the category "workclass"
    #combine "without pay" and "Never-worked" in workclass as "No salary"
    data['workclass'] = data['workclass'].str.replace('Without-pay', 'No salary', case=False)
    data['workclass'] = data['workclass'].str.replace('Never-worked', 'No salary', case=False)
    #combine "Local-gov" and "State-gov" as "other-gov"
    data['workclass'] = data['workclass'].str.replace('Local-gov', 'other-gov', case=False)
    data['workclass'] = data['workclass'].str.replace('State-gov', 'other-gov', case=False)
    # combine "Self-emp-inc" and "Self-emp-not-inc" as "Self-emp"
    data['workclass'] = data['workclass'].str.replace('Self-emp-inc', 'Self-emp', case=False)
    data['workclass'] = data['workclass'].str.replace('Self-emp-not-inc','Self-emp', case=False)

#    data['workclass'] = data['workclass'].str.replace('No salary','?', case=False)
#    data['workclass'] = data['workclass'].str.replace('Private','?', case=False)

    data['marital_status'] = data['marital_status'].str.replace('Married-civ-spouse','Married', case=False)
    data['marital_status'] = data['marital_status'].str.replace('Married-AF-spouse', 'Married', case=False)
    # combine "Married-spouse-absent" and "Separated" as "Separated"
    data['marital_status'] = data['marital_status'].str.replace('Married-spouse-absent', 'Separated', case=False)
    '''  
#try to deal with "relationship"
    data['relationship'] = data['relationship'].str.replace('Husband','Married', case=False)
    data['relationship'] = data['relationship'].str.replace('Wife','Married', case=False)
    data['relationship'] = data['relationship'].str.replace('Own-child','Married', case=False)
    data['relationship'] = data['relationship'].str.replace('Not-in-family','Unmarried', case=False)
    b=data.groupby(by='relationship').size()
    print(b)

##try to deal with "race"
#   data['race'] = data['race'].str.replace('Amer-Indian-Eskimo','Other', case=False)
   
    data['occupation'] = data['occupation'].str.replace('Armed-Forces', '?', case=False)
    data['occupation'] = data['occupation'].str.replace('Priv-house-serv', '?', case=False)
    
    data['occupation'] = data['occupation'].str.replace('Craft-repair', '?', case=False)
    '''


    data['occupation'] = data['occupation'].str.replace('Sales', 'Service and sales', case=False)
    data['occupation'] = data['occupation'].str.replace('Other-service', 'Service and sales', case=False)
    data['occupation'] = data['occupation'].str.replace('Priv-house-serv', 'Service and sales', case=False)
    data['occupation'] = data['occupation'].str.replace('Protective-serv', 'Service and sales', case=False)

    data['occupation'] = data['occupation'].str.replace('Tech-support', 'Professionals', case=False)
    data['occupation'] = data['occupation'].str.replace('Prof-specialty', 'Professionals', case=False)

    data['occupation'] = data['occupation'].str.replace('Farming-fishing', 'Elementary', case=False)
    data['occupation'] = data['occupation'].str.replace('Handlers-cleaners', 'Elementary', case=False)
    data['occupation'] = data['occupation'].str.replace('Transport-moving', 'Elementary', case=False)
    data['occupation'] = data['occupation'].str.replace('Exec-managerial', 'Managers', case=False)
    data['occupation'] = data['occupation'].str.replace('Armed-Forces', '?', case=False)

    #dealing with country
    country_list = [' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador',
                    ' El-Salvador', ' England',' France', ' Germany', ' Greece', ' Guatemala',' Haiti',
                      ' Honduras', ' Hong', ' Hungary',' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan',

                    ' Laos', ' Mexico', ' Nicaragua',' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland',

                   ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan',' Thailand',' Trinadad&Tobago', ' United-States',

                     ' Vietnam', ' Yugoslavia']
    arr_country_train = data['native_country'].unique()
    temp = [x for x in arr_country_train if x not in country_list]
#    print('after', temp)

    # if the country is not in the list, label it as "?"
    if len(temp) > 0:
        for i in range(len(temp)):
            data['native_country'] = data['native_country'].str.replace(temp[i],' ?', case=False)

    # dealing with the catagory "native_country"
    '''  
    # We classify the country by its income
    data['native_country'] = data['native_country'].str.replace('Outlying-US(Guam-USVI-etc)', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Cambodia', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Canada', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('China', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Columbia', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Cuba', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Dominican-Republic', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Ecuador', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('El-Salvador', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('England', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('France', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Germany', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Greece', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Guatemala', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Haiti', 'Low', case=False)
    data['native_country'] = data['native_country'].str.replace('Honduras', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Hong', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Hungary', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('India', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Iran', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Ireland', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Italy', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Jamaica', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Japan', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Laos', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Mexico', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Nicaragua', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Peru', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Philippines', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Poland', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Portugal', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Puerto-Rico', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Scotland', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('South', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Taiwan', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Thailand', 'Upper middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Trinadad&Tobago', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('United-States', 'High', case=False)
    data['native_country'] = data['native_country'].str.replace('Vietnam', 'Lower middle', case=False)
    data['native_country'] = data['native_country'].str.replace('Yugoslavia', 'Upper middle', case=False)
    '''
    
    # We classify the country by its location
    data['native_country'] = data['native_country'].str.replace('South', 'East Asia', case=False)
    data['native_country'] = data['native_country'].replace(' Outlying-US(Guam-USVI-etc)', ' Australia')
    data['native_country'] = data['native_country'].str.replace('Cambodia', 'South Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Canada', 'North America', case=False)
    data['native_country'] = data['native_country'].str.replace('China', 'East Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Columbia', 'South America', case=False)
    data['native_country'] = data['native_country'].str.replace('Cuba', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Dominican-Republic', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Ecuador', 'South America', case=False)
    data['native_country'] = data['native_country'].str.replace('El-Salvador', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('England', 'Western Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('France', 'Western Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Germany', 'Central Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Greece', 'Southeastern Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Guatemala', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Haiti', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Honduras', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Hong', 'East Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Hungary', 'Central Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('India', 'South Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Iran', 'Western Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Ireland', 'North Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Italy', 'Southern Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Jamaica', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Japan', 'East Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Laos', 'Southeast Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Mexico', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Nicaragua', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Peru', 'South America', case=False)
    data['native_country'] = data['native_country'].str.replace('Philippines', 'Southeast Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Poland', 'Central Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Portugal', 'Southern Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Puerto-Rico', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('Scotland', 'Western Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Taiwan', 'East Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Thailand', 'Southeast Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Trinadad&Tobago', 'Central America', case=False)
    data['native_country'] = data['native_country'].str.replace('United-States', 'North America', case=False)
    data['native_country'] = data['native_country'].str.replace('Vietnam', 'Southeast Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Yugoslavia', 'Southern Europe', case=False)

    '''
    #combine Australia to Southeastern Europe which also has low proportion in label 1
#    data['native_country'] = data['native_country'].str.replace('Southeastern Europe', 'Southern Europe', case=False)
    

    data['native_country'] = data['native_country'].str.replace('South', 'Asia', case=False)
    data['native_country'] = data['native_country'].replace(' Outlying-US(Guam-USVI-etc)', ' Australia')
    data['native_country'] = data['native_country'].str.replace('Cambodia', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Canada', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('China', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Columbia', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Cuba', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Dominican-Republic', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Ecuador', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('El-Salvador', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('England', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('France', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Germany', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Greece', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Guatemala', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Haiti', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Honduras', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Hong', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Hungary', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('India', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Iran', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Ireland', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Italy', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Jamaica', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Japan', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Laos', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Mexico', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Nicaragua', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Peru', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Philippines', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Poland', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Portugal', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Puerto-Rico', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Scotland', 'Europe', case=False)
    data['native_country'] = data['native_country'].str.replace('Taiwan', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Thailand', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Trinadad&Tobago', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('United-States', 'America', case=False)
    data['native_country'] = data['native_country'].str.replace('Vietnam', 'Asia', case=False)
    data['native_country'] = data['native_country'].str.replace('Yugoslavia', 'Europe', case=False)
#combine Australia to ?
#    data['native_country'] = data['native_country'].str.replace('Australia', '?', case=False)
    
 #   print(pd.crosstab(data.label, data.capital_loss))

#   income=[' High',' Lower middle',' Upper middle',' Low',]

    # try to separate the age into Youth , YoungAdult , MiddleAge , Senior
    
    bin = [10, 20, 30, 40, 50,60,70,80,90,100]
    data['age'] = pd.cut(data['age'], bin, labels=[1, 2, 3, 4,5,6,7,8,9], right=False)
    data['age'] = data['age'].astype('float64')
    age = data['age'].unique()
    print('HERE age',data['age'])
    print('age type',type(data['age']))
    
    bin = [0, 20, 40, 60,80,100]
    data['age'] = pd.cut(data['age'], bin, labels=[1, 2, 3, 4,5], right=False)
    data['age'] = data['age'].astype('float64')
    
    bin = [15,25, 40,65,100]
    data['age'] = pd.cut(data['age'], bin, labels=[1,2,3,4], right=False)
    data['age'] = data['age'].astype('float64')
    age = data['age'].unique()
    '''

    worktime = [0, 30, 40, 60,100]
    data['work_hours'] = pd.cut(data['work_hours'], worktime, labels=[1, 2, 3,4], right=False)
    data['work_hours'] = data['work_hours'].astype('float64')

    #deal with education-num
    '''
    edu = [0,8, 11, 14, 17]
    data['education-num'] = pd.cut(data['education-num'], edu, labels=[1, 2, 3, 4], right=False)
    data['education-num'] = data['education-num'].astype('float64')
    edu = [0, 6, 12, 17]
    data['education-num'] = pd.cut(data['education-num'], edu, labels=[1, 2, 3], right=False)
    data['education-num'] = data['education-num'].astype('float64')
    '''


    #standardize numerical data
    numdata =data.loc[:,['fnlwgt','education-num','work_hours','capital_loss','capital_gain']]

#std data
    stddata = scale(numdata)
    '''
#normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(numdata)
    stddata = pd.DataFrame(np_scaled)
    '''
    #    print(stddata)
    strdata=data.loc[:,['workclass','marital_status','occupation', 'relationship', 'native_country','race']]
    dummydata=pd.get_dummies(strdata)
    npalldata = np.concatenate([stddata, dummydata], axis = 1)
    Pro_label=data.iloc[:,data.shape[1]-1]
    alldata = npalldata
    label = Pro_label

    '''
    #decide not to use SMOTE
    if trainornot == 1:
        smo = SMOTE(ratio='minority')
        alldata, label = smo.fit_resample(npalldata, Pro_label)
        np.append(alldata, label)
        print("before label=", Counter(Pro_label))
        print("after label=", Counter(label))
    else:
        alldata = npalldata
        label = Pro_label
    '''
    return alldata,label
(traindata, trainlabel) = processing(train_data,1)
(testdata, testlabel) = processing(test_data,0)


#Use Perceptron for 100 times in different initial weight vector and choose the one with the best accuracy in traindata
for i in range(0,100):
    initial_weight_train=np.random.randn(1,len(traindata[1]))
    per=Perceptron(tol=1e-3)
    per.fit(traindata,trainlabel,coef_init=initial_weight_train) #get the mean and standard deviation
    pre_label_train=per.predict(traindata)
    accuracy_train=accuracy_score(trainlabel,pre_label_train)
    pre_label_test = per.predict(testdata)
    accuracy_test=accuracy_score(testlabel,pre_label_test)

    if i==0:
        accarr_train=[]
        finalW_array_train=[]
        accarr_test=[]
    accarr_train=np.append(accarr_train,accuracy_train)
    accarr_test=np.append(accarr_test,accuracy_test)
    finalW_array_train=np.append(finalW_array_train,initial_weight_train)

best_acc_Per=max(accarr_train)
best_index=[]
random_best_index=[]
for i in range (0,len(accarr_train)):  #to check how many times the max accuracy happened in the array
    if accarr_train[i]==best_acc_Per:
        best_index=np.r_[best_index,i]
if len(best_index)>=2: #if the max accuracy happened more than once
    random_best_index=random.sample(list(best_index),1)#we need to choose the max accuracy index randomly
    a = int(random_best_index[0])
else:     #if the max accuracy happened only one time
    random_best_index=best_index[0]  #save the index info
    a = int(random_best_index)
optimal_initW=finalW_array_train[a * len(traindata[1]):(a + 1) * len(traindata[1])]

print ('The best accuracy for all features in training data in 100 times run with different initial w vector is',best_acc_Per)
print('The accuracy for complete testing data when using the optimal initial weight vector in perceptron is ',accarr_test[a])

per2 = Perceptron(tol=1e-3)
per2.fit(traindata, trainlabel, coef_init=optimal_initW)  # get the mean and standard deviation
pre_label_train = per.predict(traindata)
pre2_label_test = per2.predict(testdata)

print('Accuracy report for Perceptron=',metrics.classification_report(testlabel, pre2_label_test))

per_CM = confusion_matrix(pre2_label_test, testlabel)
print(per_CM)
plt.imshow(per_CM)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of Perceptron classifier')
plt.xlabel('True Test Label')
plt.ylabel('Predict Test Label')
plt.colorbar()
thresh = per_CM.max()
for i, j in itertools.product(range(per_CM.shape[0]), range(per_CM.shape[1])):
    plt.text(j, i, per_CM[i, j], horizontalalignment="center",
    color="white" if per_CM[i, j] > thresh else "black")
plt.show()



'''
#SVM use cross validation to find the optimal pair (r,C)
C_range = np.logspace(-2, 2, 25)
gamma_range = np.logspace(-2, 2, 25)

ACC=np.zeros([len(gamma_range),len(C_range)])
DEV=np.zeros([len(gamma_range),len(C_range)])
CV=StratifiedKFold(n_splits=5,shuffle=True)
for a in range(len(gamma_range)):
    for b in range(len(C_range)):
        CV_rbf = SVC(kernel='rbf', gamma=gamma_range[a], C=C_range[b])
        acc_CV=cross_val_score(CV_rbf,traindata,trainlabel,cv=CV)
        ACC[a][b]=np.mean(acc_CV)
        DEV[a][b]=np.std(acc_CV,ddof=1)
        print('r=',a)
        print('C=', b)
        print('acc=', np.mean(acc_CV))
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

'''

SVC_Classifier=SVC(kernel='rbf',C=3.162,gamma=0.068)
SVC_Classifier.fit(traindata,trainlabel) #get the mean and standard deviation
SVC_predict_trainlabel=SVC_Classifier.predict(traindata)
SVC_Classifier_trainaccuracy=accuracy_score(trainlabel,SVC_predict_trainlabel)
SVC_predict_testlabel=SVC_Classifier.predict(testdata)
SVC_Classifier_testaccuracy=accuracy_score(testlabel,SVC_predict_testlabel)
print('The training accuracy in SVM is ',SVC_Classifier_trainaccuracy)
print('The testing accuracy in SVM is ',SVC_Classifier_testaccuracy)
print('Accuracy report for SVM=',metrics.classification_report(testlabel, SVC_predict_testlabel))

SVM_CM = confusion_matrix(SVC_predict_testlabel, testlabel)
print(SVM_CM)
plt.imshow(SVM_CM)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of SVM classifier')
plt.xlabel('True Test Label')
plt.ylabel('Predict Test Label')
plt.colorbar()
thresh = SVM_CM.max()
for i, j in itertools.product(range(SVM_CM.shape[0]), range(SVM_CM.shape[1])):
    plt.text(j, i, SVM_CM[i, j], horizontalalignment="center",
    color="white" if SVM_CM[i, j] > thresh else "black")
plt.show()


'''
#find the optimal K for Knn
cv_scores = []
Knn_acc=[]
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(traindata, trainlabel)
    scores = cross_val_score(knn, traindata, trainlabel, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    knn_predict_trainlabel = knn.predict(traindata)
    knn_Classifier_trainaccuracy = accuracy_score(trainlabel, knn_predict_trainlabel)
    Knn_acc=np.append(Knn_acc,knn_Classifier_trainaccuracy)
    print(i)
print(cv_scores)
'''

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(traindata,trainlabel)
knn_predict_trainlabel=knn.predict(traindata)
knn_predict_testlabel=knn.predict(testdata)
knn_Classifier_trainaccuracy=accuracy_score(trainlabel,knn_predict_trainlabel)
knn_Classifier_testaccuracy=accuracy_score(testlabel,knn_predict_testlabel)

print('The training accuracy in KNN is ',knn_Classifier_trainaccuracy)
print('The testing accuracy in KNN is ',knn_Classifier_testaccuracy)
print('Accuracy report for KNN=',metrics.classification_report(testlabel, knn_predict_testlabel))

KNN_CM = confusion_matrix(knn_predict_testlabel, testlabel)
print(KNN_CM)
plt.imshow(KNN_CM)
labels = ['negative', 'positive']
xlocations = np.array(range(len(labels)))
plt.xticks(xlocations, labels, rotation=0)
plt.yticks(xlocations, labels)
plt.title('Confusion matrix of KNN classifier')
plt.xlabel('True Test Label')
plt.ylabel('Predict Test Label')
plt.colorbar()
thresh = KNN_CM.max()
for i, j in itertools.product(range(SVM_CM.shape[0]), range(KNN_CM.shape[1])):
    plt.text(j, i, KNN_CM[i, j], horizontalalignment="center",
    color="white" if KNN_CM[i, j] > thresh else "black")
plt.show()