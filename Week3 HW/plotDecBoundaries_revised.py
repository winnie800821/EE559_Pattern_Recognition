################################################
## EE559 HW Wk2, Prof. Jenkins, Spring 2018
## Created by Arindam Jati, TA
## Tested in Python 3.6.3, OSX El Captain
################################################
########Revised version########################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries_revised(training, label_train, sample_mean):

    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: training data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    nclass =  max(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    dist_mat = cdist(xy, sample_mean)
    #sample_mean = ([X1mean, Y1mean], [notX1mean, notY1mean], [X2mean, Y2mean], [notX2mean, notY2mean], [X3mean, Y3mean],[notX3mean, notY3mean])
    pred_label =np.zeros(np.size(xy,0))
    for k in range(np.size(xy,0)):
        if dist_mat[k][0]<=dist_mat[k][1] and dist_mat[k][3]<=dist_mat[k][2] and dist_mat[k][5]<=dist_mat[k][4]:
            pred_label[k]=1
        elif dist_mat[k][2]<=dist_mat[k][3] and dist_mat[k][1]<=dist_mat[k][0] and dist_mat[k][5]<=dist_mat[k][4]:
            pred_label[k] = 2
        elif dist_mat[k][4] <= dist_mat[k][5] and dist_mat[k][1] <= dist_mat[k][0] and dist_mat[k][3] <= dist_mat[k][2]:
            pred_label[k] = 3

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    plt.plot(training[label_train == 1,0], training[label_train == 1,1], 'rx')
    plt.plot(training[label_train == 2,0], training[label_train == 2,1], 'go')
    plt.plot(training[label_train == 3,0], training[label_train == 3,1], 'b*')

    # include legend for training data

    l = plt.legend(('Class 1', 'Class 2', 'Class 3','indeterminant region'), loc=2)

    plt.gca().add_artist(l)

    # plot the class mean vector.
    m1, = plt.plot(sample_mean[0][0], sample_mean[0][1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(sample_mean[2][0], sample_mean[2][1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    m3, = plt.plot(sample_mean[4][0], sample_mean[4][1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    # include legend for class mean vector

    l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)

    plt.gca().add_artist(l1)

    plt.show()


