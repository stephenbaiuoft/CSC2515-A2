'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        l2_ary = self.l2_distance(test_point)
        k_indice = np.argpartition(l2_ary, k)[:k]
        # bin_count to process train_labels -> get argmax index -> digit
        digit = np.bincount(self.train_labels[k_indice].astype(int)).argmax()
        return digit


def cross_validation(knn, k_range=np.arange(1, 16)):
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    correct_count = 0
    total_count = eval_data.shape[0]
    for i in range(total_count):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == eval_labels[i]:
            correct_count += 1

    return correct_count/total_count


def part_2_1_1(knn, train_data, train_labels, test_data, test_labels):
    accuracy_train_1 = classification_accuracy(knn, 1, train_data, train_labels)
    accuracy_train_15 = classification_accuracy(knn, 15, train_data, train_labels)

    accuracy_test_1 = classification_accuracy(knn, 1, test_data, test_labels)
    accuracy_test_15 = classification_accuracy(knn, 15, test_data, test_labels)

    print("accuracy for Train with k = 1: {}, k = 15: {}".format(accuracy_train_1,
                                                                 accuracy_train_15))
    print("accuracy for Test with k = 1: {}, k = 15: {}".format(accuracy_test_1,
                                                                 accuracy_test_15))

# find the optimal k
def part_2_1_3(X, Y, test_data, test_labels):
    # 1 - 15
    k_max = 16
    kf = KFold(n_splits=10)

    k_cross_accuracy_set = []
    for i in range(1, 16):
        # compute cross validation error for and get the index?
        cross_accuracy = 0.0
        for train_index, test_index in kf.split(X):
            x_train_set, x_test_set = X[train_index], X[test_index]
            y_train_set, y_test_set = Y[train_index], Y[test_index]

            knn = KNearestNeighbor(x_train_set, y_train_set)
            test_accuracy = \
                classification_accuracy(knn, i, x_test_set, y_test_set)

            cross_accuracy += test_accuracy
        cross_accuracy /= 10

        # add to k_cross_accracy_Set
        k_cross_accuracy_set.append(cross_accuracy)

    # remember the one offset
    print("k_cross_accuracy_set is:\n", k_cross_accuracy_set)
    optimal_k = np.argmax(k_cross_accuracy_set) + 1
    print("optimal_k is: ", optimal_k)

    # now compute the entire train data set
    knn = KNearestNeighbor(X, Y)
    train_accuracy = classification_accuracy(knn, optimal_k, X, Y)
    test_accuracy = classification_accuracy(knn, optimal_k, test_data, test_labels)
    # avg_cross_fold is accuracy across 1- 16 fold for each cross_validation?
    avg_cross_fold = np.sum(k_cross_accuracy_set)/16

    print("train_accuracy is {}\ntest_accuracy is{}\navg_cross_fold is{}".
          format(train_accuracy, test_accuracy, avg_cross_fold))



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    part_2_1_1(knn, train_data, train_labels, test_data, test_labels)

    # part 2.1.3
    part_2_1_3(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()