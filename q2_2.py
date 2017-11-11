'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import data

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class => computing mu matrix, 10x64

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''

    # lec_9 multivariate gaussian MLE
    means = []
    # digit_k for each digit k
    for i in range(10):
        # k_
        k_digits = data.get_digits_by_label(train_data, train_labels, i)
        k_count = k_digits.shape[0]
        k_mean = np.sum(k_digits, axis=0)/k_count
        means.append(k_mean)

    means = np.stack(means, axis=0)
    print("means shape is: ", means.shape)
    # 10 x 64 =>
    # Compute means
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class => 10x64x64

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''

    # issue: covariance is wrong: matrix multiplication is not right
    covariance_matrix_set = []
    i_matrix = np.identity(64, dtype=float) * 0.01
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        k_digits = data.get_digits_by_label(train_data, train_labels, i)
        k_count = k_digits.shape[0]
        k_mean = means[i]
        d_set = (k_digits - k_mean)

        covariance_matrix_i_sum = i_matrix
        for d in d_set:
            # d should be (64,) ==> to (64,1)
            d = d.reshape(64, 1)
            # covariance_d: (64,64)
            covariance_d = np.dot(d, np.transpose(d))
            # print("covariance_d shape is: ", covariance_d.shape)
            # adding over all data points
            covariance_matrix_i_sum += covariance_d
        covariance_matrix_i = covariance_matrix_i_sum / k_count
        # add to set
        covariance_matrix_set.append(covariance_matrix_i)

    all_stack = np.stack(covariance_matrix_set, axis=0)
    # covariances = np.zeros((10, 64, 64))
    print("covariance_matrix shape is: ", all_stack.shape)
    # Compute covariances
    return all_stack


def plot_cov_diagonal(covariances):
    cov_diag_set = []
    # Plot the diagonal of each covariance matrix side by side

    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # shape of 64?
        log_cov_diag = np.log(cov_diag).reshape(8, 8)
        cov_diag_set.append(log_cov_diag)

    all_concat = np.concatenate(cov_diag_set, 1)
    # side by side?
    plt.imshow(all_concat, cmap='gray')
    plt.title('10 cov_diag side by side graph')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    digits: n x 64   | 10 comes from the set of labels
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    likelihood_set = generative_likelihood_probability(digits, means, covariances)
    # return the log-likelihood now
    return np.log(likelihood_set)


def generative_likelihood_probability(digits, means, covariances):
    '''
    digits: n x 64   | 10 comes from the set of labels
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array in terms of probability
    '''
    # for storing digit likelihood 0 - 9
    likelihood_set = []
    n = digits.shape[0]
    for i in range(10):
        cov_inv = np.linalg.inv(covariances[i])
        det = np.linalg.det(covariances[i])
        # shape of (n, 64)
        digits_diff = digits - means[i]
        # n is number of digits
        n_by_n_matrix = np.dot(np.dot(digits_diff, cov_inv),
                               np.transpose(digits_diff))
        # shape (n,)
        n_by_one_tmp = np.diag(n_by_n_matrix)
        n_by_one = n_by_one_tmp.reshape(n, 1)

        # d is 64!! # of x dimension!!!
        term_1 = np.float_power(2*np.pi, -64/2)
        term_2 = np.float_power(det, -0.5)
        term_3 = np.exp(-0.5 * n_by_one)

        p_i = term_1 * term_2 * term_3
        likelihood_set.append(p_i)

    all_concat = np.concatenate(likelihood_set, axis=1)
    # should be shape of nx10
    print("generative_likelihood_probability shape: ", all_concat.shape)
    return all_concat


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    g_likelihood_set = generative_likelihood_probability(digits, means, covariances)
    # for each class of y
    # n x 10, ==> summing over g_likelihood_set, axis = 1 (n x 10 -> n x 1)
    # keeping the dimension
    conditional_likelihood_probability =\
        g_likelihood_set/np.sum(g_likelihood_set, axis=1, keepdims=True)
    # return the log likelihood now
    return np.log(conditional_likelihood_probability)


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_log_likelihood = conditional_likelihood(digits, means, covariances)
    tot_probability = 0
    for i in range(10):
        # get conditional likelihood n x 10
        # of each i and sum them up
        tot_probability += np.sum(cond_log_likelihood[labels == i])

    # Compute as described above and return
    # over all digits
    return tot_probability/digits.shape[0]


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass


# draw cov_diagonal side by side
def part_2_2_1(covariances):
    #2.2.1
    plot_cov_diagonal(covariances=covariances)


# get likelihood on train and test data
def part_2_2_2(train_data, train_labels, test_data, test_labels):

    train_means = compute_mean_mles(train_data, train_labels)
    train_covariances = compute_sigma_mles(train_data, train_labels)
    train_avg_likelihood = avg_conditional_likelihood(digits=train_data, labels=train_labels,
                                                      means=train_means,
                                                      covariances=train_covariances)
    print("train_avg_likelihood is: ", train_avg_likelihood)

    test_means = compute_mean_mles(test_data, test_labels)
    test_covariances = compute_sigma_mles(train_data, train_labels)
    test_avg_likelihood = avg_conditional_likelihood(digits=test_data, labels=test_labels,
                                                     means=test_means,
                                                     covariances=test_covariances)
    print("test_avg_likelihood is: ", test_avg_likelihood)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # part_2_2_1(covariances)


    #2.2.2
    part_2_2_2(train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()