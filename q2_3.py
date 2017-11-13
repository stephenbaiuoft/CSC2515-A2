'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    # k means class type, j means feature dimension
    for k in range(10):
        # count number of datums of the k class
        k_data = train_data[train_labels == k]
        # based on derivation
        eta[k] = (np.sum(k_data, axis=0) + 1) / (k_data.shape[0] * 3)
        #eta[k] = np.sum(k_data, axis=0)/(k_data.shape[0]*3) + 1/3
        print("the shape is: ", eta[k].shape)

    print("eta shape is: ", eta.shape)
    return eta


def plot_images(class_images):
    '''
    Plot each of the image eta corresponding to each class side by side in grayscale
    '''
    img = []
    for i in range(10):
        img_i = class_images[i].reshape(8, 8)
        img.append(img_i)

    all_concat = np.concatenate(img, axis=1)
    plt.imshow(all_concat, cmap='gray')
    #plt.title('10 class eta side by side graph')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))

    for k in range(10):
        data_k = np.random.binomial(1, eta[k])
        generated_data[k] = data_k

    plot_images(generated_data)


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    likelihood_set = generative_likelihood_probability(bin_digits, eta)
    log_likelihood_set = np.log(likelihood_set)

    return log_likelihood_set


def generative_likelihood_probability(bin_digits, eta):
    '''
    Compute the generative likelihood:
             p(x|y, eta)
    Should return an n x 10 numpy array
    '''

    # eta: 10x64, bin_digits: nx64
    # 10 classes

    # Result ==> n x 10
    n = bin_digits.shape[0]
    likelihood_set = np.zeros((n, 10))
    for i in range(n):
        for k in range(10):
            likelihood_v = np.power(eta[k], bin_digits[i]) * \
                         np.power(1-eta[k], 1-bin_digits[i])

            likelihood = np.product(likelihood_v)
            likelihood_set[i][k] = likelihood

    return likelihood_set


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    # n x 10 ==>
    generative_likelihood_set = generative_likelihood_probability(bin_digits, eta)
    conditional_likelihood_probability = generative_likelihood_set\
                                         /np.sum(generative_likelihood_set,
                                                 axis=1, keepdims=True)

    log_c_l_p = np.log(conditional_likelihood_probability)
    return log_c_l_p


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_log_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    tot_probability = 0
    for i in range(10):
        # get conditional likelihood n x 10
        # of each i and sum them up
        matching_index = labels == i
        cond_log_likelihood_i = cond_log_likelihood[matching_index]

        # take the ith_col
        cond_log_likelihood_i_col = cond_log_likelihood_i[:, i]
        i_tot = np.sum(cond_log_likelihood_i_col)

        tot_probability += i_tot
    print("tot_probability: ", tot_probability)
    # Compute as described above and return
    # over all digits
    return tot_probability/bin_digits.shape[0]


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class

    cond_likelihood = conditional_likelihood(bin_digits, eta)

    posterior_class_prediction = np.argmax(cond_likelihood, axis=1)
    return posterior_class_prediction


def posterior_accuracy(bin_digits, labels, eta):
    posterior_class_prediction = classify_data(bin_digits, eta)
    correct_count = posterior_class_prediction[posterior_class_prediction
                                               == labels].shape[0]
    accuracy = correct_count / bin_digits.shape[0]
    # print("posterior accuracy is: ", accuracy)
    return accuracy

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    # 2.3.2
    eta = compute_parameters(train_data, train_labels)

    # Evaluation 2.3.3
    # plot_images(eta)

    generate_new_data(eta)

    # avg_cond_likelihood = avg_conditional_likelihood(train_data,
    #                                                  train_labels, eta)
    #
    # print("avg_cond_likelihood train is:{}, probability: {} ".format(avg_cond_likelihood,
    #                                              np.exp(avg_cond_likelihood)))
    #
    # avg_cond_likelihood = avg_conditional_likelihood(test_data,
    #                                                  test_labels, eta)
    #
    # print("avg_cond_likelihood test is:{}, probability: {} ".format(avg_cond_likelihood,
    #                                                    np.exp(avg_cond_likelihood)))
    #
    # # test posterior class accuracy
    # train_posterior_accuracy = posterior_accuracy(train_data, train_labels, eta)
    # print("train posterior_accuracy is: ", train_posterior_accuracy)
    #
    # test_posterior_accuracy = posterior_accuracy(test_data, test_labels, eta)
    # print("test posterior_accuracy is: ", test_posterior_accuracy)


if __name__ == '__main__':
    main()
