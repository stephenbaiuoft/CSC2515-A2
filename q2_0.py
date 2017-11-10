'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i, num x 64
        i_means = np.mean(i_digits, axis=0).reshape(8, 8)
        # print(i_means.shape)
        means.append(i_means)

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    # 8x80, and 80 ==> horizontal expanding
    # print(all_concat.shape)
    plt.imshow(all_concat, cmap='gray')
    plt.title('10 means side by side graph')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)
