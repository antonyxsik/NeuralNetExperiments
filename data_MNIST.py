"""
babyNN_data
~~~~~~~~~~~~~
This file loads the MNIST handwritten digits data 
and splits it into training, testing, and validation
data sets. 
"""

# Libraries
import pickle
import gzip
import numpy as np

#function to change an integer classification to a unit vector
def oneHot(ints):
    e = np.zeros((10, 1))
    e[ints] = 1
    return e

def data_loader():
    """
    Takes the MNIST dataset and returns a tuple of training, validation, 
    and testing data. 

    The training_data is a tuple, with the first entry being a numpy ndarray
    with 50,000 images. Each of these images is a numpy ndarray representing 
    the 28 * 28 = 784 pixel images. The second entry is digit (0 through 9)
    values that classify the corresponding handwritten images. 
    The validation_data and testing_data are different only in the fact that 
    they contain 10,000 images. 

    All 3 data sets are then modified to come in lists of tuples where the first 
    entry is a vector with 784 entries, and the second entry is the classification. 
    In the case of training_data, the classification is a unit vector with 
    10 entries that corresponds to the correct digit (ex: a vector containing all 
    0s except a 1 in the 3rd entry would correspond to a number 2 classification).
    In the case of the other two, the classification is simply the correct digit 
    value (integer).
    """

    #obtaining the data 
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_d, validate_d, test_d = pickle.load(f, encoding="latin1")
    f.close()

    #formatting
    train_img = [np.reshape(inputs, (784, 1)) for inputs in train_d[0]]
    train_classify = [oneHot(integers) for integers in train_d[1]]
    training_data = zip(train_img, train_classify)

    validate_img = [np.reshape(inputs, (784, 1)) for inputs in validate_d[0]]
    validate_classify = validate_d[1]
    validation_data = zip(validate_img, validate_classify)

    test_img = [np.reshape(inputs, (784, 1)) for inputs in test_d[0]]
    test_classify = test_d[1]
    test_data = zip(test_img, test_classify)

    return (training_data, validation_data, test_data)
    