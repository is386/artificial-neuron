from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

YALE_PATH = "./yalefaces/"
IMAGE_X = 40
IMAGE_Y = 40
SEED = 11519991

# Hyper Parameters
LEARNING_RATE = 0.0001
TERM_CRITERIA = 100
REG_TERM = 0.5
BIAS = 1
BATCH_SIZE = 8

classes = {}


def parse_yale_faces():
    """
    Parses the yale faces data.

    :return: `np.ndarray` the image data
    """
    data_matrix = []
    yale_faces = [i for i in listdir(
        YALE_PATH) if isfile(join(YALE_PATH, i))]

    for face in yale_faces:
        try:
            face_img = Image.open(join(YALE_PATH, face))
            face_img = face_img.resize((IMAGE_X, IMAGE_Y))
            pixels = np.asarray(face_img).flatten()
            pixels = np.insert(pixels, 0, BIAS)
            face_img.close()
            sub_n = parse_subj_num(face)
            pixels = np.append(pixels, sub_n)
            data_matrix.append(pixels)

            # saves each class and its total
            if sub_n not in classes:
                classes[sub_n] = 1
            else:
                classes[sub_n] += 1
        except OSError:
            pass

    return np.asarray(data_matrix)


def split_data(data_mat):
    """
    Splits the overall data set into 2/3 training and 1/3 testing.

    :param data_mat: `np.ndarray` the data from the images
    :return: `np.ndarray` the training data
    :return: `np.ndarray` the training labels
    :return: `np.ndarray` the testing data
    :return: `np.ndarray` the testing labels
    """
    training = np.zeros((1, data_mat.shape[1]))
    testing = np.zeros((1, data_mat.shape[1]))

    for i in classes.keys():
        num_training = 2 * round(classes[i] / 3)
        # gets all of the data for each subject, i
        subj_mat = data_mat[data_mat[:, -1] == i]
        # shuffles that subject's data to reduce bias
        np.random.shuffle(subj_mat)
        # adds 2/3s of the data to training, and 1/3 to testing
        training = np.append(training, subj_mat[0:num_training, :], axis=0)
        testing = np.append(testing, subj_mat[num_training:, :], axis=0)

    # reshuffle to reduce bias
    np.random.shuffle(training)
    np.random.shuffle(testing)
    # labels are in the last column of the data
    return training[1:, :-1], training[1:, -1], testing[1:, :-1], testing[1:, -1]


def parse_subj_num(subject):
    """
    Parses the subject number from the subject name.

    :param subject: `str` the subject name
    :return: `int` the subject number
    """
    return int("".join(subject.split("subject")[1][0:2]))


def activation(X, theta):
    """
    Computes the logistic activation.

    :param X: `numpy.ndarray` the data
    :param theta: `numpy.ndarray` weight values
    :return: `np.ndarray` y_hat
    """
    z = X @ theta
    return 1 / (1 + (np.exp(-1 * z)))


def objective(y, y_hat, theta):
    """
    Computes the log likelihood.

    :param y: `numpy.ndarray` the label
    :param y_hat: `numpy.ndarray` the computation of the activation function
    :param theta: `numpy.ndarray` weight values
    :return: `np.ndarray` the log likelihoods
    """
    return (y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))


def gradient(x, y, y_hat, theta):
    """
    Computes the gradient.

    :param x: `numpy.ndarray` the data
    :param y: `numpy.ndarray` the label
    :param y_hat: `numpy.ndarray` the computation of the activation function
    :param theta: `numpy.ndarray` weight values
    :return: `np.ndarray` the gradients
    """
    l2 = 2 * (REG_TERM * theta)
    return (x.T @ (y - y_hat)) + l2


def train_network(train_X, train_Y, weights):
    """
    Trains the weights using batch gradient descent until the termination criteria is reached.

    :param train_X: `numpy.ndarray` all of the training data
    :param train_Y: `numpy.ndarray` all of the training labels
    :param weights: `numpy.ndarray` untrained/randomized weights
    :return: `np.ndarray` the trained weights
    :return: `np.ndarray` the average log likelihoods per iteration
    """
    num_batches = len(train_X) // BATCH_SIZE
    avg_J = []
    for i in range(TERM_CRITERIA):
        J = np.asarray([])

        for i in range(num_batches - 1):
            # if the batch split is uneven, just take the rest of the data at the end
            if i == BATCH_SIZE:
                batch_data = train_X[i * BATCH_SIZE:, :]
                batch_labels = train_Y[i * BATCH_SIZE:]
            else:
                batch_data = train_X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
                batch_labels = train_Y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            weights, j = train_weights(
                batch_data, batch_labels, weights)
            J = np.append(J, j)

        np.nan_to_num(J, 0)
        avg_J.append(J.mean())
    return weights, np.asarray(avg_J)


def train_weights(data, labels, weights):
    """
    Trains the weights with the given data.

    :param data: `numpy.ndarray` a batch of training data
    :param labels: `numpy.ndarray` a batch of corresponding test labels
    :param weights: `numpy.ndarray` trained weights
    :return: `np.ndarray` the average
    :return: `float` average log likelihood
    """
    y_hat = activation(data, weights)
    grad = LEARNING_RATE * gradient(data, labels, y_hat, weights)
    j = objective(labels, y_hat, weights)
    return weights + grad, j.mean()


def standardize(train_X, test_X):
    """
    Standardizes the training and test data.

    :param train_X: `numpy.ndarray` training data
    :param train_Y: `numpy.ndarray` test data
    """
    for i in range(1, train_X.shape[1]):
        s = np.std(train_X[:, i])
        m = np.mean(train_X[:, i])
        # if the std is 0, then set that feature to 0
        if s == 0:
            train_X[:, i] = 0
        train_X[:, i] = (train_X[:, i] - m) / s
        test_X[:, i] = (test_X[:, i] - m) / s


def create_labels(Y):
    """
    Takes the given labels and converts them to array of 0 and 1
    0 means it is not that class, and 1 means that is the correct class

    :param Y: `numpy.ndarray` labels
    :return: `numpy.ndarray` the multi class labels
    """
    labels = []
    for i in range(len(Y)):
        label = np.zeros((1, 15))
        # gets the subject number
        val = int(Y[i])
        # puts a one at the index representing the subject number
        label[0, val - 1] = 1
        labels.append(label)
    return np.asarray(labels).squeeze()[:, 1:]


def test_network(test_X, test_Y, weights):
    """
    Tests the given test data with the trained weights.

    :param test_X: `numpy.ndarray` test data
    :param test_Y: `numpy.ndarray` test labels
    :param weights: `numpy.ndarray` trained weights
    :return: `float` the average
    :return: `numpy.ndarray` the confusion matrix
    """
    correct = 0
    confuse_mat = np.zeros((len(classes), len(classes)))
    for i, t in enumerate(test_X):
        y_hat = activation(t, weights)
        label = test_Y[i]
        # gets the index where the label contains a 1. represents the subject
        actual = np.where(label == 1)[0][0]
        # gets the index where the label contains a 1. represents most likely subject
        guess = np.where(y_hat == np.amax(y_hat))[0][0]
        confuse_mat[guess][actual] += 1
        # get the index of the position that has a max, then add 2 to get the subject number
        if int(actual) + 2 == guess + 2:
            correct += 1
    return (correct / len(test_Y)) * 100, confuse_mat


def plot_avg_j(avg_J):
    """
    Plots the average log likelihood

    :param avg_J: `numpy.ndarray` test data
    """
    plt.plot(range(len(avg_J)), avg_J)
    plt.xlabel("Iterations")
    plt.ylabel("Average Log Likelihood")
    plt.savefig('log.png', bbox_inches='tight')
    plt.close()


def plot_confuse(confuse_mat):
    """
    Plots the confusion matrix

    :param confuse_mat: `numpy.ndarray` confusion matrix
    """
    df_cm = pd.DataFrame(
        confuse_mat,
        index=[i for i in range(2, 16)],
        columns=[i for i in range(2, 16)]
    )
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confuse.png', bbox_inches='tight')
    plt.close()


def main():
    np.random.seed(SEED)
    data_mat = parse_yale_faces()
    train_X, train_Y, test_X, test_Y = split_data(data_mat)
    standardize(train_X, test_X)
    train_labels = create_labels(train_Y)
    test_labels = create_labels(test_Y)

    weights = np.random.randint(0, 10, size=(
        (IMAGE_X * IMAGE_Y) + 1, len(classes)))
    weights = weights * 0.01

    weights, avg_j = train_network(train_X, train_labels, weights)
    test_accuracy, confuse_mat = test_network(test_X, test_labels, weights)
    plot_avg_j(avg_j)
    plot_confuse(confuse_mat)
    print("Testing Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
