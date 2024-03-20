############################################
# Assignment 1                             #
# Tristan PERROT                           #
# March 2024                               #
############################################

###### Utils ######
import numpy as np
np.random.seed(42)


DATASET_PATH = 'Dataset/'


def load_batch(filename):
    """ Copied from the dataset website """
    import pickle
    with open(DATASET_PATH + filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    fig.suptitle("Montage of the dataset")
    for i in range(2):
        for j in range(5):
            im = W[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def compute_grads_num(X, Y, P, W, b, lmbda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = compute_cost(X, Y, W, b, lmbda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lmbda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost(X, Y, W_try, b, lmbda)
            grad_W[i, j] = (c2-c) / h

    return [grad_W, grad_b]


def compute_grads_num_slow(X, Y, P, W, b, lmbda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = compute_cost(X, Y, W, b_try, lmbda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lmbda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = compute_cost(X, Y, W_try, b, lmbda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost(X, Y, W_try, b, lmbda)

            grad_W[i, j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]

###### Main ######
# Load the data
# data = load_batch('data_batch_1')
# montage(data[b'data'])

# Exercise 1: Training a multi-linear classifier
# 1.1: Read in and store the training, validation and test data.


def load_data(filename):
    """ Read the data from the file """
    data = load_batch(filename)
    X = data[b'data'].T / 255
    y = np.array(data[b'labels'])
    Y = np.zeros((10, X.shape[1]))
    for i in range(y.shape[0]):
        Y[y[i], i] = 1
    return X, Y, y


def split_data(X, Y, y, split_ratio=0.8):
    """ Split the data into a training set and a validation set """
    n = X.shape[1]
    n_train = int(n*split_ratio)
    X_train = X[:, :n_train]
    Y_train = Y[:, :n_train]
    y_train = y[:n_train]
    X_val = X[:, n_train:]
    Y_val = Y[:, n_train:]
    y_val = y[n_train:]
    return X_train, Y_train, y_train, X_val, Y_val, y_val


def save_data(filename, X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test):
    """ Save the data to a file """
    np.savez(DATASET_PATH+filename, X_train=X_train, Y_train=Y_train, y_train=y_train,
             X_val=X_val, Y_val=Y_val, y_val=y_val, X_test=X_test, Y_test=Y_test, y_test=y_test)


def read_data(filename):
    """ Read the data from a file """
    data = np.load(DATASET_PATH+filename+'.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    y_train = data['y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    y_val = data['y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']
    y_test = data['y_test']
    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test


X, Y, y = load_data('data_batch_1')
X_test, Y_test, y_test = load_data('test_batch')
X_train, Y_train, y_train, X_val, Y_val, y_val = split_data(X, Y, y)
save_data('data_batch_1', X_train, Y_train, y_train,
          X_val, Y_val, y_val, X_test, Y_test, y_test)
X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = read_data(
    'data_batch_1')
print("1.1: Read and stored the training data")
print("X.shape:", X.shape, "; Y.shape:", Y.shape, "; y.shape:", y.shape)

# 1.2: Compute the mean and standard deviation vector for the training data and then normalize the training, validation and test data w.r.t. these mean and standard deviation vectors
mean_X = np.mean(X_train, axis=1).reshape(-1, 1)
std_X = np.std(X_train, axis=1).reshape(-1, 1)
X_train = (X_train-mean_X) / std_X
X_val = (X_val-mean_X) / std_X
X_test = (X_test-mean_X) / std_X
print("\n1.2: Normalized the data")

# 1.3: Initialize the parameters


def initialize_parameters(K, d):
    """ Initialize the parameters """
    W = np.random.normal(0, 0.01, (K, d))
    b = np.random.normal(0, 0.01, (K, 1))
    return W, b


W, b = initialize_parameters(Y_train.shape[0], X_train.shape[0])
print("\n1.3: Initialized the parameters")
print("W.shape:", W.shape, "; b.shape:", b.shape)

# 1.4: Evaluate network function


def evaluate_classifier(X, W, b):
    """ Evaluate the network function """
    return softmax(W @ X + b)


P = evaluate_classifier(X_train[:, :20], W, b)
print("\n1.4: Evaluated the network function")
print("P.shape:", P.shape)

# 1.5: Compute the cost function


def compute_cost(X, Y, W, b, lmbda):
    """ Compute the cost function """
    n = X.shape[1]
    P = evaluate_classifier(X, W, b)
    cross_entropy = -np.log(np.sum(Y * P, axis=0))
    return np.sum(cross_entropy) / n + lmbda * np.sum(W**2)

# 1.6: Compute the accuracy


def compute_accuracy(X, y, W, b):
    """ Compute the accuracy of the classifier """
    P = evaluate_classifier(X, W, b)
    return np.sum(np.argmax(P, axis=0) == y) / X.shape[1]

# 1.7: Compute the gradients of the cost function


def compute_gradients(X, Y, P, W, b, lmbda):
    """ Compute the gradients of the cost function """
    n = X.shape[1]
    G = -(Y - P)
    grad_W = G @ X.T / n + 2 * lmbda * W
    grad_b = np.sum(G, axis=1).reshape(-1, 1) / n
    return grad_W, grad_b


def compute_relative_error(grad_analytical, grad_numerical, eps=1e-9):
    """ Compute the relative error between the analytical and numerical gradients """
    return np.max(np.abs(grad_analytical - grad_numerical) / np.maximum(eps, np.abs(grad_analytical) + np.abs(grad_numerical)))


# Comparing the gradients
n = 20
dim = 2
X_train_reduced = X_train[:dim, :n]
Y_train_reduced = Y_train[:, :n]
W_reduced = W[:, :dim]
b_reduced = b
lmbda = 0
P = evaluate_classifier(X_train_reduced, W_reduced, b_reduced)
grad_W, grad_b = compute_gradients(
    X_train_reduced, Y_train_reduced, P, W_reduced, b_reduced, lmbda)
grad_W_num, grad_b_num = compute_grads_num(
    X_train_reduced, Y_train_reduced, P, W_reduced, b_reduced, lmbda, 1e-6)
grad_W_num_slow, grad_b_num_slow = compute_grads_num_slow(
    X_train_reduced, Y_train_reduced, P, W_reduced, b_reduced, lmbda, 1e-6)
print("\n1.7: Computed the gradients")
print("grad_W.shape:", grad_W.shape, "; grad_b.shape:", grad_b.shape)
print("grad_W_num.shape:", grad_W_num.shape,
      "; grad_b_num.shape:", grad_b_num.shape)
print("grad_W_num_slow.shape:", grad_W_num_slow.shape,
      "; grad_b_num_slow.shape:", grad_b_num_slow.shape)
print("Relative error grad_W:", compute_relative_error(grad_W, grad_W_num))
print("Relative error grad_b:", compute_relative_error(grad_b, grad_b_num))
print("Relative error grad_W_slow:",
      compute_relative_error(grad_W, grad_W_num_slow))
print("Relative error grad_b_slow:",
      compute_relative_error(grad_b, grad_b_num_slow))
