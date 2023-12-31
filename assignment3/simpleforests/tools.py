import numpy as np

def encode_one_of_n(Y):
    """
    Y: A size (n_data,) numpy vector of numeric class labels in the interval [0, n)

    Returns: A size (n_data, n) numpy matrix of class labels in a one-of-n
    encoding.

    example:
    # y = np.array([0, 2, 1])
    # np.equal.outer(y, np.arange(3))
        array([[ True, False, False],
               [False, False,  True],
               [False,  True, False]])
    """
    Y = np.equal.outer(Y, np.arange(Y.max()+1)).astype(np.float)
    assert np.all(Y.sum(axis=1) == 1.)
    return Y

def decode_one_of_n(Y):
    """
    Converts a one-of-n encoding to numeric class labels.
    """
    return max_of_n_prediction(Y)

def max_of_n_prediction(Y_prob):
    """
    Y_prob: A size (n_data, n) numpy matrix where Y_prob[i,j] is proportional to
    the probability that data point i belongs to class j.

    Returns: A size (n_data,) numpy vector Y where Y[i] is the most likely class
    for data point i, based on Y_prob.
    """
    return Y_prob.argmax(axis=1)


def max_prob_of_n_prediction(Y_prob):
    """
    Y_prob: A size (n_data, n) numpy matrix where Y_prob[i,j] is proportional to
    the probability that data point i belongs to class j.

    Returns: A size (n_data,) numpy vector Y where Y[i] is the probability of the
    most likely class for data point i, based on Y_prob.
    """
    return Y_prob.max(axis=1)


def colors_from_predictions(Y_hat, colors):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    colors: List of rgb values to assign for each class

    Returns: A size (n_data,3) numpy vector y_colors where y_colors[i] is the color
    of the class Y_hat[i]
    """
    n_colors, color_dim = colors.shape
    assert n_colors == (np.max(Y_hat) + 1)  # number colors equals number of classes

    y_colors = np.zeros((Y_hat.shape[0], color_dim))
    for yi in range(n_colors):
        y_colors[Y_hat == yi] = colors[yi]
    return y_colors


def image_from_predictions(Y_hat, Y_probs, colors, shape):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    Y_probs:  A length n numpy array where Y_probs[i] is the probability of Y_hat[i]
    colors: List of rgb values to assign for each class
    shape: 2 dimension shape of output image

    Returns: A size (shape[0],shape[1],3) numpy vector img where img[i,j] is
    the color of the class Y_hat[i*shape[1] + j] and shaded by the probability
    """
    img_flat = colors_from_predictions(Y_hat, colors)
    # darken colors by probability
    img_flat *= Y_probs.reshape(-1, 1) ** 2

    img = img_flat.reshape((shape[0], shape[1], colors.shape[1]))
    return img


