#!/usr/bin/env python

import numpy as np
import scipy, scipy.io, scipy.optimize

# Globals used by the objective function
global features, labels, Y, num_samples, feature_size, num_classes

def objective(w):
    """The main objective function to minimize
    """

    global features, labels, Y, num_samples, feature_size, num_classes
    num_classes = 10
    C = 100.0

    weights = np.matrix(np.reshape(w, (feature_size, num_classes)))
    
    # 0.5 * ||w||^2 + C\sum_epsilon
    value = np.maximum(0.0, 1.0 - np.multiply(Y, (features * weights)))
    t1 = 0.5 * np.sum(np.power(weights, 2), 0)
    t2 = C * np.mean(np.power(value, 2), 0)

    return np.sum(t1 + t2)

def objective_derivative(w):
    """The derivative of the objective function - used to guide the optimization
    algorithm.
    """
    global features, labels, Y, num_samples, feature_size, num_classes
    num_classes = 10
    C = 100.0

    weights = np.matrix(np.reshape(w, (feature_size, num_classes)))

    value = np.maximum(0.0, 1.0 - np.multiply(Y, (features * weights)))
    gradient = weights - (2.0*C/num_samples) * (features.transpose() * np.multiply(value, Y))

    return gradient.flatten().transpose()

def main():
    """Calculates the weights that solve the SVM perfectly
    """
    global features, labels, Y, num_samples, feature_size, num_classes
    mat = scipy.io.loadmat('./training.mat')
    features = np.matrix(mat['trainXCs'])
    labels = mat['trainY'].flatten()
    (num_samples, feature_size) = features.shape
    num_classes = 10

    Y = np.zeros( (num_samples, num_classes) )
    Y = Y - 1 
    for i in xrange(num_classes):
        # Matlab indices start from 1 - but i iterates from 0 to 9
        Y[labels==i+1, i] = 1

    (num_samples, feature_size) = features.shape
    num_classes = 10

    weights = np.zeros( (feature_size*num_classes, 1) )
    weights_final = scipy.optimize.minimize(objective, weights, method='L-BFGS-B', jac=objective_derivative, options={'disp': True, 'maxiter': 500})
    print weights_final

    scipy.io.savemat('solved_weights.mat', {'svm': np.reshape(weights_final.x, (feature_size, num_classes) )}, do_compression=True)

    return

if __name__ == "__main__":
    main()
