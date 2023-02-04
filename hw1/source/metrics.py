import numpy as np    

def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_positives = (y_pred == y_true).sum()
    true_negatives = (y_pred == y_true).sum() 
    false_positives = (y_pred != y_true).sum()
    false_negatives = (y_pred != y_true).sum()
    
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    
    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    true_positives = (y_pred == y_true).sum()
    true_negatives = (y_pred == y_true).sum() 
    false_positives = (y_pred != y_true).sum()
    false_negatives = (y_pred != y_true).sum()
    
    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    
    return accuracy

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    pass
    