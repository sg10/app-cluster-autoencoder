# has been removed from keras in v2,
# taken from https://github.com/keras-team/keras/issues/5400

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support

from appclusters.util.inspect import copy_func


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def get_fbeta_micro(beta=1.0):

    def fbeta_micro(y_true, y_pred):
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)

        return (1+beta*beta) * (p * r) / ((beta*beta*p) + r + K.epsilon())

    return fbeta_micro


def get_fbeta_macro(beta):

    def fbeta_macro(y_true, y_pred):
        # https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
        y_pred = K.round(y_pred)

        true_positives = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        false_positives = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        false_negatives = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = true_positives / (true_positives + false_positives + K.epsilon())
        r = true_positives / (true_positives + false_negatives + K.epsilon())

        f1 = (1+beta*beta) * p * r / ((beta*beta*p) + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

        return K.mean(f1)

    return fbeta_macro


def fb_micro(y_true, y_pred):
    return get_fbeta_micro(0.5)(y_true, y_pred)


def fb_macro(y_true, y_pred):
    return get_fbeta_macro(0.5)(y_true, y_pred)



# set in a way so that the median number of values per samples larger than the threshold
# is 10

_tfidf_discretize_threshold_descriptions = 0.14


def tfidf_accuracy(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    y_true_gt0_mask = tf.greater(y_true, threshold)
    y_pred_gt0_mask = tf.greater(y_pred, threshold)

    and_mask = tf.logical_and(y_true_gt0_mask, y_pred_gt0_mask)

    ones_correct = tf.boolean_mask(tf.ones(tf.shape(y_true)), and_mask)
    num_correct = tf.size(ones_correct)

    ones_true = tf.boolean_mask(tf.ones(tf.shape(y_true)), y_true_gt0_mask)
    num_true = tf.size(ones_true)

    return np.true_divide(num_correct, num_true)


def tfidf_metrics(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    y_true_gt0_mask = tf.greater(y_true, threshold)
    y_pred_gt0_mask = tf.greater(y_pred, threshold)

    # reduce both to the elements where either one or the other is > threshold
    mask_both = tf.logical_or(y_true_gt0_mask, y_pred_gt0_mask)

    y_true_relevant = tf.boolean_mask(y_true, mask_both)
    y_pred_relevant = tf.boolean_mask(y_pred, mask_both)

    y_true_relevant_gt0_mask = tf.greater(y_true_relevant, threshold)
    y_pred_relevant_gt0_mask = tf.greater(y_pred_relevant, threshold)

    y_true_relevant_binary = tf.cast(y_true_relevant_gt0_mask, dtype=tf.int32)
    y_pred_relevant_binary = tf.cast(y_pred_relevant_gt0_mask, dtype=tf.int32)

    correct_true_positives = tf.reduce_sum(tf.multiply(y_true_relevant_binary, y_pred_relevant_binary))
    pred_positives = tf.reduce_sum(y_pred_relevant_binary)
    true_positives = tf.reduce_sum(y_true_relevant_binary)

    precision = tf.clip_by_value(np.true_divide(correct_true_positives, pred_positives), 0.0, 1.0)
    recall = np.true_divide(correct_true_positives, true_positives)
    f1_score = 2 * tf.divide(tf.multiply(precision, recall), tf.add(precision, recall))

    return precision, recall, f1_score


def tfidf_precision(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[0]


def tfidf_recall(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[1]


def tfidf_f1(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[2]


def get_tfidf_parameterized(beta, threshold):

    def tfidf_fbeta(y_true, y_pred):
        p, r, _ = tfidf_metrics(y_true, y_pred, threshold)
        fb = (1 + beta * beta) * p * r / ((beta * beta * p) + r + K.epsilon())
        return fb

    def tfidf_precision(y_true, y_pred):
        return tfidf_metrics(y_true, y_pred, threshold)[0]

    def tfidf_recall(y_true, y_pred):
        return tfidf_metrics(y_true, y_pred, threshold)[1]

    th_str = ("%.4f" % threshold).strip("0").strip(".")

    return [copy_func(tfidf_fbeta, "f_%s" % th_str),
            copy_func(tfidf_precision, "pr_%s" % th_str),
            copy_func(tfidf_recall, "rc_%s" % th_str)]

