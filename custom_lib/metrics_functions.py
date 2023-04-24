import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import classification_report


def predictions_merit(predictions, labels):
    """

    :param predictions:
    :param labels:
    :return:
    """
    for index in range(labels[1].size):  # iterate all predictions within dataset
        if labels.ndim == 1:
            prediction_array = predictions[(labels > 0), 0]
        else:
            prediction_array = predictions[(labels[:, index] > 0), index]

        if index == 0:
            merit = np.expand_dims(
                np.array([np.amin(prediction_array), np.median(prediction_array), np.std(prediction_array)]), axis=0)
        else:
            merit = np.append(merit, np.expand_dims(
                np.array([np.amin(prediction_array), np.median(prediction_array), np.std(prediction_array)]), axis=0),
                              axis=0)
    return merit


def predictions_arrays(predictions, labels, class_idx):
    # create an indx array to capture img ids
    idx = np.linspace(0, labels.shape[0] - 1, num=labels.shape[0], dtype=float)
    true_class_idx = class_idx
    false_class_idx = (-1 * class_idx) + 1
    if len(labels.shape) == 1:  # Binary
        if class_idx == 0:
            prediction_array_true_class = predictions[(labels > 0), 0]
            prediction_array_false_class = np.zeros(np.count_nonzero(labels > 0))
            index_array = idx[(labels > 0)]
        else:
            prediction_array_true_class = predictions[(labels == 0), 0]
            prediction_array_false_class = np.zeros(np.count_nonzero(labels == 0))
            index_array = idx[(labels == 0)]
    else:  # 2 classes
        prediction_array_true_class = predictions[(labels[:, true_class_idx] > 0), true_class_idx]
        prediction_array_false_class = predictions[(labels[:, true_class_idx] > 0), false_class_idx]
        index_array = idx[(labels[:, class_idx] > 0)]
    return np.vstack((index_array, prediction_array_true_class * 100, prediction_array_false_class * 100)).T


# Same result as bscai.metrics.predict_metrics.cross_entropy_loss with skleanr.metrics library
def log_loss_multi_class(y_true, y_pred, class_name_list=[]):
    """
    Calculate the cross entropy (Logarithmic loss) to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.

    Returns: Dictionary with results per class

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)

    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    ce_loss = dict()

    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    if y_true.ndim > 1:
        for i in range(num_of_classes):
            ce_loss[class_name_list[i]] = log_loss(np.float64(y_true[..., i].flatten()), y_pred[..., i].flatten())
    else:
        ce_loss[class_name_list[0]] = log_loss(np.float64(y_true.flatten()), y_pred.flatten())
    return ce_loss


def accuracy_score_multi_class(y_true, y_pred, class_name_list=[], threshold=[]):
    """
    Calculate the accuracy to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.

    Returns: Dictionary with results per class

    """

    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    acc = dict()

    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if len(threshold) == num_of_classes:
            if len(threshold) == 1:
                th = threshold[0]
            else:
                th = threshold[i]
        else:
            th = 0.5
        if y_true.ndim > 1:
            acc[class_name_list[i]] = accuracy_score(y_true[..., i].flatten(), np.int64(y_pred[..., i].flatten() >= th),
                                                     normalize=True)
        else:
            acc[class_name_list[i]] = accuracy_score(y_true.flatten(), np.int64(y_pred.flatten() >= th),
                                                     normalize=True)
    return acc


def merit_score_multi_class(y_true, y_pred, class_name_list=[], plot_title='', report_path_name='',
                            threshold_best_acc=dict(), threshold_lowest_scape_rate=dict(), show_plots = True):
    """
    Calculate the merit to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.

    Returns: Dictionary with results per class

    """
    # Overall Distance: SUM for each class, lowest true score  - best Acc threshold.
    # Std / Media: make a vector with the score of the true class only. aply to vector.




    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    h_std = dict()
    h_median = dict()
    class_distance = dict()

    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    hist_bins = np.round(np.linspace(0, 1, 1000), 2)

    cumulative_min_segregation = 0.0

    for i in range(num_of_classes):

        if y_true.ndim > 1:
            y_pred_N = y_pred[y_true[..., i] == 0, i]
            y_pred_P = y_pred[y_true[..., i] == 1, i]
        else:
            y_pred_N = y_pred[y_true == 0]
            y_pred_P = y_pred[y_true == 1]

        # if i == 0: cumulative_y_pred_P = y_pred_P
        # else: cumulative_y_pred_P = np.concatenate((cumulative_y_pred_P, y_pred_P))

        # y_pred_zero_center = np.sort(np.concatenate((y_pred_N, y_pred_P - 1)))
        # std[class_name_list[i]] = np.std(y_pred_zero_center)
        # median[class_name_list[i]] = np.median(y_pred_zero_center)
        class_distance[class_name_list[i]] = np.min(y_pred_P) - np.max(y_pred_N)

        th1 = threshold_best_acc[class_name_list[i]]
        th2 = threshold_lowest_scape_rate[class_name_list[i]]

        #Adding distance histogram
        y_pred_distance_to_th1 = np.sort(np.concatenate((th1 - y_pred_N, y_pred_P - th1)))
        h_median[class_name_list[i]] = np.median(y_pred_distance_to_th1)
        h_std[class_name_list[i]] = np.std(y_pred_distance_to_th1)



        #Cumulative predictiors for positive classes
        cumulative_min_segregation += np.min(y_pred_P) - th1

        class_name = str(class_name_list[i])
        title = plot_title + class_name

        plt.hist([y_pred_N, y_pred_P], hist_bins, alpha=0.5, label=['N', 'P'], color=['g', 'r'])
        plt.plot([th1, th1], plt.ylim(), label='Threshold: ' + str(np.round(th1,4)), color='y')
        #plt.plot([th2, th2], plt.ylim(), label='best Q: ' + str(th2), color='b')
        plt.legend(loc='upper right')
        plt.title(title)
        if len(report_path_name) > 0: plt.savefig(os.path.join(report_path_name, title) + '.png', format='png', dpi=300)
        if len(plot_title) > 0 and show_plots: plt.show()
        plt.close()

    # cumulative = {
    #             'std': np.std(cumulative_y_pred_P),
    #             'median': np.median(cumulative_y_pred_P),
    #             'min_segregation': cumulative_min_segregation
    #}

    return h_std, h_median, class_distance, y_pred_N, y_pred_P


def roc_curve_multi_class(y_true, y_pred, class_name_list=[], plot_title='', report_path_name='', show_plots = True):
    """
    Calculate the ROC curve to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.

    Returns: Dictionary with results per class

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()

    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1
    for i in range(num_of_classes):
        if y_true.ndim > 1:
            fpr[class_name_list[i]], tpr[class_name_list[i]], threshold[class_name_list[i]] = roc_curve(
                y_true[..., i].flatten(), y_pred[..., i].flatten())
        else:
            fpr[class_name_list[i]], tpr[class_name_list[i]], threshold[class_name_list[i]] = roc_curve(
                y_true.flatten(), y_pred.flatten())
        roc_auc[class_name_list[i]] = auc(fpr[class_name_list[i]], tpr[class_name_list[i]])

        fpr[class_name_list[i]], tpr[class_name_list[i]], roc_auc[class_name_list[i]], class_name_list[i]
        class_name = str(class_name_list[i])
        title = plot_title + class_name
        plt.figure()
        lw = 2
        plt.plot(fpr[class_name_list[i]], tpr[class_name_list[i]], color='darkorange', lw=lw,
                 label='ROC curve (area = %0.2f)' % roc_auc[class_name_list[i]])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        if len(report_path_name) > 0: plt.savefig(os.path.join(report_path_name, title + '.png'), format='png', dpi=300)
        if len(plot_title) > 0 and show_plots: plt.show()
        plt.close()

    return fpr, tpr, threshold, roc_auc


def get_thresholds(y_true, y_pred, class_name_list=[], fpr=dict, tpr=dict, roc_threshold=dict, roc_auc=dict):
    """
    Gets the best threshold based on balanced accuracy and the best threshold based on
    lowest scape rate (i.e False Negatives)
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        fpr: Flase Positive Rate from roc_curve_multi_class function (single class)
        tpr: True Positive Rate from roc_curve_multi_class function (single class)
        roc_threshold: Trhesholds obtained by roc_curve_multi_class function

    Returns: thresholds (best overall acc, lowest scape rate)

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    threshold_best_acc = dict()
    threshold_lowest_scape_rate = dict()
    P = np.sum(y_true, axis=0)
    N = np.sum(1 - y_true, axis=0)
    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if np.ndim(P) > 0:
            P_value = P[i]
            N_value = N[i]
        else:
            P_value = P
            N_value = N
        TP = tpr[class_name_list[i]] * P_value
        TN = (1 - fpr[class_name_list[i]]) * N_value
        threshold_lowest_scape_rate[class_name_list[i]] = roc_threshold[class_name_list[i]][np.argmax(TP)]
        # If acc = 1 (i.e. full separation) then select best acc threshol is the middle point
        if roc_auc[class_name_list[i]] < 1:
            threshold_best_acc[class_name_list[i]] = roc_threshold[class_name_list[i]][np.argmax(TP + TN)]
        elif y_true.ndim > 1:
            threshold_best_acc[class_name_list[i]] = ((np.min(y_pred[y_true[..., i] > 0, i]) - np.max(
                y_pred[y_true[..., i] == 0, i])) / 2) + np.max(y_pred[y_true[..., i] == 0, i])
        else:
            threshold_best_acc[class_name_list[i]] = ((np.min(y_pred[y_true > 0, i]) - np.max(
                y_pred[y_true == 0, i])) / 2) + np.max(y_pred[y_true == 0, i])
    return threshold_best_acc, threshold_lowest_scape_rate, P, N


def precision_score_multi_class(y_true, y_pred, class_name_list=[], threshold=[]):
    """
    Calculate the precission to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        threshold =  split threshold.
    Returns: Dictionary with results per class

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    precision = dict()
    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if len(threshold) == num_of_classes:
            if len(threshold) == 1:
                th = threshold[0]
            else:
                th = threshold[i]
        else:
            th = 0.5
        if y_true.ndim > 1:
            precision[class_name_list[i]] = precision_score(y_true[..., i].flatten(),
                                                            np.int64(y_pred[..., i].flatten() >= th),
                                                            average='binary', zero_division=0)
        else:
            precision[class_name_list[i]] = precision_score(y_true.flatten(),
                                                            np.int64(y_pred.flatten() >= th),
                                                            average='binary', zero_division=0)
    return precision


def recall_score_multi_class(y_true, y_pred, class_name_list=[], threshold=0.5):
    """
    Calculate the recall to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        threshold =  split threshold.
    Returns: Dictionary with results per class

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    reacall = dict()
    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if len(threshold) == num_of_classes:
            if len(threshold) == 1:
                th = threshold[0]
            else:
                th = threshold[i]
        else:
            th = 0.5
        if y_true.ndim > 1:
            reacall[class_name_list[i]] = recall_score(y_true[..., i].flatten(),
                                                       np.int64(y_pred[..., i].flatten() >= th),
                                                       average='binary', zero_division=0)
        else:
            reacall[class_name_list[i]] = recall_score(y_true.flatten(), np.int64(y_pred.flatten() >= th),
                                                       average='binary', zero_division=0)
    return reacall


def f1_score_multi_class(y_true, y_pred, class_name_list=[], threshold=0.5):
    """
    Calculate the f1 to a multiclass clasificaiton
    using sklearn.metrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        threshold =  split threshold.
    Returns: Dictionary with results per class

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    f1 = dict()
    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if len(threshold) == num_of_classes:
            if len(threshold) == 1:
                th = threshold[0]
            else:
                th = threshold[i]
        else:
            th = 0.5
        if y_true.ndim > 1:
            f1[class_name_list[i]] = f1_score(y_true[..., i].flatten(), np.int64(y_pred[..., i].flatten() >= th),
                                              average='binary', zero_division=0)
        else:
            f1[class_name_list[i]] = f1_score(y_true.flatten(), np.int64(y_pred.flatten() >= th),
                                              average='binary', zero_division=0)
    return f1


def confussion_matrix_multi_class(y_true, y_pred, class_name_list=[], threshold=0.5):
    """
    Calculate confusion matrix
    [True Positives , False Positives]
    [False Negatives , True Negatives]
    Note that
    False Positives = False Alarm - Alpha - Type I Err
    False Negatives = Scape rate - Beta - Type II Err
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        threshold =  split threshold.
    Returns: Dictionary with confusion matrix

    """
    if y_pred.dtype != 'float64': y_pred = np.float64(y_pred)
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']
    cm = dict()
    ###########################
    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    for i in range(num_of_classes):
        if len(threshold) == num_of_classes:
            if len(threshold) == 1:
                th = threshold[0]
            else:
                th = threshold[i]
        else:
            th = 0.5
        if y_true.ndim > 1:
            cm[class_name_list[i]] = confusion_matrix(y_true[..., i].flatten(),
                                                      np.int64(y_pred[..., i].flatten() >= th))
        else:
            cm[class_name_list[i]] = confusion_matrix(y_true.flatten(), np.int64(y_pred.flatten() >= th))

    return cm


def create_prediction_test_array(num_elements=100, multi_class=True, split=0.5, num_of_FP=0, num_of_FN=0,
                                 threshold=0.5):
    """
    Generates a prediction test array for testing classifier metric functions.
    Args:
        num_elements: number of elements in array
        multi_class: if true the return array will have dimension 2. If false, dimension will be 1.
        split: [0 to 1] percent of Trues in y_true array
        num_of_FP: number of False Positives to be added to the prediction array. Set value = Threshold + 0.1
        num_of_FN: number of False Negatives to be added to the prediction arrya. Set value = Threshold - 0.1
        threshold: separation thresold use to add 10% of each class with score threshold + 0.1 and 10% with
        threshold + 0.2. The rest of predictions will be set to 1 or 0.
    Returns: y_true (int32), y_pred(float32), P, N

    """
    P = np.int32(np.round(num_elements * split))
    N = num_elements - P
    y_true = np.int32(np.concatenate([np.ones(P), np.zeros(N)]))

    y_pred = np.float64(y_true)
    # 10 % over at .1 of threshold / 10 % at 0.2 of threshold
    th_10_0 = threshold - 0.1
    th_10_1 = threshold + 0.1
    th_20_0 = threshold - 0.2
    th_20_1 = threshold + 0.2

    inc_P = np.int32(np.round(P * 0.1))
    inc_N = np.int32(np.round(N * 0.1))

    for i in range(inc_P * 2):
        if i < inc_P:
            y_pred[i] = th_10_1
        else:
            y_pred[i] = th_20_1
    for i in range(inc_N * 2):
        if i < inc_N:
            y_pred[-(i + 1)] = th_10_0
        else:
            y_pred[-(i + 1)] = th_20_0

    for i in range(inc_P * 2, inc_P * 2 + num_of_FN):
        y_pred[i] = th_10_0

    for i in range(inc_N * 2, inc_N * 2 + num_of_FP):
        y_pred[-(i + 1)] = th_10_1

    if multi_class:
        y_true = np.concatenate((np.expand_dims(y_true, axis=1), np.expand_dims(1 - y_true, axis=1)), axis=1)
        y_pred = np.concatenate((np.expand_dims(y_pred, axis=1), np.expand_dims(1 - y_pred, axis=1)), axis=1)

    return y_true, y_pred, P, N


def get_performance(TP: int, TN: int, FP: int, FN: int):
    P = TP + FN
    N = FP + TN
    acc = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    f1 = 2 * (precision * recall)/(precision + recall)
    return acc, precision, recall, f1

def metrics_multi_class(y_true, y_pred, class_name_list=[], plot_title='', report_path_name='', dataset_name='',
                        dataset_split='', model_name='', show_plots = True):
    """
    Calculates classification matrics
    Args:
        y_true: True labels array (int32) where [.., x] is the multy class index
        y_pred: Predicted labels array (float32) where [.., x] is the multy class index
        class_name_list: Optinal argument with a list of class names.
        plot_title: title to be printed in plots and to be added to the file name
        report_path_name: path name to save plots.

    Returns: metrics dictionary

    """
    if len(class_name_list) == 0:
        if y_true.ndim > 1:
            class_name_list = [('class ' + str(x)) for x in range(0, y_true.shape[-1])]
        else:
            class_name_list = ['class 0']

    if y_true.ndim > 1:
        num_of_classes = len(y_true[-1])
    else:
        num_of_classes = 1

    loss = log_loss_multi_class(y_true, y_pred, class_name_list)
    ROC_plot_title = ''
    if len(plot_title) > 0: ROC_plot_title = plot_title + '_ROC_'
    fpr, tpr, roc_threshold, roc_auc = roc_curve_multi_class(y_true, y_pred, class_name_list, plot_title=ROC_plot_title,
                                                             report_path_name=report_path_name, show_plots = show_plots)
    threshold_best_acc, threshold_lowest_scape_rate, P, N = get_thresholds(y_true, y_pred, class_name_list, fpr, tpr,
                                                                           roc_threshold, roc_auc)

    confussion_matrix_best_acc = confussion_matrix_multi_class(y_true, y_pred, class_name_list,
                                                               list(threshold_best_acc.values()))
    confussion_matrix_lowest_scape_rate = confussion_matrix_multi_class(y_true, y_pred, class_name_list,
                                                                        list(threshold_lowest_scape_rate.values()))

    acc_best_acc = accuracy_score_multi_class(y_true, y_pred, class_name_list, list(threshold_best_acc.values()))
    precision_best_acc = precision_score_multi_class(y_true, y_pred, class_name_list, list(threshold_best_acc.values()))
    recall_best_acc = recall_score_multi_class(y_true, y_pred, class_name_list, list(threshold_best_acc.values()))
    f1_best_acc = f1_score_multi_class(y_true, y_pred, class_name_list, list(threshold_best_acc.values()))

    acc_lowest_scape_rate = accuracy_score_multi_class(y_true, y_pred, class_name_list,
                                                       list(threshold_lowest_scape_rate.values()))
    precision_lowest_scape_rate = precision_score_multi_class(y_true, y_pred, class_name_list,
                                                              list(threshold_lowest_scape_rate.values()))
    recall_lowest_scape_rate = recall_score_multi_class(y_true, y_pred, class_name_list,
                                                        list(threshold_lowest_scape_rate.values()))

    f1_lowest_scape_rate = f1_score_multi_class(y_true, y_pred, class_name_list,
                                                list(threshold_lowest_scape_rate.values()))

    HIST_plot_title = ''
    if len(plot_title) > 0: HIST_plot_title = plot_title + '_HIST_'
    h_std, h_median, class_distance, y_pred_N, y_pred_P  = merit_score_multi_class(y_true, y_pred, class_name_list, plot_title=HIST_plot_title,
                                                       report_path_name=report_path_name,
                                                       threshold_best_acc=threshold_best_acc,
                                                       threshold_lowest_scape_rate=threshold_lowest_scape_rate,
                                                        show_plots=show_plots)



    confusion_matrix_array = []
    title_array = []

    if len(plot_title) > 0:

        # Best Accuracy
        for i in range(num_of_classes):
            confusion_matrix_array.append(confussion_matrix_best_acc[class_name_list[i]])
            title_array.append('BestAcc-'+class_name_list[i])
        # lowest scape rate
        for i in range(num_of_classes):
            confusion_matrix_array.append(confussion_matrix_lowest_scape_rate[class_name_list[i]])
            title_array.append('LowestSR-'+class_name_list[i])

        if len(report_path_name)>0:
            report_name_and_path = os.path.join(report_path_name, plot_title + '_CONF_matrix')
        else: report_name_and_path = ''
        print_multiclas_confusion_matrix(class_name_array=class_name_list, confusion_matrix_array=confusion_matrix_array,
                                         title_array=title_array, folder_path = report_name_and_path, show_plots = show_plots)


    metrics = dict()
    metrics['info'] = dict()
    metrics['info']['dataset_name'] = dataset_name
    if type(class_name_list) != list:
        metrics['info']['class_name_list'] = class_name_list.tolist()
    else:
        metrics['info']['class_name_list'] = class_name_list
    metrics['info']['dataset_split'] = dataset_split
    metrics['info']['dataset_samples'] = int(P[0] + N[0])
    metrics['info']['model_name'] = model_name

    for i in range(num_of_classes):
        metrics[class_name_list[i]] = dict()
        metrics[class_name_list[i]]['positive_samples'] = int(P[i])
        metrics[class_name_list[i]]['negative_samples'] = int(N[i])
        metrics[class_name_list[i]]['loss'] = loss[class_name_list[i]]
        metrics[class_name_list[i]]['fpr'] = fpr[class_name_list[i]].tolist()
        metrics[class_name_list[i]]['tpr'] = tpr[class_name_list[i]].tolist()
        metrics[class_name_list[i]]['roc_threshold'] = roc_threshold[class_name_list[i]].tolist()
        metrics[class_name_list[i]]['roc_auc'] = roc_auc[class_name_list[i]]
        metrics[class_name_list[i]]['roc_img_file'] = plot_title + '_ROC_' + class_name_list[i] + '.png'
        metrics[class_name_list[i]]['Histogram_img_file'] = plot_title + '_HIST_' + class_name_list[i] + '.png'
        metrics[class_name_list[i]]['H_Pred_N'] = y_pred_N
        metrics[class_name_list[i]]['H_Pred_P'] = y_pred_P
        metrics[class_name_list[i]]['threshold'] = threshold_best_acc[class_name_list[i]]
        metrics[class_name_list[i]]['acc'] = acc_best_acc[class_name_list[i]]
        metrics[class_name_list[i]]['precision'] = precision_best_acc[class_name_list[i]]
        metrics[class_name_list[i]]['recall'] = recall_best_acc[class_name_list[i]]
        metrics[class_name_list[i]]['confussion'] = confussion_matrix_best_acc[class_name_list[i]].tolist()
        metrics[class_name_list[i]]['f1'] = f1_best_acc[class_name_list[i]]
        metrics[class_name_list[i]]['h_std'] = h_std[class_name_list[i]]
        metrics[class_name_list[i]]['h_median'] = h_median[class_name_list[i]]
        metrics[class_name_list[i]]['class_distance'] = class_distance[class_name_list[i]]

    return metrics



def subplot_confusion_matrix(axs, classes = [''], cm = [[0,0], [0,0]], title = 'TITLE', print_corner_label = False, cmap= plt.cm.BuGn):
    cm = np.array(cm)
    cm_median = (cm.max() - cm.min()) / 2
    axs.imshow(cm, cmap=cmap)
    axs.set_title(title)
    axs.set_xticks(np.arange(len(classes)))
    axs.set_yticks(np.arange(len(classes)))
    axs.set_xticklabels(classes)
    axs.set_yticklabels(classes)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= cm_median: text_color = 'w'
            else: text_color = 'r'
            text = axs.text(j, i, cm[i, j], ha="center", va="center", color=text_color, fontsize=14)

    if print_corner_label:
        axs.annotate('PREDICTION->', (0, 0), xytext=(-120, -50),
                           textcoords='offset points', xycoords='axes fraction',
                           ha='left', va='bottom', size=14)
        axs.annotate('TRUE -->', (0, 0), xytext=(-120, 0),
                           textcoords='offset points', xycoords='axes fraction',
                           ha='left', va='bottom', size=14, rotation=90)


def print_multiclas_confusion_matrix(class_name_array= [], confusion_matrix_array = [], title_array = [], folder_path = '', show_plots = True):
    num_cols = len(class_name_array)
    num_rows = 2
    if num_cols == 1:
        class_name_array = class_name_array.tolist()
        class_name_array.append('No Deffect')
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True, squeeze=True)
    idx = 0
    print_corner_label = False
    for i in range(num_rows):
        if num_cols > 1:
            for j in range(num_cols):
                if i == 1 and j == 0:
                    print_corner_label = True
                else:
                    print_corner_label = False
                subplot_confusion_matrix(axs[i, j], classes=class_name_array, cm=confusion_matrix_array[idx], title=title_array[idx], print_corner_label= print_corner_label)
                idx += 1
        else:
            if i == 1: print_corner_label = True
            subplot_confusion_matrix(axs[i], classes=class_name_array, cm=confusion_matrix_array[idx], title=title_array[idx], print_corner_label= print_corner_label)
            idx += 1
    fig.tight_layout()
    fig.align_xlabels()
    if len(folder_path) > 0: plt.savefig(folder_path + '.png', format='png', dpi=300)
    if show_plots: fig.show()


def Xselect_outlayers(y_true, y_pred, class_name_list=[], threshold = 0, num_of_outlayers = 10):
    # for segmentation:
    # y_true = [idx, 1] - 0 or 1
    # y_pred = [idx, 1] - 0 to 1 prediction
    # For classification, same but two dimensions
    idx = np.linspace(0, y_pred.shape[0] - 1, y_pred.shape[0])
    indexed_array = np.hstack((np.expand_dims(idx, axis=1), np.atleast_2d(y_pred)))
    P = indexed_array[y_true[:,0] == 1]
    N = indexed_array[y_true[:, 0] == 0]
    P = P[P[:, 1].argsort()]
    N = np.flip(N[N[:, 1].argsort()], axis=0)

    P_outlayer_idx = P[0:num_of_outlayers - 1, 0].astype(int)
    N_outlayer_idx = N[0:num_of_outlayers - 1, 0].astype(int)

    P_outlayer_mask = np.zeros(y_pred.shape[0]).astype(bool)
    P_outlayer_mask[P_outlayer_idx] = True
    P_outlayer_mask = np.expand_dims(P_outlayer_mask, axis = 1)

    N_outlayer_mask = np.zeros(y_pred.shape[0]).astype(bool)
    N_outlayer_mask[N_outlayer_idx] = True
    N_outlayer_mask = np.expand_dims(N_outlayer_mask, axis = 1)

    mask = np.hstack((N_outlayer_mask, np.atleast_2d(P_outlayer_mask)))

    return mask


def select_outlayers(y_true, y_pred, num_of_outlayers = 10):
    # for segmentation:
    # y_true = [idx, 1] - 0 or 1
    # y_pred = [idx, 1] - 0 to 1 prediction
    # For classification, same but two dimensions
    idx = np.linspace(0, y_pred.shape[0] - 1, y_pred.shape[0])
    indexed_array = np.hstack((np.expand_dims(idx, axis=1), np.atleast_2d(y_pred)))
    for i in range(y_pred.shape[1]):


        P = indexed_array[y_true[:,i] == 1]
        N = indexed_array[y_true[:, i] == 0]
        P = P[P[:, 1].argsort()]
        N = np.flip(N[N[:, 1].argsort()], axis=0)

        P_outlayer_idx = P[0:num_of_outlayers - 1, 0].astype(int)
        N_outlayer_idx = N[0:num_of_outlayers - 1, 0].astype(int)

        P_outlayer_mask = np.zeros(y_pred.shape[0]).astype(bool)
        P_outlayer_mask[P_outlayer_idx] = True
        P_outlayer_mask = np.expand_dims(P_outlayer_mask, axis = 1)

        N_outlayer_mask = np.zeros(y_pred.shape[0]).astype(bool)
        N_outlayer_mask[N_outlayer_idx] = True
        N_outlayer_mask = np.expand_dims(N_outlayer_mask, axis = 1)

        outlayer_mask = np.expand_dims(np.hstack((N_outlayer_mask, np.atleast_2d(P_outlayer_mask))), axis=0)
        if i == 0:
            mask = outlayer_mask
        else:
            mask = np.concatenate([mask, outlayer_mask], axis=0)

    return mask

