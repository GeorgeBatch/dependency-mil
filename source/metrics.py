import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from torchmetrics import (AUROC, Accuracy, F1Score, MetricCollection,
                          Precision, Recall)


def get_classification_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'roc_auc': AUROC(**kwargs),
        'accuracy': Accuracy(**kwargs),
        'f1': F1Score(**kwargs),
        'precision': Precision(**kwargs),
        'recall': Recall(**kwargs),
    })


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def multi_label_roc(labels, predictions, num_classes, mode, test_label_weight_masks=None, pos_label=1):
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        sample_weight = None if test_label_weight_masks is None else test_label_weight_masks[:, c]
        c_auc = roc_auc_score(label, prediction, sample_weight=sample_weight)
        aucs.append(c_auc)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=pos_label, sample_weight=sample_weight)
        thresholds.append(threshold)
        if mode == 'valid':
            fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
            thresholds_optimal.append(threshold_optimal)
        elif mode == 'test':
            thresholds_optimal.append(0.5)
    return aucs, thresholds, thresholds_optimal


def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1 - np.count_nonzero(np.array(bag_labels).astype(int) - bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore
