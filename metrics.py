import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import numpy as np


def compute_metrics(labels, logits):
    logits = F.softmax(logits, dim=1)  # 计算 softmax

    # 先计算 predicted_classes
    predicted_classes = torch.argmax(logits, dim=1)  # 仍然是 Tensor
    predicted_classes = predicted_classes.cpu().numpy()  # 然后转换为 NumPy 数组
    labels_np = labels.cpu().numpy()
    outputs_np = logits.cpu().detach().numpy()
    accuracy = accuracy_score(labels_np, predicted_classes)
    precision = precision_score(labels_np, predicted_classes, average='macro')
    recall = recall_score(labels_np, predicted_classes, average='macro')
    f1 = f1_score(labels_np, predicted_classes, average='macro')
    roc_auc = roc_auc_score(labels_np, outputs_np, multi_class='ovr')
    pr_auc = average_precision_score(labels_np, outputs_np, average='weighted')
    return accuracy,precision,recall,f1,roc_auc,pr_auc

def compute_binary_metrics(labels, logits):
    probabilities = torch.sigmoid(logits)  # 计算 softmax

    # 先计算 predicted_classes
    predicted_classes = (probabilities >= 0.5).int()
    predicted_classes = predicted_classes.cpu().numpy()  # 然后转换为 NumPy 数组
    labels_np = labels.cpu().numpy()
    outputs_np = probabilities.cpu().detach().numpy()

    accuracy = accuracy_score(labels_np, predicted_classes)
    precision = precision_score(labels_np, predicted_classes)
    recall = recall_score(labels_np, predicted_classes)
    f1 = f1_score(labels_np, predicted_classes)
    roc_auc = roc_auc_score(labels_np, outputs_np)
    pr_auc = average_precision_score(labels_np, outputs_np)
    return accuracy,precision,recall,f1,roc_auc,pr_auc


