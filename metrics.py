import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, recall_score, precision_score
from transformers import EvalPrediction
from main import cfg

def count_f1_max(pred, target):
    """
        F1 score with the optimal threshold, Copied from TorchDrug.

        This function first enumerates all possible thresholds for deciding positive and negative
        samples, and then pick the threshold with the maximal F1 score.

        Parameters:
            pred (Tensor): predictions of shape :math:`(B, N)`
            target (Tensor): binary targets of shape :math:`(B, N)`
      """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    max_index = all_f1.argmax()
    max_f1 = all_f1[max_index]
    max_precision = all_precision[max_index]
    max_recall = all_recall[max_index]
    return max_f1.item(), max_precision.item(), max_recall.item()


def classification_metrics(predictions, labels):
    preds = torch.tensor(predictions)
    y_true = torch.tensor(labels, dtype=torch.float)
    if cfg.task_type == 'multilabel':
        fmax, best_precision, best_recall = count_f1_max(preds, y_true)
        probs = torch.sigmoid(preds)
        y_pred = (probs >= 0.5).int().cpu().numpy()
    else:
        probs = F.softmax(preds, dim=-1)
        y_pred = (probs >= 0.5).int().cpu().numpy()
        fmax = f1_score(y_true, y_pred, average='weighted')
        best_precision = precision_score(y_true, y_pred, average='weighted')
        best_recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        'f1': fmax,
        'precision': best_precision,
        'recall': best_recall,
        'accuracy': accuracy,
    }
    return metrics


def regression_metrics(predictions, labels):
    preds = torch.tensor(predictions)
    y_true = torch.tensor(labels, dtype=torch.float32)

    preds_np = preds.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    mse = mean_squared_error(y_true_np, preds_np)
    r2 = r2_score(y_true_np, preds_np)

    metrics = {
        'MSE': mse,
        'R2': r2
    }
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    if cfg.task_type == 'regression':
        result = regression_metrics(
        predictions=preds,
        labels=p.label_ids)
    else:
        result = classification_metrics(
            predictions=preds,
            labels=p.label_ids)
    return result