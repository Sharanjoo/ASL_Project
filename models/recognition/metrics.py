import numpy as np
import torch


def predict_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            all_logits.append(model(features.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def classification_metrics(logits, labels, num_classes):
    ranked = np.argsort(-logits, axis=1)
    top1 = float((ranked[:, 0] == labels).mean())
    top5 = float((ranked[:, :5] == labels[:, None]).any(axis=1).mean())

    preds = ranked[:, 0]
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        if tp + fp + fn == 0:
            continue
        f1_scores.append(2 * tp / (2 * tp + fp + fn))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {"top1": top1, "top5": top5, "macro_f1": macro_f1, "n": int(len(labels))}
