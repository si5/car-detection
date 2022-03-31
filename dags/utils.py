import numpy as np

import settings


### Compute IoU
def compute_iou(u, v):
    iou = 0.0
    x_min = max(u[0], v[0])
    y_min = max(u[1], v[1])
    x_max = min(u[2], v[2])
    y_max = min(u[3], v[3])

    if x_min < x_max and y_min < y_max:
        intersection = (x_max - x_min) * (y_max - y_min)
        union = (
            (u[2] - u[0]) * (u[3] - u[1]) + (v[2] - v[0]) * (v[3] - v[1]) - intersection
        )
        iou = intersection / union

    return iou


### Conpute individual average precision
def average_precision(output_group, label_group, threshold):
    output_group_sorted = sorted(output_group, key=lambda x: x[1], reverse=True)
    positive = 0
    precision_recall = []
    visited = []
    for i, x in enumerate(output_group_sorted):
        for j, bbox_label in enumerate(label_group):
            bbox_pred = x[0]
            iou = compute_iou(bbox_label, bbox_pred)
            if j not in visited and iou >= threshold:
                positive += 1
                visited.append(j)
                break

        precision = positive / (i + 1)
        recall = positive / len(label_group)
        precision_recall.append([precision, recall])

    ap_sum = 0
    for i in range(settings.NUM_AP_POINT):
        x = i * 0.1
        precision_recall = np.array(precision_recall)
        for j in range(len(precision_recall)):
            if precision_recall[j][1] >= x:
                ap_sum += max(precision_recall[j:][0])
                break
    ap = ap_sum / settings.NUM_AP_POINT
    return ap


### Compute mean average precision (no batch basis)
def mean_average_precision(output, target, threshold=0.5):
    label_group = {}
    output_group = {}

    for idx, label in enumerate(target['labels']):
        label_group.setdefault(label.item(), []).append(target['boxes'][idx])

    for idx, label_pred in enumerate(output['labels']):
        if label_pred.item() in label_group.keys():
            output_group.setdefault(label_pred.item(), []).append(
                [output['boxes'][idx], output['scores'][idx]]
            )

    ap_list = []
    for label in label_group:
        if label in output_group.keys():
            ap = average_precision(output_group[label], label_group[label], threshold)
        else:
            ap = 0.0
        ap_list.append(ap)

    if not ap_list:
        return 0.0

    map = sum(ap_list) / len(ap_list)

    return map


def collate_fn(batch):
    return tuple(zip(*batch))
