def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    interArea = inter_w * inter_h

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    union = boxAArea + boxBArea - interArea + 1e-6

    return interArea / union


def evaluate(preds, gts):
    """
    Evaluate predictions using IoU

    Args:
        preds: list of prediction dicts
        gts: list of ground truth dicts

    Returns:
        dict with metrics
    """

    ious = []

    min_len = min(len(preds), len(gts))

    for i in range(min_len):
        iou = compute_iou(preds[i]["bbox"], gts[i]["bbox"])
        ious.append(iou)

    avg_iou = sum(ious) / len(ious) if ious else 0

    return {
        "IoU": avg_iou,
        "num_samples": len(ious)
    }