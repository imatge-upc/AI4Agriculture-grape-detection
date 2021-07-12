# -*- coding: utf-8 -*-
#
# Functions by Pau Marquez, UPC, 2021
# File & small mods. by Ramon Morros, UPC, 2021
#

import torch
import json
import pickle
from torchvision.ops import box_iou
import numpy as np
from skimage import measure
import pydensecrf.densecrf as dcrf


def bbox_intersect(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1 : return (0, 0, 0, 0) # None

    return (x1, y1, x2, y2)

def bbox_area(a):
    return (a[2] - a[0]) * (a[3] - a[1])

def bbox_IoU(a, b, return_area_a=False):
    inter  = bbox_area(bbox_intersect(a, b))
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    if area_a == 0 or area_b == 0:
        iou = 1
    else:
        iou = inter / (area_a + area_b - inter)
    if return_area_a:
        return iou, area_a
    else:
        return iou

def crf_post_process(img, probs, H: int, W: int) -> None:
    K = 2 # Number of classes
    #assert img.shape == (H, W, 3), "wrong image shape"
    #assert probs.shape == (K, H, W), "wrong probs shape"

    inf_neglogp = (-probs.log()).cpu().numpy()
    final_np    = np.zeros((K, H, W))

    d = dcrf.DenseCRF2D(W, H, K)

    inf_neglogp_flat: np.ndarray = inf_neglogp.reshape((K, -1))
    d.setUnaryEnergy(inf_neglogp_flat)

    d.addPairwiseGaussian(sxy=1, compat=3)

    # im = np.ascontiguousarray(np.rollaxis(uintimage, 0, 3), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    d.addPairwiseBilateral(sxy=10, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)

    return np.array(Q).reshape(K, H, W)


def apply_nms(results, nms_threshold):
    nms_idx = [torch.ops.torchvision.nms(r["boxes"], r["scores"], nms_threshold) for r in results]
    results_nms = [{
        **res,
        "boxes":  res['boxes'][nms_idx[i]],
        "labels": res['labels'][nms_idx[i]],
        "scores": res['scores'][nms_idx[i]]
    } for i, res in enumerate(results)]
    return results_nms


def get_faster_bboxes(trainer, data_d):
    data = trainer.transform_data(data_d)
    with torch.no_grad():
        res = trainer.forward(trainer.model, data)[0]
    nms_res = apply_nms([res], 0.4)[0]
    nms_res = {k: v.to(torch.device('cpu')) if type(v) == torch.Tensor else v for k, v in nms_res.items()}
    return nms_res

def get_all_faster_bboxes(trainer,loader):
    results = []
    for data_d, target_d in trainer.custom_collate(loader):
        results.append({
            "faster_forward": get_faster_bboxes(trainer, data_d),
            "image_id": target_d[0]["image_id_str"]
        })
    return results

def get_bbox_from_blob(blob):
    assert blob.dtype == np.bool
    rows = np.any(blob, axis=0)
    cols = np.any(blob, axis=1)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = (rmin, cmin, rmax, cmax)
    return bbox

def get_bboxes_from_mask(binary_mask, min_area=0):
    blobs, n_blob = measure.label(binary_mask, background=0, return_num=True)
    bboxes = [
        get_bbox_from_blob(blobs == blob_id)
            for blob_id in range(1, n_blob + 1)
                if (blobs==blob_id).sum() >= min_area
    ]
    return bboxes

def add_mask(mask, boxes, fill_value=1):
    for box in boxes:
        mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = fill_value


def get_masks_from_boxes(results_boxes, target_boxes):
    all_boxes = np.concatenate([results_boxes, target_boxes])

    width, height = all_boxes.max(0)[[2,3]]

    pred_mask   = np.zeros((height, width))
    target_mask = np.zeros((height, width))
    add_mask(pred_mask,   results_boxes)
    add_mask(target_mask, target_boxes)

    return pred_mask, target_mask

def get_metrics_from_masks(pred_mask, target_mask):
    tp = np.sum(pred_mask * target_mask)
    fp = np.sum(pred_mask * (1-target_mask))
    recall    = tp/target_mask.sum()
    precision = tp/(tp+fp)
    f1_score  = hmean([recall, precision])
    return recall, precision, f1_score

def get_mask_metrics(results, target):
    pred_mask, target_mask = get_masks_from_boxes(results, target)
    return get_metrics_from_masks(pred_mask, target_mask)

        
def get_mask(image, boxes):
    #item = loader.dataset[image_id]
    #image = item["image"]
    #boxes = item["target"]["boxes"]
    mask = np.zeros(image.shape[:-1])
    add_mask(mask, boxes, fill_value=1)
    return mask
        
# use get_matching_boxes() instead
def get_wrong_bboxes(gt, pred, lower_iou_thresh=0.0, upper_iou_thresh=1.0):
    all_ious            = box_iou(gt['boxes'], pred['boxes'])
    not_matched_idxs    = [i for i, ious in enumerate(all_ious) if ious.max() < lower_iou_thresh]
    almost_matched_idxs = [i for i, ious in enumerate(all_ious) if (ious.max() >= lower_iou_thresh) & (ious.max() < upper_iou_thresh)]
    matched_idxs        = [i for i, ious in enumerate(all_ious) if ious.max() >= upper_iou_thresh]
    return gt['boxes'][not_matched_idxs], gt['boxes'][almost_matched_idxs], gt['boxes'][matched_idxs]


def get_matching_bboxes(gt_boxes, pred_boxes, lower_iou_thresh=0.0, upper_iou_thresh=1.0, find_gt_boxes=True):
    """
    find_gt_boxes: [bool] 
    if true will return boxes from ground truth where the
    iou thresholds are matched with a predicted bounding box.
    if false will return boxes from predictions where the
    iou thresholds are matched with a ground truth bounding box.
    """
    assert lower_iou_thresh <= upper_iou_thresh

    all_ious = box_iou(gt_boxes, pred_boxes)

    if not find_gt_boxes:
        all_ious = all_ious.permute(1,0)

    unmatched_idxs = [i for i, ious in enumerate(all_ious) if (ious.max() <= upper_iou_thresh) & (ious.max() >= lower_iou_thresh)]

    return gt_boxes[unmatched_idxs] if find_gt_boxes else pred_boxes[unmatched_idxs]


# EXTRACT FASTER RCNN INFERENCES TO JSON
def extract_faster_inferences(faster_trainer, out_dir):
    train_loader = faster_trainer.get_data_loader(faster_trainer.opt, 'train')
    train_res    = get_all_faster_bboxes(faster_trainer, train_loader)

    val_loader   = faster_trainer.get_data_loader(faster_trainer.opt, 'val')
    val_res      = get_all_faster_bboxes(faster_trainer, val_loader)

    test_loader  = faster_trainer.get_data_loader(faster_trainer.opt, 'test')
    test_res     = get_all_faster_bboxes(faster_trainer, test_loader)

    parsed_res = []
    for res in train_res:
        parsed_res.append({
            "boxes": res["faster_forward"]["boxes"].tolist(),
            "scores": res["faster_forward"]["scores"].tolist(),
            "image_id": res["image_id"],
            "split": "train"
        })
    for res in val_res:
        parsed_res.append({
            "boxes": res["faster_forward"]["boxes"].tolist(),
            "scores": res["faster_forward"]["scores"].tolist(),
            "image_id": res["image_id"],
            "split": "val"
        })
    for res in test_res:
        parsed_res.append({
            "boxes": res["faster_forward"]["boxes"].tolist(),
            "scores": res["faster_forward"]["scores"].tolist(),
            "image_id": res["image_id"],
            "split": "test"
        })

    with open(out_dir, "w") as fd:
        json.dump(parsed_res, fd)


def get_unmatched_boxes(trainer, max_batches=None, find_gt_boxes=False):

    assert trainer.opt.step_batch_size == 1 # Forward only processes one image
    not_matched = {}
    
    for i, (data_d, target_d) in enumerate(trainer.custom_collate(trainer.val_loaders[0][1])):
        if max_batches and i > max_batches:
            break

        data, target = trainer.transform_data(data_d, target_d)

        with torch.no_grad():
            res = trainer.forward(trainer.model, data, target)

        if trainer.opt.model == "ResidualUNet":
            rgb_image = (unorm(data[0].permute((1,2,0)))*255).type(torch.uint8).cpu().numpy()
            crf_res   = crf_post_process(np.ascontiguousarray(rgb_image), res[0], *rgb_image.shape[:2])
            mask      = crf_res.transpose((1,2,0))[:,:,1] > 0.9
            bboxes    = torch.tensor(get_bboxes_from_mask(mask), device=trainer.device)
        else:
            bboxes    = res[0]["boxes"]
            
        not_matched = get_matching_bboxes(
            target[0]["boxes"], bboxes, lower_iou_thresh=0.0, upper_iou_thresh=0.0, find_gt_boxes=find_gt_boxes
        )
        not_matched[target[0]["image_id_str"]] = not_matched_curr.cpu().numpy().copy()
    return not_matched
