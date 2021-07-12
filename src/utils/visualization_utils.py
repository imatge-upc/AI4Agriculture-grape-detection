# -*- coding: utf-8 -*-
#
# Functions by Pau Marquez, UPC, 2021
# File and small mods. by Ramon Morros, UPC, 2021
#

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import colorConverter
import matplotlib as mpl
import torch
import numpy as np
from itertools import chain


from .file_utils      import get_unet_mask_pkl
from .mask_bbox_utils import get_faster_bboxes


def display_image_w_bbox(im, bboxes, annotators = None, colors_annotators = ['blue','red','pink','brown','black','grey', 'white', 'yellow']):    
    if annotators is None:
        annotators = bboxes['annotator'].unique()
        
    fig, ax = plt.subplots(figsize=(20,30))
    ax.imshow(im)
    for bbox in bboxes[bboxes['annotator'].isin(annotators)].values:
        annotator = bbox[0]
        print (bbox)
        if len(bbox) > 4:
            bbox = bbox[1:]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=colors_annotators[annotator], facecolor='none')
        
        ax.add_patch(rect)
    plt.title(f'(w, h) = ()')
    fig.show()
    
def create_masked_plot(image, mask, boxes):
    fig, ax = plt.subplots(figsize=(10,15))
    
    color1  = colorConverter.to_rgba('white')
    color2  = colorConverter.to_rgba('red')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init()
    cmap2._lut[:,-1] = 0.2
    ax.imshow(image)
    ax.imshow(mask > 0.9, cmap=cmap2)
    for bbox in boxes:
        bbox=bbox.int()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def get_item(loader, image_id):
    item     = loader.dataset[image_id]
    data_d   = [item["image"]]
    target_d = [item["target"]]

    return data_d, target_d


def general_plot (image_id, results_dir, faster_trainer=None, title="", loader=None, positive_mask_color='red', one_plot=False, lim_dict=None, out_dir=None):
    
    if loader is None:
        print ('ERROR!! : loader argument cannot be None')
        return

    data, target = get_item(loader, image_id)

    mask, unet_boxes  = get_unet_mask_pkl(image_id, results_dir)
    unet_boxes        = torch.tensor(unet_boxes)

    gt_boxes    = target[0]["boxes"]

    if one_plot:
        fig, axes = plt.subplots(1,1, figsize=(7,15))
        main_ax = axes
        mask_ax = None
        raw_ax  = None
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        main_ax = axes[0]
        mask_ax = axes[1]
        raw_ax  = axes[2]
    
    # Build mask cmap
    color1 = colorConverter.to_rgba('white',alpha=0)
    color2 = colorConverter.to_rgba(positive_mask_color, alpha=0.3)
    cmap   = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],2)
    cmap._init()
    

    main_ax.imshow(data[0])
    main_ax.imshow(mask, cmap=cmap)
    
    if raw_ax  is not None: raw_ax.imshow(data[0])
    if mask_ax is not None: mask_ax.imshow(mask, cmap=cmap)

    for bbox in gt_boxes:
        bbox = bbox.int()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='red', facecolor='none')
        main_ax.add_patch(rect)

    for bbox in unet_boxes:
        bbox = bbox.int()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='blue', facecolor='none')
        main_ax.add_patch(rect)
    
    if faster_trainer:
        faster_boxes = get_faster_bboxes(faster_trainer, data)["boxes"]
        
        for bbox in faster_boxes:
            bbox = bbox.int()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='yellow', facecolor='none')
            main_ax.add_patch(rect)



    all_boxes = np.array(list(chain(
        gt_boxes.cpu().numpy(),
        unet_boxes.cpu().numpy(),
        [] if faster_boxes is None else faster_boxes.cpu().numpy()
    )))
    min_x, min_y = all_boxes.min(0)[[0,1]]
    max_x, max_y = all_boxes.max(0)[[2,3]]
    if lim_dict is not None and image_id in lim_dict:
        (min_x, max_x), (min_y, max_y) = lim_dict[image_id]
    delta = 400 # pixels
    for a in [main_ax, raw_ax, mask_ax]:
        if a is None:
            continue
        a.set_xlim(max(min_x - delta, 0), min(max_x + delta, data[0].shape[1]))
        a.set_ylim(max(0, max_y + delta), min(min_y - delta, data[0].shape[0]))
        a.set_xticks([])
        a.set_yticks([])
    plt.suptitle(title)
    fig.tight_layout(h_pad=2)
    #plt.savefig("figures/random_gt_image.png")
    if out_dir is not None:
        #plt.savefig(f"{DATA_DIR}/figures/random_gt_image.png")
        plt.savefig(out_dir)
    plt.show()


def plot_mask (full_mask, image_id, loader, out_dir=None):
    fig, ax  = plt.subplots(1,2,figsize=(40,30))# v3
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[colorConverter.to_rgba('white',alpha=0.0),colorConverter.to_rgba('red',alpha=0.3)],256)
    cmap2._init()
    
    item      = loader.dataset.__getitem__(image_id)
    rgb_image = item["image"]
    target    = item["target"]

    ax[0].imshow(rgb_image)
    ax[0].imshow(full_mask, cmap=cmap2)
    ax[1].imshow(full_mask, cmap=cmap2)
    ax[0].set_title("Masked image")
    ax[1].set_title("Mask")
    ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir)
    plt.show()
