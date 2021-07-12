# -*- coding: utf-8 -*-
#
# Functions by Pau Marquez, UPC, 2021
# File & small mods. by Ramon Morros, UPC, 2021
#

import pickle
import torch
import os
import numpy as np
import xmltodict
import pandas as pd
from .mask_bbox_utils import  get_bboxes_from_mask, crf_post_process

    
def get_unet_mask_pkl (image_id, results_dir, crf=True, min_area=100):
    if crf:
        name = "crf_mask_0,5"
    else:
        name = "unet_mask_0,9"

    with open(f"{results_dir}/{image_id}.pkl", "rb") as fd:
        mask = pickle.load(fd)[name].toarray()

    bboxes = get_bboxes_from_mask(mask, min_area)
    
    return mask,bboxes


def get_unet_masks(trainer, image_ids=None, loader=None):
    if loader is None:
        loader = trainer.val_loaders[0][1]
    if image_ids is None:
        image_ids = loader.dataset.ids
    for image_id in image_ids:
        item = loader.dataset[image_id]
        data_dd = [item["image"]]
        target_dd = [item["target"]]
        
        print(f"starting with {image_id}")
        
        full_im_shape   = data_dd[0].shape[:-1]
        full_mask       = torch.zeros(full_im_shape)
        full_image      = torch.zeros(data_dd[0].shape)
        full_mask_count = torch.zeros(full_im_shape)
        target_dd[0]["boxes"]  = torch.tensor([[0,0,0,0]])
        target_dd[0]["masks"]  = torch.zeros((1,*full_im_shape))
        target_dd[0]["area"]   = torch.tensor([0])
        target_dd[0]["labels"] = torch.tensor([0])

        # Split image, apply a forward pass to each subimage
        for data_d, target_d in trainer.custom_collate([[data_dd, target_dd]]):
            data, target = trainer.transform_data(data_d, target_d)
            with torch.no_grad():
                res = trainer.forward(trainer.model, data, target)
            curr_pred_probs = res[0][1].cpu().numpy()
            slice_i = target[0]["current_slice"][:-1]
            full_mask[slice_i] += torch.from_numpy(np.array(Image.fromarray(curr_pred_probs).resize(data_d[0].shape[:-1], Image.NEAREST)))
            full_mask_count[slice_i] += 1
            full_image[slice_i] = data_d[0].cpu()

        full_mask_probs   = full_mask/full_mask_count
        mask_probs_crf    = torch.zeros((2,*full_mask_probs.shape))
        mask_probs_crf[0] = 1 - full_mask_probs
        mask_probs_crf[1] = full_mask_probs

        image = loader.dataset.__getitem__(image_id)["image"]
        image = np.ascontiguousarray((image*255).type(torch.uint8).cpu().numpy())

        # crf_probs = np.array([1,2])#crf_post_process(image, mask_probs_crf, *image.shape[:-1])[1] 
        crf_probs = crf_post_process(image, mask_probs_crf, *image.shape[:-1])[1] # JRMR

        yield {
            "image_id":      image_id,
            "unet_mask_0,9": csr_matrix((full_mask_probs.cpu().numpy() > 0.9).astype(np.byte)),
            "crf_mask_0,5":  csr_matrix((crf_probs > 0.5).astype(np.byte)),
            "crf_mask_0,9":  csr_matrix((crf_probs > 0.9).astype(np.byte)),
            "full_image":    full_image.cpu().numpy()
        }


def store_unet_masks(trainer, loader, out_dir):
    unet_masks = get_unet_masks(trainer, loader = loader)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        for unet_mask in unet_masks:
            image_id = unet_mask["image_id"]
            with open(f'{out_dir}/{image_id}.pkl', "wb") as fd:
                pickle.dump(unet_mask, fd)

                
def get_image_info(image_path, get_shape=False):
    '''
    Args:
        image_path (str): Path to the xml file containing the annotation of the image
    Returns:
            Pandas.DataFrame: Dataframe containing the bounding boxes and the annotator of each bounding box
    '''
    with open(image_path, 'rb') as fd:
        xml_dict = xmltodict.parse(fd)

    shape = xml_dict["annotation"]["size"].values()
    
    if 'object' not in xml_dict["annotation"]:
        data=[]
    else:
        data=[(str(image_id), r['name'], *r['bndbox'].values()) for r in xml_dict["annotation"]["object"]]
    bboxes = pd.DataFrame(
        data,
        columns=['image_id','annotator', 'xmin', 'ymin', 'xmax', 'ymax']).astype(np.int)

    if get_shape:
        return bboxes, shape
    
    return bboxes
