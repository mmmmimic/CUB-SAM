import argparse
from datasets import DataLoader
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import imgviz
import torch

parser = argparse.ArgumentParser(
                    prog='CUB-SAM',
                    description='Employ SAM for bird segmentation')
parser.add_argument('--prompt', type=str, default='box', help="prompt type for segmentation from [box, point, click]")
parser.add_argument('--model', type=str, default='vit_l', help="SAM model type from [vit_b, vit_l, vit_h]")

args = parser.parse_args()

def merge_segmentation(part_masks, point_labels):
    # merge segmentation results for different point prompts
    assert len(part_masks),'there is no prediction from sam'
    mask = np.zeros_like(part_masks[0], dtype=np.float32)
    areas = list(map(lambda x: x.sum(), part_masks))
    inds = list(sorted(list(range(len(point_labels))), key=lambda x: areas[x], reverse=True))
    # painting from parts with larger areas, so that smaller parts can cover the larger parts
    for i in inds:
        m, l = part_masks[i], point_labels[i]
        mask[m] = l
    return mask

def save_colored_mask(mask, save_path):
    mask = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap()
    mask.putpalette(colormap.flatten())
    mask.save(save_path)

if __name__ == "__main__":
    if not os.path.exists('segmentations'):
        os.mkdir('segmentations')
    if args.prompt == 'box':
        # binary segmentation
        mask_dir = os.path.join('segmentations', 'bin_mask')
    elif args.prompt == 'point':
        # multi-class segmentation
        mask_dir = os.path.join('segmentations', 'mul_mask')
    elif args.prompt == 'click':
        # multi-class segmentation with dense points (clicks)
        mask_dir = os.path.join('segmentations', 'click_mask')
    else:
        raise NameError

    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    
    # download pretrained weights
    if args.model == 'vit_b':
        model_path = 'sam_vit_b_01ec64.pth'
    elif args.model == 'vit_l':
        model_path = 'sam_vit_l_0b3195.pth'
    elif args.model == 'vit_h':
        model_path = 'sam_vit_h_4b8939.pth' 
    else:
        raise NameError       
    
    if not os.path.isfile(model_path):
        os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running model on {device}")
    
    sam = sam_model_registry[args.model](checkpoint=model_path)
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    
    loader = DataLoader('data/CUB_200_2011')
    
    for index in tqdm(range(len(loader))):
        image_path, image, bboxes, point_coords, point_labels, click_coords, click_labels = loader(index)
        predictor.set_image(image)
        
        if args.prompt == 'box':
            # use box
            masks, _, _ = predictor.predict(box=np.array(bboxes))
            mask = masks[0] # by default only take the first predicted mask 
        elif args.prompt == 'point':
            # use point
            part_masks = []
            for c, l in zip(point_coords, point_labels):
                masks, _, _ = predictor.predict(point_coords=np.array([c]), point_labels=np.array([l]))
                # masks, _, _ = predictor.predict(point_coords=np.array(point_coords), point_labels=np.array(point_labels))
                mask = masks[1] # part of the bird
                part_masks.append(mask)
            # merge part segmentations
            mask = merge_segmentation(part_masks, point_labels)
        elif args.prompt == 'click':
            # use point
            part_masks = []
            label = list(set(click_labels))
            for l in label:
                ind = np.array(list(filter(lambda x: click_labels[x]==l, list(range(len(click_labels))))))
                c = np.array(click_coords)[ind]
                l = np.array(click_labels)[ind]
                masks, _, _ = predictor.predict(point_coords=np.array(c), point_labels=np.array(l))
                # masks, _, _ = predictor.predict(point_coords=np.array(point_coords), point_labels=np.array(point_labels))
                mask = masks[1] # part of the bird
                part_masks.append(mask)
            # merge part segmentations
            mask = merge_segmentation(part_masks, point_labels)
        else:
            raise NameError
        
        # save mask
        if not os.path.exists(os.path.join(mask_dir, image_path.split('/')[0])):
            os.mkdir(os.path.join(mask_dir, image_path.split('/')[0]))
        save_colored_mask(mask, save_path=os.path.join(mask_dir, image_path.replace('jpg', 'png')))
        
            