# Segmenting Birds with SAM

This repository provides a very simple pipeline for calling the segment everything model (SAM) on the well-known Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset.

CUB-200-2011 dataset includes 11788 images of 200 different bird species, with annotations on 15 part locations and 1 bounding box representing the whole bird. These parts, e.g., bird beaks, are demonstrated to play a key role in fine-grained bird recognition. In the original annotation, part locations were marked by key points - this repository aims at presenting the bird parts with segmentation masks. This is done by employing the key point coordinates as prompts in the SAM model pre-trained on large-scale datasets. Besides, we tried to adopt the provided bounding box as prompts, which generated binary segmentation on the bird and the image background.

<center class="half">
<img src=assets/image.jpg width="200"/><img src=assets/binary_mask.png width="200"/><img src=assets/multi_class_mask.png width="200"/>
</center>



## Dependencies
 - SAM dependencies following [SAM](https://github.com/facebookresearch/segment-anything) 
 - imgviz

## Hands-on the model
**Prepare dataset**  
Download the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset, and unzip it in data/. The directory should be like:
data/  
    CUB_200_2011/
        attributes/
        images/
        parts/
        images.txt
        bounding_boxes.txt
        ...

**Generate segmentation masks with SAM**

    python3 sam_segmentation.py --prompt=<prompt> --model=<model>

`prompt` means the type of SAM prompts, which can be 'box', 'point', or 'click'. 'box' is associated with the bounding box annotation of birds. This option leads to a binary segmentation prediction. 'point' refers to the key point annotations of part locations, where each location is annotated with one key point, while 'click' includes multiple points on each bird part (from multiple annotators).   

`model` represents the model type of SAM backbone, which can be 'vit_b', 'vit_l', or 'vit_h'. 

Segmentation results can be found under 'segmentation/'. 

## License
MIT License

