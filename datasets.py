from typing import Any
import os.path as pth
import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_text(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.split('\n')
    return content[:-1] # the last element is an intent

def read_image(file_path):
    return cv2.imread(file_path)[...,[2,1,0]]

class DataLoader:
    def __init__(self, root_path='data/CUB_200_2011') -> None:
        super().__init__()
        self.image_paths = read_text(pth.join(root_path, 'images.txt'))
        self.bboxes = read_text(pth.join(root_path, 'bounding_boxes.txt')) # box means background-foreground binary segmentation
        self.parts = read_text(pth.join(root_path, 'parts', 'part_locs.txt')) # fine-grained segmentation for different parts
        self.clicks = read_text(pth.join(root_path, 'parts', 'part_click_locs.txt')) # more points for each part
        self.root_path = root_path
        
    def __call__(self, index: Any) -> Any:
        image_path_line = self.image_paths[index]
        index, image_path = image_path_line.split(' ')
        full_image_path = pth.join(self.root_path, 'images', image_path)
        image = read_image(full_image_path)

        bboxes = list(filter(lambda x: x.split(' ')[0] == index, self.bboxes))
        parts = list(filter(lambda x: x.split(' ')[0] == index, self.parts))
        
        bboxes = list(map(lambda x: np.array(x.split(' ')[1:], dtype=np.float64), bboxes))
        bboxes = list(map(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]], bboxes))
        
        parts = list(map(lambda x: np.array(x.split(' ')[1:], np.float64), parts)) # (type, x, y, status)
        parts = list(filter(lambda x: x[-1], parts)) # only keep valid annotations
        
        point_coords = list(map(lambda x: (x[1], x[2]), parts))
        point_labels = list(map(lambda x: int(x[0]), parts))
        
        clicks = list(filter(lambda x: x.split(' ')[0] == index, self.clicks))
        clicks = list(map(lambda x: np.array(x.split(' ')[1:], np.float64), clicks))
        clicks = list(filter(lambda x: x[-2], clicks)) # only keep valid annotations
        click_coords = list(map(lambda x: (x[1], x[2]), clicks))
        click_labels = list(map(lambda x: int(x[0]), clicks))        
        
        return image_path, image, bboxes, point_coords, point_labels, click_coords, click_labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __repr__(self) -> str:
        pass
    
if __name__ == "__main__":
    loader = DataLoader()
    image_path, image, bboxes, point_coords, point_labels, click_coords, click_labels = loader(3)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for b in bboxes:
        image = cv2.rectangle(image, pt1=(int(b[0]), int(b[1])), pt2=(int(b[2]), int(b[3])), color=(255, 255, 0), thickness=5)
    for p in point_coords:
        image = cv2.circle(image, center=(int(p[0]), int(p[1])), radius=1, color=(0, 255, 255), thickness=5)
    for p in click_coords:
        image = cv2.circle(image, center=(int(p[0]), int(p[1])), radius=1, color=(255, 0, 255), thickness=5)    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()