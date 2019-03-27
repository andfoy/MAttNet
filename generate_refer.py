
import json
import tqdm
import torch
import numpy as np
from skimage import measure
import pycocotools.mask as mask
# from pycocotools.mask import _mask


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to
            approximated polygonal chain. If tolerance is 0,
            the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may
        # get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


gt_robo = json.load(open(
    'data_endovis2017_instances_cropped/group1/'
    'RobotSeg2017_inst_class_group1.json', 'r'))

out_robo = torch.load('cand_dataset_group1_train.pth')

categories = {x['id']: x['name'].lower() for x in gt_robo['categories']}
img_anns = {x['id']: [] for x in gt_robo['images']}
for ann in gt_robo['annotations']:
    ann['rle'] = mask.frPyObjects(
        ann['segmentation'], ann['height'], ann['width'])
    img_anns[ann['image_id']].append(ann)

robo_coco = {'info': gt_robo['info'],
             'licenses': gt_robo['licenses'],
             'images': [], 'annotations': [],
             'categories': gt_robo['categories']}
ref_id = 0
sent_id = 0
instance_id = 0
refs = []

for img in tqdm.tqdm(gt_robo['images']):
    img_id = img['id']
    robo_coco['images'].append(img)
    gt_instances = img_anns[img_id]
    out_instances = list(out_robo[img_id - 1].values())
    for instance in out_instances:
        rle = instance['mask']
        instance['bbox'] = instance['bbox'][:, :-1]
        max_iou, cat = 0, 7
        for gt_instance in gt_instances:
            iou = mask.iou([rle], gt_instance['rle'], [False]).max()
            if iou > max_iou:
                max_iou, cat = iou, gt_instance['category_id']
        H, W = rle['size']
        area = mask.area([rle]).item()
        segmentation = binary_mask_to_polygon(mask.decode(rle))
        sent = f'{instance["sentences"][0].split()[0]} {categories[cat]}'
        ann = {'id': instance_id, 'image_id': img_id, 'category_id': cat,
               'iscrowd': 0, 'area': area,
               'bbox': instance['bbox'][0].tolist(),
               'segmentation': segmentation,
               'width': W, 'height': H}
        ref = {'sent_ids': [sent_id], 'file_name': img['file_name'],
               'ann_id': instance_id, 'ref_id': ref_id, 'image_id': img_id,
               'split': 'train', 'category_id': cat,
               'sentences': [{
                   'tokens': sent.split(), 'raw': sent, 'sent_id': sent_id,
                   'sent': sent
                }]}
        refs.append(ref)
        robo_coco['annotations'].append(ann)
        ref_id += 1
        sent_id += 1
        instance_id += 1

gt_robo = json.load(open(
    'data_endovis2017_instances_cropped/group2/'
    'RobotSeg2017_inst_class_group2.json', 'r'))
out_robo = torch.load('cand_dataset_group2.pth')

img_anns = {x['id']: [] for x in gt_robo['images']}
for ann in gt_robo['annotations']:
    ann['rle'] = mask.frPyObjects(
        ann['segmentation'], ann['height'], ann['width'])
    img_anns[ann['image_id']].append(ann)

for img in tqdm.tqdm(gt_robo['images']):
    img_id = img['id']
    out_instances = list(out_robo[img_id - 1].values())
    img_id += len(robo_coco['images'])
    img['id'] = img_id
    robo_coco['images'].append(img)
    for instance in out_instances:
        rle = instance['mask']
        instance['bbox'] = instance['bbox'][:, :-1]
        H, W = rle['size']
        area = mask.area([rle]).item()
        segmentation = binary_mask_to_polygon(mask.decode(rle))
        sents = instance['sentences']
        cat_id = instance['cats'][0, 0]
        ann = {'id': instance_id,
               'image_id': img_id,
               'category_id': cat,
               'iscrowd': 0, 'area': area,
               'bbox': instance['bbox'][0].tolist(),
               'segmentation': segmentation,
               'width': W, 'height': H}
        ref = {'sent_ids': [],
               'file_name': img['file_name'],
               'ann_id': instance_id, 'ref_id': ref_id, 'image_id': img_id,
               'split': 'test', 'sentences': [], 'category_id': cat}
        for sent in sents:
            sent_info = {'tokens': sent.split(), 'raw': sent,
                         'sent_id': sent_id, 'sent': sent}
            ref['sent_ids'].append(sent_id)
            ref['sentences'].append(sent_info)
            sent_id += 1
        refs.append(ref)
        robo_coco['annotations'].append(ann)
        ref_id += 1
        instance_id += 1
