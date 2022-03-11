from transformers import DetrFeatureExtractor
from transformers import DetrForSegmentation

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2
import torch
from copy import deepcopy
from PIL import Image
import numpy

from transformers.models.detr.feature_extraction_detr import rgb_to_id, id_to_rgb
import io

img_path = '/home/manoj/Desktop/Work/segmentation/ExampleTrainData/ExampleTrainData/Site1Good/Site1_Good1.jpg'
# img = Image.open(img_path)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# stuff classes
# 0 things 1 banner 2 blanket 3 bridge 4 cardboard 5 counter 6 curtain 7 door-stuff 8 floor-wood 9 flower 10 fruit 
# 11 gravel 12 house 13 light 14 mirror-stuff 15 net 16 pillow 17 platform 18 playingfield 19 railroad 20 river 
# 21 road 22 roof 23 sand 24 sea 25 shelf 26 snow 27 stairs 28 tent 29 towel 30 wall-brick 
# 31 wall-stone 32 wall-tile 33 wall-wood 34 water 35 window-blind 36 window 37 tree 38 fence 39 ceiling 40 sky 
# 41 cabinet 42 table 43 floor 44 pavement 45 mountain 46 grass 47 dirt 48 paper 49 food 50 building 51 rock 52 wall 53 rug 

# thing classes
# 0 person 1 bicycle 2 car 3 motorcycle 4 airplane 5 bus 6 train 7 truck 8 boat 9 traffic light 10 fire hydrant 
# 11 stop sign 12 parking meter 13 bench 14 bird 15 cat 16 dog 17 horse 18 sheep 19 cow 20 elephant 
# 21 bear 22 zebra 23 giraffe 24 backpack 25 umbrella 26 handbag 27 tie 28 suitcase 29 frisbee 30 skis 
# 31 snowboard 32 sports ball 33 kite 34 baseball bat 35 baseball glove 36 skateboard 37 surfboard 38 tennis racket 39 bottle 40 wine glass 
# 41 cup 42 fork 43 knife 44 spoon 45 bowl 46 banana 47 apple 48 sandwich 49 orange 50 broccoli 
# 51 carrot 52 hot dog 53 pizza 54 donut 55 cake 56 chair 57 couch 58 potted plant 59 bed 60 dining table 
# 61 toilet 62 tv 63 laptop 64 mouse 65 remote 66 keyboard 67 cell phone 68 microwave 69 oven 70 toaster 
# 71 sink 72 refrigerator 73 book 74 clock 75 vase 76 scissors 77 teddy bear 78 hair drier 79 toothbrush

classes_to_exclude_in_stuff = [40]
classes_to_exclude_in_thing = [0,1,2,3,4,5,6,7,8,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,60,61,62,63,64,65,66,67,68.69,70,71,72,73,74,75,76,77,78,79]

# intrested_list = [3,7,8,11,12,17,18,21,22,23,24,26,27,28,30,31,32,33,34,35,36,38,39,50,51,52]


def get_mask(img_path):
    img = Image.open(img_path)
    encoding = feature_extractor(img, return_tensors="pt")
    outputs = model(**encoding)

    processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]


    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb_to_id(panoptic_seg))

    panoptic_seg_np = numpy.array(panoptic_seg, dtype=numpy.uint8)

    
    # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    v = Visualizer(numpy.array(img.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)


# cv2.waitKey(0)

    
    mask = numpy.zeros_like(panoptic_seg_np)

# print('panoptic_seg',panoptic_seg.shape)
# print('panoptic_seg_np',panoptic_seg_np.shape)
# print('mask',mask.shape)



    wanted_list = []
    for info in segments_info:
        if info['isthing'] == False and info['category_id'] not in classes_to_exclude_in_stuff:
            wanted_list.append(info['id'])

        if info['isthing'] == True and info['category_id'] not in classes_to_exclude_in_thing:
            wanted_list.append(info['id'])

    for i in range(len(wanted_list)):
        id = wanted_list[i]
        # img[panoptic_seg!=id] = 0
        mask[panoptic_seg_np==id] = 1

    return(v.get_image(),mask)


# cv2.imshow('mat',v.get_image())
# cv2.imshow('mask',mask*255)
# cv2.waitKey(0)

# # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
# meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
# for i in range(len(segments_info)):
#     c = segments_info[i]["category_id"]
#     segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

# # Finally we visualize the prediction
# v = Visualizer(numpy.array(img.copy().resize((w, h)))[:, :, ::-1], meta, scale=1.0)
# v._default_font_size = 20
# v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
# cv2.imshow(v.get_image())
# cv2.waitKey(0)



def get_mask_from_im(img):
    # img = Image.open(img_path)
    encoding = feature_extractor(img, return_tensors="pt")
    outputs = model(**encoding)

    processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]


    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb_to_id(panoptic_seg))

    panoptic_seg_np = numpy.array(panoptic_seg, dtype=numpy.uint8)

    
    # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    v = Visualizer(numpy.array(img.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)


# cv2.waitKey(0)

    intrested_list = []
    mask = numpy.zeros_like(panoptic_seg_np)

# print('panoptic_seg',panoptic_seg.shape)
# print('panoptic_seg_np',panoptic_seg_np.shape)
# print('mask',mask.shape)




    for info in segments_info:
        if info['isthing'] == False and info['category_id'] == 21:
            intrested_list.append(info['id'])

    for i in range(len(intrested_list)):
        id = intrested_list[i]
        # img[panoptic_seg!=id] = 0
        mask[panoptic_seg_np==id] = 1

    return(v.get_image(),mask)