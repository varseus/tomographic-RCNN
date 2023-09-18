import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import CTDataset
from utils import *

import math
import sys

from torchvision.transforms.functional import pil_to_tensor

def transform_pil_to_image(image, target):
    return pil_to_tensor(image), target

if __name__ == "__main__":
    # use metal
    device = torch.device('mps') if torch.has_mps else torch.device('cpu')

    dataset = CTDataset(transform_pil_to_image)
    indices = torch.arange(0, len(dataset)).tolist() # do I need to make a tensor before a list?
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=4,collate_fn=collate_fn)

    # load pretrained FasterRCNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # create new box predictor for model to identify lesions
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # move model to device
    model.to(device)
    model.load_state_dict(torch.load("./model_state_dict.pt"))

    evaluate(model, data_loader_test, device=device)