import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data import CTDataset
from utils import *

import math
import sys

from torchvision.transforms.functional import pil_to_tensor

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, format="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        # linearly warm up the learning rate from 1/1000 to intended rate
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

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

    # stochastic gradient descent with momentum
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005,momentum=0.1, weight_decay=0.0005)
    # decay the lr every step_size epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    torch.save(model.state_dict(), "./model_state_dict.pt")