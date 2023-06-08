import os
from collections import OrderedDict
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

COLORS = (
    "#FF3838",
    "#2C99A8",
    "#FF701F",
    "#6473FF",
    "#CFD231",
    "#48F90A",
    "#92CC17",
    "#3DDB86",
    "#1A9334",
    "#00D4BB",
    "#FF9D97",
    "#00C2FF",
    "#344593",
    "#FFB21D",
    "#0018EC",
    "#8438FF",
    "#520085",
    "#CB38FF",
    "#FF95C8",
    "#FF37C7",
)


def yolo_draw_predections(image, object_predictions):
    # Convert PIL image to ImageDraw object
    draw = ImageDraw.Draw(image)

    prediction = object_predictions[0]

    scores = prediction['scores'].cpu().detach().numpy()

    if scores.size == 0:
        return image

    for idx, score in enumerate(scores):
        # Extract prediction details
        bbox = prediction['boxes'][idx].cpu().detach().numpy()
        # score = prediction['scores'][0].cpu().detach().numpy()
        category_id = prediction['labels'][idx].cpu().detach().numpy()
        category_name = f'name_{category_id}'

        color = COLORS[category_id.item()]

        # Define the font properties for text overlay
        font = ImageFont.load_default()

        # Draw the bounding box rectangle
        print(f'{score}: {bbox}')
        draw.rectangle(bbox, outline=color, width=3)

        # Prepare the text overlay
        text = f"{category_name} ({score:.2f})"

        # Calculate the position for text overlay
        text_width, text_height = draw.textsize(text, font=font)  # font = font
        x = bbox[0]
        y = bbox[1] - text_height

        # Draw the text overlay
        draw.rectangle((x, y, x + text_width, y + text_height), fill=color)
        draw.text((x, y), text, fill="white", font=font)  # font = font

    return image


def draw_predections(image, boxes, scores, labels, thr):
    # Convert PIL image to ImageDraw object
    image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    width = 3

    for boxe, score, label in zip(boxes, scores, labels):
        if score > thr:
            # Extract prediction details
            bbox = boxe.cpu().detach().numpy()
            # score = prediction['scores'][0].cpu().detach().numpy()
            category_id = label.cpu().detach().numpy()
            category_name = f'name_{category_id}'

            color = COLORS[category_id.item()]

            # Define the font properties for text overlay
            font = ImageFont.load_default()

            # Draw the bounding box rectangle
            draw.rectangle(tuple(bbox), outline=color, width=width)

            # Prepare the text overlay
            text = f"{category_name} ({score:.2f})"

            # Calculate the position for text overlay
            # text_width, text_height = draw.textsize(text, font=font) # font =
            # font
            x = bbox[0]
            y = bbox[1]

            l, t, r, b = draw.textbbox((x, y), text, font=font)
            h = b - t + width
            # Draw the text overlay
            draw.rectangle((l, t - h, r, b - h), fill=color)
            draw.text((x, y - h), text, fill="white", font=font)  # font = font

    return image


def eval_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    if model.training:
        model.eval()

    original_image_sizes = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.")

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training = True
    # model.roi_heads.training=True

    # proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2]
                             for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(
        proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(
        anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(
        proposals, targets)
    box_features = model.roi_heads.box_roi_pool(
        features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets)
    detector_losses = {
        "loss_classifier": loss_classifier,
        "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(
        detections,
        images.image_sizes,
        original_image_sizes)  # type: ignore[operator]
    model.rpn.training = False
    model.roi_heads.training = False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


def read_source(source, model):
    """
    Read source, and return an image.

    Args:
        model (str): Type of model.
        source (str | PIL | np.ndarray): The source of the image to make predictions on.
    """
    if isinstance(source, str):
        if os.path.isfile(source):
            source = read_image(source)
        else:
            raise Exception(f'File {source} not found')

    if model == 'pl' or model == 'torch':
        source = transforms.ToTensor()(source) # C, H, W
        return source
    else:
        return source

def read_image(path):
    with open(path, "rb") as f:
        img = Image.open(f)  # PIL.Image.Image
        return img.convert("RGB")
