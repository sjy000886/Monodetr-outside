import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops


OUTSIDE_CENTER_DECODE_MODES = (
    'full',
    'zero_offset_new',
    'legacy',
    'full_legacy_ry',
)


def validate_outside_center_decode_mode(mode):
    if mode not in OUTSIDE_CENTER_DECODE_MODES:
        raise ValueError(
            "Invalid outside_center_decode_mode '{}'. Expected one of: {}."
            .format(mode, ', '.join(OUTSIDE_CENTER_DECODE_MODES))
        )
    return mode


def resolve_outside_center_decode_mode(
        use_outside_center_modeling, configured_mode=None):
    if configured_mode is None:
        return 'full' if use_outside_center_modeling else 'legacy'
    validate_outside_center_decode_mode(configured_mode)
    if not use_outside_center_modeling:
        return 'legacy'
    return configured_mode


def decode_projected_center(representative_center, center_offset):
    """Decode the normalized projected 3D center without range restriction."""
    return representative_center + center_offset


def select_projected_center(
        representative_center, predicted_offset, decode_mode):
    """Select the projected-center path without changing query ordering."""
    validate_outside_center_decode_mode(decode_mode)
    if decode_mode == 'legacy':
        used_offset = torch.zeros_like(representative_center)
        return representative_center, used_offset
    if predicted_offset is None:
        raise KeyError(
            "pred_outside_center_offset is required for decode mode '{}'."
            .format(decode_mode)
        )
    used_offset = (
        predicted_offset
        if decode_mode in ('full', 'full_legacy_ry')
        else torch.zeros_like(predicted_offset)
    )
    return decode_projected_center(
        representative_center, used_offset
    ), used_offset


def normalized_center_to_image(center, image_size):
    """Map normalized center coordinates to the original MonoDETR image space."""
    return center * np.asarray(image_size, dtype=np.float32)


def decode_detections(
        dets, info, calibs, cls_mean_size, threshold,
        outside_center_decode_mode='legacy'):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    decode_mode = validate_outside_center_decode_mode(
        outside_center_decode_mode
    )
    use_new_center_path = decode_mode != 'legacy'
    use_legacy_ry_path = decode_mode in ('legacy', 'full_legacy_ry')
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['img_size'][i][0]
            y = dets[i, j, 3] * info['img_size'][i][1]
            w = dets[i, j, 4] * info['img_size'][i][0]
            h = dets[i, j, 5] * info['img_size'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]

            # positions decoding
            if use_new_center_path:
                center_3d = normalized_center_to_image(
                    dets[i, j, 34:36], info['img_size'][i]
                )
                x3d, y3d = center_3d
            else:
                # Original MonoDETR path, kept separate from the new decoder.
                x3d = dets[i, j, 34] * info['img_size'][i][0]
                y3d = dets[i, j, 35] * info['img_size'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            orientation_x = x if use_legacy_ry_path else x3d
            ry = calibs[i].alpha2ry(alpha, orientation_x)


            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs(
        outputs, K=50, topk=50, outside_center_decode_mode='legacy',
        return_debug=False):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]
    
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)


    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4

    representative_center = boxes[:, :, 0:2]
    decode_mode = validate_outside_center_decode_mode(
        outside_center_decode_mode
    )
    predicted_offset = None
    if decode_mode != 'legacy':
        if 'pred_outside_center_offset' not in outputs:
            raise KeyError(
                "pred_outside_center_offset is required for decode mode '{}'."
                .format(decode_mode)
            )
        predicted_offset = torch.gather(
            outputs['pred_outside_center_offset'],
            1,
            topk_boxes.repeat(1, 1, 2))
    projected_center, used_offset = select_projected_center(
        representative_center, predicted_offset, decode_mode
    )

    xs3d = projected_center[:, :, 0:1]
    ys3d = projected_center[:, :, 1:2]

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)

    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    if not return_debug:
        return detections

    debug = {
        'decode_mode': decode_mode,
        'query_ids': topk_boxes.squeeze(-1),
        'class_ids': labels.squeeze(-1),
        'scores': scores.squeeze(-1),
        'representative_center_norm': representative_center,
        'predicted_offset_norm': predicted_offset,
        'used_offset_norm': used_offset,
        'decoded_center_norm': projected_center,
        'depth': depth.squeeze(-1),
        'dimensions': size_3d,
        'angle_head': heading,
    }
    return detections, debug


def extract_dedicated_dets_from_outputs(
        outputs, normal_topk=50, outside_topk=10, return_debug=False):
    """Run independent top-k selection for normal and outside query groups."""
    required_keys = (
        'pred_outside_logits',
        'pred_outside_boxes',
        'pred_outside_3d_dim',
        'pred_outside_depth',
        'pred_outside_angle',
        'pred_outside_center_offset',
    )
    missing = [key for key in required_keys if key not in outputs]
    if missing:
        raise KeyError(
            "Dedicated outside-query outputs are missing: {}."
            .format(', '.join(missing))
        )

    normal_outputs = {
        'pred_logits': outputs['pred_logits'],
        'pred_boxes': outputs['pred_boxes'],
        'pred_3d_dim': outputs['pred_3d_dim'],
        'pred_depth': outputs['pred_depth'],
        'pred_angle': outputs['pred_angle'],
    }
    outside_outputs = {
        'pred_logits': outputs['pred_outside_logits'],
        'pred_boxes': outputs['pred_outside_boxes'],
        'pred_3d_dim': outputs['pred_outside_3d_dim'],
        'pred_depth': outputs['pred_outside_depth'],
        'pred_angle': outputs['pred_outside_angle'],
        'pred_outside_center_offset': outputs[
            'pred_outside_center_offset'
        ],
    }
    normal_topk = min(
        normal_topk,
        normal_outputs['pred_logits'].shape[1]
        * normal_outputs['pred_logits'].shape[2],
    )
    outside_topk = min(
        outside_topk,
        outside_outputs['pred_logits'].shape[1]
        * outside_outputs['pred_logits'].shape[2],
    )
    normal = extract_dets_from_outputs(
        normal_outputs,
        topk=normal_topk,
        outside_center_decode_mode='legacy',
        return_debug=return_debug,
    )
    outside = extract_dets_from_outputs(
        outside_outputs,
        topk=outside_topk,
        outside_center_decode_mode='full',
        return_debug=return_debug,
    )
    if not return_debug:
        return normal, outside
    normal_dets, normal_debug = normal
    outside_dets, outside_debug = outside
    normal_debug['query_group'] = 'normal'
    outside_debug['query_group'] = 'outside'
    return normal_dets, outside_dets, {
        'normal': normal_debug,
        'outside': outside_debug,
    }


def merge_dedicated_decoded_results(normal_results, outside_results):
    """Merge decoded groups per image and sort by final score."""
    merged = {}
    image_ids = set(normal_results) | set(outside_results)
    for image_id in image_ids:
        rows = (
            list(normal_results.get(image_id, []))
            + list(outside_results.get(image_id, []))
        )
        merged[image_id] = sorted(
            rows, key=lambda row: row[-1], reverse=True
        )
    return merged


def count_cross_group_duplicates(
        normal_results, outside_results, thresholds=(0.5, 0.7, 0.8)):
    """Count same-class cross-group 2D overlaps without suppressing them."""
    counts = {threshold: 0 for threshold in thresholds}
    for image_id in set(normal_results) | set(outside_results):
        normal_rows = normal_results.get(image_id, [])
        outside_rows = outside_results.get(image_id, [])
        for normal_row in normal_rows:
            for outside_row in outside_rows:
                if int(normal_row[0]) != int(outside_row[0]):
                    continue
                normal_box = normal_row[2:6]
                outside_box = outside_row[2:6]
                left = max(normal_box[0], outside_box[0])
                top = max(normal_box[1], outside_box[1])
                right = min(normal_box[2], outside_box[2])
                bottom = min(normal_box[3], outside_box[3])
                intersection = max(0.0, right - left) * max(
                    0.0, bottom - top
                )
                normal_area = max(
                    0.0, normal_box[2] - normal_box[0]
                ) * max(0.0, normal_box[3] - normal_box[1])
                outside_area = max(
                    0.0, outside_box[2] - outside_box[0]
                ) * max(0.0, outside_box[3] - outside_box[1])
                union = normal_area + outside_area - intersection
                iou = intersection / union if union > 0 else 0.0
                for threshold in thresholds:
                    counts[threshold] += int(iou > threshold)
    return counts


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
