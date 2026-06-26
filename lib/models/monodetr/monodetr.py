"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss


TARGET_ALIGNED_KEYS = {
    'labels',
    'boxes',
    'calibs',
    'depth',
    'size_2d',
    'size_3d',
    'src_size_3d',
    'heading_bin',
    'heading_res',
    'boxes_3d',
    'boxes_3d_representative',
    'representative_box_valid_mask',
    'outside_center_offset',
    'outside_center_mask',
    'projected_3d_center',
    'representative_center',
    'locations',
    'original_target_indices',
}


def split_targets_by_outside_mask(targets):
    """Split every target-aligned field without duplicating any GT."""
    inside_targets = []
    outside_targets = []
    for target in targets:
        target_count = len(target['labels'])
        outside_mask = target['outside_center_mask'].bool()
        if outside_mask.shape != (target_count,):
            raise ValueError(
                "outside_center_mask must align with labels, got {} for {} "
                "targets.".format(tuple(outside_mask.shape), target_count)
            )
        original_indices = target.get(
            'original_target_indices',
            torch.arange(target_count, device=outside_mask.device),
        )
        source = dict(target)
        source['original_target_indices'] = original_indices
        split_pair = []
        for mask in (~outside_mask, outside_mask):
            split_target = {}
            for key, value in source.items():
                if key in TARGET_ALIGNED_KEYS:
                    if not torch.is_tensor(value):
                        raise TypeError(
                            "Target-aligned field '{}' must be a tensor."
                            .format(key)
                        )
                    if value.ndim == 0 or value.shape[0] != target_count:
                        raise ValueError(
                            "Target-aligned field '{}' has shape {}, expected "
                            "first dimension {}.".format(
                                key, tuple(value.shape), target_count
                            )
                        )
                    split_target[key] = value[mask]
                else:
                    split_target[key] = value
            split_pair.append(split_target)
        inside_targets.append(split_pair[0])
        outside_targets.append(split_pair[1])
    return inside_targets, outside_targets


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11,
                 two_stage_dino=False, use_outside_center_modeling=False,
                 use_dedicated_outside_queries=False, num_outside_queries=10):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.group_num = group_num
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.use_outside_center_modeling = use_outside_center_modeling
        self.use_dedicated_outside_queries = use_dedicated_outside_queries
        self.num_outside_queries = num_outside_queries
        if self.use_dedicated_outside_queries:
            if not self.use_outside_center_modeling:
                raise ValueError(
                    "Dedicated outside queries require outside-center modeling."
                )
            if self.num_outside_queries != 10:
                raise ValueError("num_outside_queries must be exactly 10.")
            if two_stage or use_dab or two_stage_dino:
                raise NotImplementedError(
                    "Dedicated outside queries support the current one-stage "
                    "non-DAB MonoDETR configuration only."
                )
        if self.use_outside_center_modeling:
            self.outside_center_offset_embed = MLP(hidden_dim, hidden_dim, 2, 2)
            nn.init.constant_(self.outside_center_offset_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.outside_center_offset_embed.layers[-1].bias.data, 0)
        self.use_dab = use_dab

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
                if self.use_dedicated_outside_queries:
                    self.outside_query_embed = nn.Embedding(
                        self.num_outside_queries, hidden_dim * 2
                    )
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:        
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)   
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
            if self.use_outside_center_modeling:
                self.outside_center_offset_embed = _get_clones(
                    self.outside_center_offset_embed, num_pred
                )
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            if self.use_outside_center_modeling:
                self.outside_center_offset_embed = nn.ModuleList(
                    [self.outside_center_offset_embed for _ in range(num_pred)]
                )
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
                
            else:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight[:self.num_queries]         
                refanchor = self.refpoint_embed.weight[:self.num_queries]  
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1) 
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[:self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])
        
        transformer_args = (
            srcs,
            masks,
            pos,
            query_embeds,
            depth_pos_embed,
            depth_pos_embed_ip,
        )
        if self.use_dedicated_outside_queries:
            transformer_outputs = self.depthaware_transformer(
                *transformer_args,
                outside_query_embed=self.outside_query_embed.weight,
            )
        else:
            transformer_outputs = self.depthaware_transformer(
                *transformer_args
            )
        (
            hs,
            init_reference,
            inter_references,
            inter_references_dim,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = transformer_outputs[:6]

        normal_predictions = self._predict_from_decoder(
            hs,
            init_reference,
            inter_references,
            inter_references_dim,
            calibs,
            img_sizes,
            weighted_depth,
            include_offset=(
                self.use_outside_center_modeling
                and not self.use_dedicated_outside_queries
            ),
        )
        (
            outputs_class,
            outputs_coord,
            outputs_3d_dim,
            outputs_depth,
            outputs_angle,
            outputs_outside_center_offset,
        ) = normal_predictions

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        if outputs_outside_center_offset is not None:
            out['pred_outside_center_offset'] = outputs_outside_center_offset[-1]

        if self.use_dedicated_outside_queries:
            if len(transformer_outputs) != 7:
                raise RuntimeError(
                    "Transformer did not return dedicated outside decoder outputs."
                )
            (
                outside_hs,
                outside_init_reference,
                outside_inter_references,
                outside_inter_references_dim,
            ) = transformer_outputs[6]
            outside_predictions = self._predict_from_decoder(
                outside_hs,
                outside_init_reference,
                outside_inter_references,
                outside_inter_references_dim,
                calibs,
                img_sizes,
                weighted_depth,
                include_offset=True,
            )
            (
                outside_class,
                outside_coord,
                outside_3d_dim,
                outside_depth,
                outside_angle,
                outside_offset,
            ) = outside_predictions
            out.update({
                'pred_outside_logits': outside_class[-1],
                'pred_outside_boxes': outside_coord[-1],
                'pred_outside_3d_dim': outside_3d_dim[-1],
                'pred_outside_depth': outside_depth[-1],
                'pred_outside_angle': outside_angle[-1],
                'pred_outside_center_offset': outside_offset[-1],
            })

        if self.aux_loss:
            if outputs_outside_center_offset is not None:
                out['aux_outputs'] = self._set_aux_loss_with_outside_center_offset(
                    outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,
                    outputs_depth, outputs_outside_center_offset)
            else:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,
                    outputs_depth)
            if self.use_dedicated_outside_queries:
                out['aux_outputs_outside'] = self._set_aux_loss_outside(
                    outside_class,
                    outside_coord,
                    outside_3d_dim,
                    outside_angle,
                    outside_depth,
                    outside_offset,
                )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out #, mask_dict

    def _predict_from_decoder(
            self, hs, init_reference, inter_references,
            inter_references_dim, calibs, img_sizes, weighted_depth,
            include_offset):
        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
        outputs_offsets = [] if include_offset else None

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

          
            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1),
                outputs_center3d,
                mode='bilinear',
                align_corners=True).squeeze(1)

            # depth average + sigma
            depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                    depth_reg[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

            if include_offset:
                outputs_offsets.append(
                    self.outside_center_offset_embed[lvl](hs[lvl])
                )

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        outputs_offset = (
            torch.stack(outputs_offsets) if outputs_offsets is not None else None
        )
        return (
            outputs_class,
            outputs_coord,
            outputs_3d_dim,
            outputs_depth,
            outputs_angle,
            outputs_offset,
        )

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_outside_center_offset(
            self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle,
            outputs_depth, outputs_outside_center_offset):
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e,
                 'pred_outside_center_offset': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[:-1], outputs_coord[:-1],
                    outputs_3d_dim[:-1], outputs_angle[:-1],
                                         outputs_depth[:-1], outputs_outside_center_offset[:-1])]

    @torch.jit.unused
    def _set_aux_loss_outside(
            self, outputs_class, outputs_coord, outputs_3d_dim,
            outputs_angle, outputs_depth, outputs_offset):
        return [{
            'pred_outside_logits': a,
            'pred_outside_boxes': b,
            'pred_outside_3d_dim': c,
            'pred_outside_angle': d,
            'pred_outside_depth': e,
            'pred_outside_center_offset': f,
        } for a, b, c, d, e, f in zip(
            outputs_class[:-1],
            outputs_coord[:-1],
            outputs_3d_dim[:-1],
            outputs_angle[:-1],
            outputs_depth[:-1],
            outputs_offset[:-1],
        )]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 group_num=11, use_outside_center_modeling=False,
                 use_dedicated_outside_queries=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num
        self.use_outside_center_modeling = use_outside_center_modeling
        self.use_dedicated_outside_queries = use_dedicated_outside_queries
        self.last_match_stats = {}

    def _target_box_key(self, target_box_key=None):
        if target_box_key is not None:
            return target_box_key
        return (
            'boxes_3d_representative'
            if self.use_outside_center_modeling
            else 'boxes_3d'
        )

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(
            self, outputs, targets, indices, num_boxes,
            target_box_key=None, **kwargs):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_box_key = self._target_box_key(target_box_key)
        target_3dcenter = torch.cat(
            [t[target_box_key][:, 0: 2][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(
            self, outputs, targets, indices, num_boxes,
            target_box_key=None, box_valid_mask_key=None, **kwargs):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_box_key = self._target_box_key(target_box_key)
        if target_box_key == 'boxes_3d':
            target_2dboxes = torch.cat(
                [t['boxes_3d'][:, 2: 6][i]
                 for t, (_, i) in zip(targets, indices)],
                dim=0)

            # l1
            loss_bbox = F.l1_loss(
                src_2dboxes, target_2dboxes, reduction='none')
            losses = {}
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # giou
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat(
                [t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)],
                dim=0)
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcylrtb_to_xyxy(src_boxes),
                box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
            return losses

        target_boxes = torch.cat(
            [t[target_box_key][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0)
        if box_valid_mask_key is None:
            box_valid_mask_key = 'representative_box_valid_mask'
        valid_mask = torch.cat(
            [t[box_valid_mask_key][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0).bool()
        src_boxes = outputs['pred_boxes'][idx]
        if not valid_mask.any():
            zero_loss = src_boxes.sum() * 0.0
            return {'loss_bbox': zero_loss, 'loss_giou': zero_loss}

        loss_bbox = F.l1_loss(
            src_2dboxes[valid_mask],
            target_boxes[valid_mask, 2:6],
            reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes[valid_mask]),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes[valid_mask])))
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    def loss_outside_center_offset(
            self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_offsets = outputs['pred_outside_center_offset'][idx]
        target_offsets = torch.cat(
            [t['outside_center_offset'][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0)
        outside_mask = torch.cat(
            [t['outside_center_mask'][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0).bool()

        num_outside = outside_mask.sum().to(dtype=src_offsets.dtype)
        normalizer = num_outside.detach().clone()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(normalizer)
        normalizer = torch.clamp(
            normalizer / get_world_size(), min=1.0)

        if not outside_mask.any():
            return {
                'loss_outside_center_offset': src_offsets.sum() * 0.0
            }

        loss = torch.log1p(torch.abs(
            src_offsets[outside_mask] - target_offsets[outside_mask]))
        return {
            'loss_outside_center_offset': loss.sum() / normalizer
        }

    def loss_inside_center_offset_zero(
            self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_offsets = outputs['pred_outside_center_offset'][idx]
        outside_mask = torch.cat(
            [t['outside_center_mask'][i]
             for t, (_, i) in zip(targets, indices)],
            dim=0).bool()
        inside_mask = ~outside_mask

        num_inside = inside_mask.sum().to(dtype=src_offsets.dtype)
        normalizer = num_inside.detach().clone()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(normalizer)
        normalizer = torch.clamp(
            normalizer / get_world_size(), min=1.0)

        if not inside_mask.any():
            return {
                'loss_inside_center_offset_zero': src_offsets.sum() * 0.0
            }

        loss = torch.log1p(torch.abs(src_offsets[inside_mask]))
        return {
            'loss_inside_center_offset_zero': loss.sum() / normalizer
        }

    def loss_depths(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        if src_dims.numel() == 0:
            return {'loss_dim': src_dims.sum() * 0.0}
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(
            heading_target_cls.shape[0],
            12,
            device=heading_input.device,
            dtype=heading_input.dtype,
        ).scatter_(
            dim=1, index=heading_target_cls.view(-1, 1), value=1
        )
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']
        device = depth_map_logits.device

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat(
            [t['boxes'] for t in targets], dim=0
        ) * torch.tensor(
            [80, 24, 80, 24],
            device=device,
            dtype=depth_map_logits.dtype,
        )
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        
        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'outside_center_offset': self.loss_outside_center_offset,
            'inside_center_offset_zero': self.loss_inside_center_offset_zero,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        geometry_kwargs = {}
        if loss in ('boxes', 'center'):
            geometry_kwargs = {
                key: kwargs[key]
                for key in ('target_box_key', 'box_valid_mask_key')
                if key in kwargs
            }
        if loss == 'labels' and 'log' in kwargs:
            geometry_kwargs['log'] = kwargs['log']
        return loss_map[loss](
            outputs, targets, indices, num_boxes, **geometry_kwargs
        )

    @staticmethod
    def _outside_outputs_as_standard(outputs):
        key_map = {
            'pred_outside_logits': 'pred_logits',
            'pred_outside_boxes': 'pred_boxes',
            'pred_outside_3d_dim': 'pred_3d_dim',
            'pred_outside_depth': 'pred_depth',
            'pred_outside_angle': 'pred_angle',
            'pred_outside_center_offset': 'pred_outside_center_offset',
        }
        return {
            standard_key: outputs[outside_key]
            for outside_key, standard_key in key_map.items()
            if outside_key in outputs
        }

    @staticmethod
    def _outside_loss_name(name):
        if name == 'loss_outside_center_offset':
            return name
        if name.startswith('loss_'):
            return 'loss_outside_' + name[len('loss_'):]
        return 'outside_' + name

    @staticmethod
    def _normalized_target_count(targets, multiplier, device):
        count = sum(len(target["labels"]) for target in targets) * multiplier
        count = torch.as_tensor([count], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(count)
        return torch.clamp(
            count / get_world_size(), min=1
        ).item()

    def _compute_losses(
            self, outputs, targets, indices, num_boxes, losses,
            target_box_key, box_valid_mask_key=None, log_labels=True):
        result = {}
        for loss in losses:
            kwargs = {
                'target_box_key': target_box_key,
                'box_valid_mask_key': box_valid_mask_key,
            }
            if loss == 'labels':
                kwargs['log'] = log_labels
            result.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_boxes, **kwargs
                )
            )
        return result

    def _forward_dedicated(self, outputs, targets):
        inside_targets, outside_targets = split_targets_by_outside_mask(targets)
        group_num = self.group_num if self.training else 1
        device = outputs['pred_logits'].device

        normal_outputs = {
            key: value for key, value in outputs.items()
            if key not in (
                'aux_outputs',
                'aux_outputs_outside',
                'pred_outside_logits',
                'pred_outside_boxes',
                'pred_outside_3d_dim',
                'pred_outside_depth',
                'pred_outside_angle',
                'pred_outside_center_offset',
            )
        }
        outside_outputs = self._outside_outputs_as_standard(outputs)
        normal_indices = self.matcher(
            normal_outputs,
            inside_targets,
            group_num=group_num,
            target_box_key='boxes_3d',
        )
        outside_indices = self.matcher(
            outside_outputs,
            outside_targets,
            group_num=1,
            target_box_key='boxes_3d_representative',
            box_valid_mask_key='representative_box_valid_mask',
        )
        normal_num_boxes = self._normalized_target_count(
            inside_targets, group_num, device
        )
        outside_num_boxes = self._normalized_target_count(
            outside_targets, 1, device
        )

        query_losses = [
            'labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles',
            'center',
        ]
        losses = self._compute_losses(
            normal_outputs,
            inside_targets,
            normal_indices,
            normal_num_boxes,
            query_losses,
            target_box_key='boxes_3d',
        )
        outside_losses = self._compute_losses(
            outside_outputs,
            outside_targets,
            outside_indices,
            outside_num_boxes,
            query_losses + ['outside_center_offset'],
            target_box_key='boxes_3d_representative',
            box_valid_mask_key='representative_box_valid_mask',
        )
        losses.update({
            self._outside_loss_name(key): value
            for key, value in outside_losses.items()
        })

        # The dense depth map is shared by both query groups and is supervised
        # once using the complete, non-duplicated target set.
        losses.update(
            self.get_loss(
                'depth_map',
                normal_outputs,
                targets,
                normal_indices,
                self._normalized_target_count(targets, 1, device),
            )
        )

        if 'aux_outputs' in outputs:
            for layer_index, auxiliary in enumerate(outputs['aux_outputs']):
                auxiliary_indices = self.matcher(
                    auxiliary,
                    inside_targets,
                    group_num=group_num,
                    target_box_key='boxes_3d',
                )
                layer_losses = self._compute_losses(
                    auxiliary,
                    inside_targets,
                    auxiliary_indices,
                    normal_num_boxes,
                    query_losses,
                    target_box_key='boxes_3d',
                    log_labels=False,
                )
                losses.update({
                    key + '_{}'.format(layer_index): value
                    for key, value in layer_losses.items()
                })

        if 'aux_outputs_outside' in outputs:
            for layer_index, auxiliary in enumerate(
                    outputs['aux_outputs_outside']):
                standard_auxiliary = self._outside_outputs_as_standard(
                    auxiliary
                )
                auxiliary_indices = self.matcher(
                    standard_auxiliary,
                    outside_targets,
                    group_num=1,
                    target_box_key='boxes_3d_representative',
                    box_valid_mask_key='representative_box_valid_mask',
                )
                layer_losses = self._compute_losses(
                    standard_auxiliary,
                    outside_targets,
                    auxiliary_indices,
                    outside_num_boxes,
                    query_losses + ['outside_center_offset'],
                    target_box_key='boxes_3d_representative',
                    box_valid_mask_key='representative_box_valid_mask',
                    log_labels=False,
                )
                losses.update({
                    self._outside_loss_name(key)
                    + '_{}'.format(layer_index): value
                    for key, value in layer_losses.items()
                })

        self.last_match_stats = {
            'num_inside_gt': sum(len(t['labels']) for t in inside_targets),
            'num_outside_gt': sum(len(t['labels']) for t in outside_targets),
            'num_normal_matches': sum(
                len(source) for source, _ in normal_indices
            ),
            'num_outside_matches': sum(
                len(source) for source, _ in outside_indices
            ),
            'normal_original_target_indices': [
                target['original_target_indices'][target_indices]
                for target, (_, target_indices)
                in zip(inside_targets, normal_indices)
            ],
            'outside_original_target_indices': [
                target['original_target_indices'][target_indices]
                for target, (_, target_indices)
                in zip(outside_targets, outside_indices)
            ],
            'normal_indices': normal_indices,
            'outside_indices': outside_indices,
        }
        return losses

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self.use_dedicated_outside_queries:
            return self._forward_dedicated(outputs, targets)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab = cfg['use_dab'],
        two_stage_dino=cfg['two_stage_dino'],
        use_outside_center_modeling=cfg.get(
            'use_outside_center_modeling', False
        ),
        use_dedicated_outside_queries=cfg.get(
            'use_dedicated_outside_queries', False
        ),
        num_outside_queries=cfg.get('num_outside_queries', 10))

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    use_outside_center_modeling = cfg.get(
        'use_outside_center_modeling', False
    )
    use_dedicated_outside_queries = cfg.get(
        'use_dedicated_outside_queries', False
    )
    if use_outside_center_modeling and not use_dedicated_outside_queries:
        weight_dict['loss_outside_center_offset'] = cfg[
            'outside_center_offset_loss_coef'
        ]
        weight_dict['loss_inside_center_offset_zero'] = cfg[
            'inside_center_offset_zero_loss_coef'
        ]
    if use_dedicated_outside_queries:
        dedicated_weights = cfg.get('dedicated_outside_loss', {})
        weight_dict.update({
            'loss_outside_ce': dedicated_weights.get(
                'class_coef', cfg['cls_loss_coef']
            ),
            'loss_outside_bbox': dedicated_weights.get(
                'bbox_coef', cfg['bbox_loss_coef']
            ),
            'loss_outside_giou': dedicated_weights.get(
                'giou_coef', cfg['giou_loss_coef']
            ),
            'loss_outside_dim': dedicated_weights.get(
                'dim_coef', cfg['dim_loss_coef']
            ),
            'loss_outside_angle': dedicated_weights.get(
                'angle_coef', cfg['angle_loss_coef']
            ),
            'loss_outside_depth': dedicated_weights.get(
                'depth_coef', cfg['depth_loss_coef']
            ),
            'loss_outside_center': dedicated_weights.get(
                'center_coef', cfg['3dcenter_loss_coef']
            ),
            'loss_outside_center_offset': dedicated_weights.get(
                'offset_coef', cfg['outside_center_offset_loss_coef']
            ),
        })
    
    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        encoder_weight_dict = {
            k: v for k, v in weight_dict.items()
            if (
                not k.startswith('loss_outside_')
                and k != 'loss_inside_center_offset_zero'
            )
        }
        aux_weight_dict.update(
            {k + f'_enc': v for k, v in encoder_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
    if use_outside_center_modeling and not use_dedicated_outside_queries:
        losses.append('outside_center_offset')
        losses.append('inside_center_offset_zero')
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        use_outside_center_modeling=use_outside_center_modeling,
        use_dedicated_outside_queries=use_dedicated_outside_queries)

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
