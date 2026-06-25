import os
import tqdm
import shutil

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import extract_dedicated_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.decode_helper import merge_dedicated_decoded_results
from lib.helpers.decode_helper import count_cross_group_duplicates
from lib.helpers.decode_helper import resolve_outside_center_decode_mode
import time


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.use_outside_center_modeling = getattr(
            base_model, 'use_outside_center_modeling', False
        )
        self.use_dedicated_outside_queries = getattr(
            base_model, 'use_dedicated_outside_queries', False
        )
        configured_decode_mode = cfg.get('outside_center_decode_mode')
        self.outside_center_decode_mode = (
            resolve_outside_center_decode_mode(
                self.use_outside_center_modeling,
                configured_decode_mode,
            )
        )
        self.result_tag = cfg.get('result_tag')
        self.checkpoint_path = cfg.get('checkpoint_path')
        self.max_test_batches = cfg.get('max_test_batches', -1)
        self.debug_decode_batches = cfg.get('debug_decode_batches', 0)
        self.normal_topk = cfg.get('normal_topk', cfg.get('topk', 50))
        self.outside_topk = cfg.get('outside_topk', 10)
        if self.result_tag:
            self.result_root = os.path.join(
                self.output_dir, 'outputs_{}'.format(self.result_tag)
            )
        else:
            self.result_root = os.path.join(self.output_dir, 'outputs')
        self.result_data_dir = os.path.join(self.result_root, 'data')
        self.logger.info(
            "Outside-center decode mode: %s; dedicated queries: %s; "
            "result directory: %s",
            self.outside_center_decode_mode,
            self.use_dedicated_outside_queries,
            self.result_data_dir,
        )

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.checkpoint_path:
                checkpoint_path = self.checkpoint_path
            elif self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            if self.max_test_batches is None or self.max_test_batches < 0:
                self.evaluate()
            else:
                self.logger.info(
                    "Skipping KITTI AP evaluation for limited smoke inference "
                    "(max_test_batches=%s).",
                    self.max_test_batches,
                )

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        if os.path.isdir(self.result_data_dir):
            shutil.rmtree(self.result_data_dir)
        os.makedirs(self.result_data_dir, exist_ok=True)

        results = {}
        dedicated_stats = {
            'normal_kept': 0,
            'outside_kept': 0,
            'merged_detections': 0,
            'cross_group_iou_gt_0.5': 0,
            'cross_group_iou_gt_0.7': 0,
            'cross_group_iou_gt_0.8': 0,
        }
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            threshold = self.cfg.get('threshold', 0.2)
            debug_enabled = batch_idx < self.debug_decode_batches
            if self.use_dedicated_outside_queries:
                extracted = extract_dedicated_dets_from_outputs(
                    outputs,
                    normal_topk=self.normal_topk,
                    outside_topk=self.outside_topk,
                    return_debug=debug_enabled,
                )
                if debug_enabled:
                    normal_dets, outside_dets, decode_debug = extracted
                else:
                    normal_dets, outside_dets = extracted
                    decode_debug = None
                normal_results = decode_detections(
                    normal_dets.detach().cpu().numpy(),
                    info,
                    calibs,
                    cls_mean_size,
                    threshold,
                    outside_center_decode_mode='legacy',
                )
                outside_results = decode_detections(
                    outside_dets.detach().cpu().numpy(),
                    info,
                    calibs,
                    cls_mean_size,
                    threshold,
                    outside_center_decode_mode='full',
                )
                decoded = merge_dedicated_decoded_results(
                    normal_results, outside_results
                )
                duplicates = count_cross_group_duplicates(
                    normal_results, outside_results
                )
                normal_kept = sum(map(len, normal_results.values()))
                outside_kept = sum(map(len, outside_results.values()))
                dedicated_stats['normal_kept'] += normal_kept
                dedicated_stats['outside_kept'] += outside_kept
                dedicated_stats['merged_detections'] += sum(
                    map(len, decoded.values())
                )
                for duplicate_threshold, count in duplicates.items():
                    dedicated_stats[
                        'cross_group_iou_gt_{}'.format(duplicate_threshold)
                    ] += count
                if decode_debug is not None:
                    self.logger.info(
                        "dedicated_decode_debug batch=%d normal_queries=%s "
                        "outside_queries=%s normal_kept=%d outside_kept=%d",
                        batch_idx,
                        decode_debug['normal']['query_ids'][0].tolist(),
                        decode_debug['outside']['query_ids'][0].tolist(),
                        normal_kept,
                        outside_kept,
                    )
            else:
                extracted = extract_dets_from_outputs(
                    outputs=outputs,
                    K=self.max_objs,
                    topk=self.cfg['topk'],
                    outside_center_decode_mode=self.outside_center_decode_mode,
                    return_debug=debug_enabled,
                )
                if debug_enabled:
                    dets, decode_debug = extracted
                else:
                    dets = extracted
                    decode_debug = None
                decoded = decode_detections(
                    dets=dets.detach().cpu().numpy(),
                    info=info,
                    calibs=calibs,
                    cls_mean_size=cls_mean_size,
                    threshold=threshold,
                    outside_center_decode_mode=self.outside_center_decode_mode)
                if decode_debug is not None:
                    self._log_decode_debug(
                        batch_idx, decode_debug, decoded, info
                    )

            results.update(decoded)
            progress_bar.update()
            if (
                    self.max_test_batches is not None
                    and self.max_test_batches >= 0
                    and batch_idx + 1 >= self.max_test_batches):
                break

        print("inference on {} images by {}/per image".format(
            len(results), model_infer_time / max(len(results), 1)))

        progress_bar.close()
        if self.use_dedicated_outside_queries:
            self.logger.info(
                "Dedicated inference stats: normal_kept=%d outside_kept=%d "
                "merged_detections=%d cross_group_iou_gt_0.5=%d "
                "cross_group_iou_gt_0.7=%d cross_group_iou_gt_0.8=%d",
                dedicated_stats['normal_kept'],
                dedicated_stats['outside_kept'],
                dedicated_stats['merged_detections'],
                dedicated_stats['cross_group_iou_gt_0.5'],
                dedicated_stats['cross_group_iou_gt_0.7'],
                dedicated_stats['cross_group_iou_gt_0.8'],
            )
            self.last_dedicated_inference_stats = dedicated_stats

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)
        return results

    def _log_decode_debug(self, batch_idx, debug, decoded_results, info):
        image_id = int(info['img_id'][0])
        decoded_rows = decoded_results.get(image_id, [])
        threshold = self.cfg.get('threshold', 0.2)
        valid_indices = [
            index for index in range(debug['query_ids'].shape[1])
            if float(debug['scores'][0, index]) >= threshold
        ][:3]
        image_size = info['img_size'][0]
        for row_index, index in enumerate(valid_indices):
            predicted_offset = debug['predicted_offset_norm']
            predicted_offset_value = (
                None
                if predicted_offset is None
                else predicted_offset[0, index].detach().cpu().tolist()
            )
            decoded_center = debug['decoded_center_norm'][
                0, index].detach().cpu()
            decoded_center_px = (
                decoded_center.numpy() * image_size
            ).tolist()
            row = (
                decoded_rows[row_index]
                if row_index < len(decoded_rows)
                else None
            )
            self.logger.info(
                "decode_debug batch=%d mode=%s query=%d class=%d "
                "score=%.6f representative=%s predicted_offset=%s "
                "used_offset=%s decoded=%s decoded_original_px=%s depth=%.6f "
                "location=%s alpha=%s rotation_y=%s",
                batch_idx,
                debug['decode_mode'],
                int(debug['query_ids'][0, index]),
                int(debug['class_ids'][0, index]),
                float(debug['scores'][0, index]),
                debug['representative_center_norm'][
                    0, index].detach().cpu().tolist(),
                predicted_offset_value,
                debug['used_offset_norm'][
                    0, index].detach().cpu().tolist(),
                decoded_center.tolist(),
                decoded_center_px,
                float(debug['depth'][0, index]),
                None if row is None else row[9:12],
                None if row is None else row[1],
                None if row is None else row[12],
            )

    def save_results(self, results):
        output_dir = getattr(
            self,
            'result_data_dir',
            os.path.join(self.output_dir, 'outputs', 'data'),
        )
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = self.result_data_dir
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result
