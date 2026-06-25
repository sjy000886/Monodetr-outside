import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
            mask_dict=None
            #ipdb.set_trace()
            detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            detr_losses.backward()
            if batch_idx % 30 == 0:
                base_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                if getattr(
                        base_model,
                        'use_dedicated_outside_queries',
                        False):
                    stats = getattr(
                        self.detr_loss, 'last_match_stats', {}
                    )
                    normal_total = sum(
                        value
                        for key, value in detr_losses_dict_log.items()
                        if (
                            key.startswith('loss_')
                            and not key.startswith('loss_outside_')
                            and key != 'loss_detr'
                        )
                    )
                    outside_total = sum(
                        value
                        for key, value in detr_losses_dict_log.items()
                        if key.startswith('loss_outside_')
                    )
                    print(
                        "dedicated_stats num_inside_gt={} num_outside_gt={} "
                        "num_normal_matches={} num_outside_matches={} "
                        "loss_normal_total={:.4f} loss_outside_total={:.4f} "
                        "loss_outside_center_offset={:.4f}".format(
                            stats.get('num_inside_gt', 0),
                            stats.get('num_outside_gt', 0),
                            stats.get('num_normal_matches', 0),
                            stats.get('num_outside_matches', 0),
                            normal_total,
                            outside_total,
                            detr_losses_dict_log.get(
                                'loss_outside_center_offset', 0.0
                            ),
                        )
                    )
                    print(
                        "dedicated_grad_norms normal_query={:.6f} "
                        "outside_query={:.6f} outside_offset_head={:.6f}"
                        .format(
                            self._gradient_norm(
                                base_model.query_embed.parameters()
                            ),
                            self._gradient_norm(
                                base_model.outside_query_embed.parameters()
                            ),
                            self._gradient_norm(
                                base_model.outside_center_offset_embed.parameters()
                            ),
                        )
                    )
            self.optimizer.step()

            progress_bar.update()
            max_train_batches = self.cfg.get('max_train_batches', -1)
            if (
                    max_train_batches is not None
                    and max_train_batches >= 0
                    and batch_idx + 1 >= max_train_batches):
                break
        progress_bar.close()

    @staticmethod
    def _gradient_norm(parameters):
        squared_norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().norm()) ** 2
        return squared_norm ** 0.5

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        if 'boxes_3d_representative' in targets:
            key_list.extend([
                'boxes_3d_representative',
                'representative_box_valid_mask',
                'outside_center_offset',
                'outside_center_mask',
                'projected_3d_center',
                'representative_center',
            ])
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
