import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def _is_outside_center_offset_key(key):
    return (
        key.startswith('outside_center_offset_embed.')
        or key.startswith('module.outside_center_offset_embed.')
    )


def _is_outside_query_key(key):
    return (
        key.startswith('outside_query_embed.')
        or key.startswith('module.outside_query_embed.')
    )


def _is_allowed_new_model_key(key):
    return _is_outside_center_offset_key(key) or _is_outside_query_key(key)


def _load_optimizer_with_new_offset_head(optimizer, optimizer_state, model):
    current_state = optimizer.state_dict()
    if len(current_state['param_groups']) != len(optimizer_state['param_groups']):
        raise RuntimeError(
            "Cannot migrate optimizer state: parameter group count changed."
        )

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    parameter_names = {
        id(parameter): name for name, parameter in base_model.named_parameters()
    }
    migrated_state = {'state': {}, 'param_groups': []}

    for group_index, (current_group, saved_group) in enumerate(zip(
            optimizer.param_groups, optimizer_state['param_groups'])):
        current_ids = current_state['param_groups'][group_index]['params']
        saved_ids = saved_group['params']
        current_legacy_ids = []
        for parameter, parameter_id in zip(current_group['params'], current_ids):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "Cannot migrate optimizer state: optimizer parameter is "
                    "not present in the model."
                )
            if not _is_outside_center_offset_key(name):
                current_legacy_ids.append(parameter_id)

        if len(current_legacy_ids) != len(saved_ids):
            raise RuntimeError(
                "Cannot migrate optimizer state: non-offset parameter count "
                "changed in group {}.".format(group_index)
            )

        for saved_id, current_id in zip(saved_ids, current_legacy_ids):
            if saved_id in optimizer_state['state']:
                migrated_state['state'][current_id] = optimizer_state['state'][saved_id]

        migrated_group = dict(saved_group)
        migrated_group['params'] = current_ids
        migrated_state['param_groups'].append(migrated_group)

    optimizer.load_state_dict(migrated_state)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        loaded_legacy_without_offset_head = False
        loaded_legacy_without_outside_queries = False
        if model is not None and checkpoint['model_state'] is not None:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            if getattr(base_model, 'use_outside_center_modeling', False):
                incompatible = model.load_state_dict(
                    checkpoint['model_state'], strict=False
                )
                missing_keys = incompatible.missing_keys
                unexpected_keys = incompatible.unexpected_keys
                invalid_missing_keys = [
                    key for key in missing_keys
                    if not _is_allowed_new_model_key(key)
                ]
                if invalid_missing_keys or unexpected_keys:
                    raise RuntimeError(
                        "Checkpoint is incompatible with MonoDETR. "
                        "Missing keys: {}. Unexpected keys: {}.".format(
                            invalid_missing_keys, unexpected_keys
                        )
                    )
                if missing_keys:
                    loaded_legacy_without_offset_head = any(
                        _is_outside_center_offset_key(key)
                        for key in missing_keys
                    )
                    loaded_legacy_without_outside_queries = any(
                        _is_outside_query_key(key) for key in missing_keys
                    )
                    logger.info(
                        "Allowed missing model keys: %s",
                        sorted(missing_keys),
                    )
                    if loaded_legacy_without_offset_head:
                        logger.info(
                            "Initialized missing outside-center offset head "
                            "from its zero-initialized output layer."
                        )
                    if loaded_legacy_without_outside_queries:
                        logger.info(
                            "Initialized outside_query_embed with the model's "
                            "default nn.Embedding initialization."
                        )
                loaded_count = (
                    len(base_model.state_dict())
                    - len(missing_keys)
                )
                logger.info(
                    "Successfully loaded %d model tensors; unexpected keys: %s",
                    loaded_count,
                    sorted(unexpected_keys),
                )
            else:
                model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            if (
                    loaded_legacy_without_offset_head
                    and not loaded_legacy_without_outside_queries):
                _load_optimizer_with_new_offset_head(
                    optimizer, checkpoint['optimizer_state'], model
                )
                logger.info(
                    "Migrated legacy optimizer state; the new outside-center "
                    "offset head starts with empty optimizer state."
                )
            elif loaded_legacy_without_outside_queries:
                raise RuntimeError(
                    "Cannot migrate a legacy optimizer state to dedicated "
                    "outside-query parameter groups. Load this checkpoint as "
                    "pretrain_model (optimizer=None) or resume a checkpoint "
                    "created with dedicated outside queries."
                )
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch
