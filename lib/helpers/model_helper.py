from lib.models.monodetr import build_monodetr


def build_model(cfg, dataset_cfg=None):
    model_cfg = dict(cfg)
    if dataset_cfg is not None:
        model_cfg['use_outside_center_modeling'] = dataset_cfg.get(
            'use_outside_center_modeling', False
        )
    return build_monodetr(model_cfg)
