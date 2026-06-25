import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_optimizer(cfg_optimizer, model):
    base_model = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    dedicated = getattr(
        base_model, 'use_dedicated_outside_queries', False
    )
    weights, biases = [], []
    outside_query_weights = []
    outside_offset_weights, outside_offset_biases = [], []
    names_by_group = {
        'base_bias': [],
        'base_weight': [],
        'outside_query': [],
        'outside_offset_bias': [],
        'outside_offset_weight': [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = (
            name[len('module.'):] if name.startswith('module.') else name
        )
        if dedicated and clean_name.startswith('outside_query_embed.'):
            outside_query_weights.append(param)
            names_by_group['outside_query'].append(name)
        elif dedicated and clean_name.startswith(
                'outside_center_offset_embed.'):
            if 'bias' in clean_name:
                outside_offset_biases.append(param)
                names_by_group['outside_offset_bias'].append(name)
            else:
                outside_offset_weights.append(param)
                names_by_group['outside_offset_weight'].append(name)
        elif 'bias' in clean_name:
            biases += [param]
            names_by_group['base_bias'].append(name)
        else:
            weights += [param]
            names_by_group['base_weight'].append(name)

    parameters = [
        {
            'params': biases,
            'weight_decay': 0,
            'lr': cfg_optimizer['lr'],
            'group_name': 'base_bias',
        },
        {
            'params': weights,
            'weight_decay': cfg_optimizer['weight_decay'],
            'lr': cfg_optimizer['lr'],
            'group_name': 'base_weight',
        },
    ]
    if dedicated:
        outside_query_lr = cfg_optimizer['outside_query_lr']
        outside_offset_lr = cfg_optimizer['outside_offset_head_lr']
        parameters.extend([
            {
                'params': outside_query_weights,
                'weight_decay': cfg_optimizer['weight_decay'],
                'lr': outside_query_lr,
                'group_name': 'outside_query',
            },
            {
                'params': outside_offset_biases,
                'weight_decay': 0,
                'lr': outside_offset_lr,
                'group_name': 'outside_offset_bias',
            },
            {
                'params': outside_offset_weights,
                'weight_decay': cfg_optimizer['weight_decay'],
                'lr': outside_offset_lr,
                'group_name': 'outside_offset_weight',
            },
        ])
    parameters = [group for group in parameters if group['params']]

    seen = set()
    for group in parameters:
        for parameter in group['params']:
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise RuntimeError(
                    "A model parameter was assigned to multiple optimizer groups."
                )
            seen.add(parameter_id)
    trainable = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad
    }
    if seen != trainable:
        raise RuntimeError(
            "Optimizer parameter groups do not cover the trainable model "
            "parameters exactly."
        )

    for group in parameters:
        name = group['group_name']
        parameter_count = sum(
            parameter.numel() for parameter in group['params']
        )
        print(
            "optimizer_group={} tensors={} parameters={} lr={} "
            "weight_decay={}".format(
                name,
                len(group['params']),
                parameter_count,
                group['lr'],
                group['weight_decay'],
            )
        )
        print("  names={}".format(', '.join(names_by_group[name])))

    if cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    elif cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'adamw':
        optimizer = AdamW(parameters, lr=cfg_optimizer['lr'])
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss
