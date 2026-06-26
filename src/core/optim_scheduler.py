import json
import warnings
from torch import optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
try:
    from timm.optim.nadam import NAdamLegacy as Nadam
except ImportError:
    from timm.optim.nadam import Nadam
    warnings.warn("NAdamLegacy not found, falling back to Nadam (decoupled weight decay may differ)")
from timm.optim.nvnovograd import NvNovoGrad
try:
    from timm.optim.radam import RAdamLegacy as RAdam
except ImportError:
    from timm.optim.radam import RAdam
    warnings.warn("RAdamLegacy not found, falling back to RAdam (decoupled weight decay may differ)")
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler

from .optim_constant import optim_parameters


timm_schedulers = [
    CosineLRScheduler, MultiStepLRScheduler, StepLRScheduler, TanhLRScheduler
]


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def split_backbone_param_groups(parameters, model, lr, lr_backbone, weight_decay):
    """Split existing optimizer groups by backbone membership while preserving group options."""
    param_to_name = {
        id(p): name
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    if isinstance(parameters, list):
        source_groups = parameters
    else:
        source_groups = [{'params': list(parameters), 'weight_decay': weight_decay}]

    split_groups = []
    for group in source_groups:
        base_group = {k: v for k, v in group.items() if k != 'params'}
        backbone_params = []
        other_params = []

        for param in group['params']:
            name = param_to_name.get(id(param), '')
            if 'backbone' in name.lower():
                backbone_params.append(param)
            else:
                other_params.append(param)

        if backbone_params:
            backbone_group = dict(base_group)
            backbone_group.update(params=backbone_params, lr=lr_backbone)
            split_groups.append(backbone_group)
        if other_params:
            other_group = dict(base_group)
            other_group.update(params=other_params, lr=lr)
            split_groups.append(other_group)

    return split_groups


def get_optim_scheduler(args, epoch, model, steps_per_epoch):
    opt_lower = (args.opt or 'adam').lower()
    weight_decay = args.weight_decay if args.weight_decay is not None else 1e-4

    # if weight_decay and filter_bias_and_bn:
    if args.filter_bias_and_bn:
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        else:
            skip = {}
        parameters = get_parameter_groups(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # Backbone low-LR param group: configs set lr_backbone (e.g. 1e-5) but
    # previously it was only used to gate trainability.  Honour it by
    # splitting existing groups by backbone/non-backbone with distinct LRs.
    lr_backbone = getattr(args, 'lr_backbone', None)
    if lr_backbone and lr_backbone > 0 and lr_backbone != args.lr:
        parameters = split_backbone_param_groups(
            parameters, model, args.lr, lr_backbone, weight_decay
        )
        weight_decay = 0.  # already applied per-group

    opt_args = optim_parameters.get(opt_lower, dict())
    opt_args.update(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    sched_lower = args.sched.lower()
    total_steps = epoch * steps_per_epoch
    by_epoch = True
    if sched_lower == 'onecycle':
        max_lr = (
            [group['lr'] for group in optimizer.param_groups]
            if len(optimizer.param_groups) > 1
            else args.lr
        )
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=getattr(args, 'final_div_factor', 1e4))
        by_epoch = False
    elif sched_lower == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True,  # update lr by_epoch
            k_decay=getattr(args, 'lr_k_decay', 1.0))
    elif sched_lower == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True)  # update lr by_epoch
    elif sched_lower == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    elif sched_lower == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epoch,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch)
    else:
        assert False and "Invalid scheduler"

    return optimizer, lr_scheduler, by_epoch
