import os
from typing import List, NamedTuple, Tuple, Optional
import torch
import logging
logger = logging.getLogger(__name__)

class IncompatibleKeys(NamedTuple):
    missing_keys: List[str]
    unexpected_keys: List[str]
    incorrect_shapes: List[Tuple[str, Tuple[int], Tuple[int]]]

def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> IncompatibleKeys:
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            model_param = model_state_dict[k]
            if not isinstance(model_param, torch.Tensor):
                logger.warning(f'Skipping non-tensor parameter {k}')
                continue
            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                logger.warning(f'Shape mismatch for {k}: model {shape_model} vs checkpoint {shape_checkpoint}')
                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    return IncompatibleKeys(missing_keys=incompatible.missing_keys, unexpected_keys=incompatible.unexpected_keys, incorrect_shapes=incorrect_shapes)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int, save_path: str, additional_state: dict=None) -> None:
    state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': iteration}
    if additional_state:
        state_dict.update(additional_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = save_path + '.tmp'
    torch.save(state_dict, temp_path)
    os.rename(temp_path, save_path)
    logger.info(f'Checkpoint saved to {save_path}')

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]=None, strict: bool=True, map_location: str='cpu') -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    if strict:
        model.load_state_dict(state_dict['model'])
    else:
        incompatible = non_strict_load_model(model, state_dict['model'])
        if incompatible.missing_keys:
            logger.warning(f'Missing keys: {incompatible.missing_keys}')
        if incompatible.unexpected_keys:
            logger.warning(f'Unexpected keys: {incompatible.unexpected_keys}')
        if incompatible.incorrect_shapes:
            logger.warning(f'Incorrect shapes: {incompatible.incorrect_shapes}')
    if optimizer is not None and 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler is not None and 'scheduler' in state_dict:
        scheduler.load_state_dict(state_dict['scheduler'])
    logger.info('Checkpoint loaded successfully')
    additional_state = {k: v for k, v in state_dict.items() if k not in ['model', 'optimizer', 'scheduler']}
    return additional_state
