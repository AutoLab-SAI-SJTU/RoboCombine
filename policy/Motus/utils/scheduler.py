import torch
from typing import Optional
import math

class LambdaLinearScheduler:

    def __init__(self, optimizer, warm_up_steps: int, cycle_length: int, f_max: float=1.0, f_min: float=0.1, f_start: float=1e-06):
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.cycle_length = cycle_length
        self.f_max = f_max
        self.f_min = f_min
        self.f_start = f_start
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        assert warm_up_steps >= 0, 'warm_up_steps must be non-negative'
        assert cycle_length > warm_up_steps, 'cycle_length must be greater than warm_up_steps'
        assert f_max >= f_min, 'f_max must be >= f_min'
        assert f_start >= 0, 'f_start must be non-negative'

    def step(self):
        self.step_count += 1
        lr_multiplier = self.get_lr_multiplier(self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_multiplier

    def get_lr_multiplier(self, step: int) -> float:
        if step <= 0:
            return self.f_start
        elif step <= self.warm_up_steps:
            return self.f_start + (self.f_max - self.f_start) * step / self.warm_up_steps
        elif step < self.cycle_length:
            remaining_steps = self.cycle_length - step
            decay_steps = self.cycle_length - self.warm_up_steps
            return self.f_min + (self.f_max - self.f_min) * remaining_steps / decay_steps
        else:
            return self.f_min

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        return {'step_count': self.step_count, 'base_lr': self.base_lr, 'warm_up_steps': self.warm_up_steps, 'cycle_length': self.cycle_length, 'f_max': self.f_max, 'f_min': self.f_min, 'f_start': self.f_start}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict['base_lr']
        self.warm_up_steps = state_dict['warm_up_steps']
        self.cycle_length = state_dict['cycle_length']
        self.f_max = state_dict['f_max']
        self.f_min = state_dict['f_min']
        self.f_start = state_dict['f_start']

def create_scheduler(optimizer, config):
    scheduler_type = getattr(config.training, 'scheduler_type', 'cosine')
    if scheduler_type == 'linear':
        return LambdaLinearScheduler(optimizer, warm_up_steps=config.training.warmup_steps, cycle_length=config.training.cycle_length, f_max=config.training.f_max, f_min=config.training.f_min, f_start=getattr(config.training, 'f_start', 1e-06))
    elif scheduler_type == 'cosine':
        T_max = getattr(config.training, 'max_steps', 10000)
        eta_min = config.training.learning_rate * getattr(config.training, 'min_lr_ratio', 0.1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f'Unknown scheduler type: {scheduler_type}')
