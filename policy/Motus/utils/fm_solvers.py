import inspect
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils import deprecate, is_scipy_available
from diffusers.utils.torch_utils import randn_tensor
if is_scipy_available():
    pass

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma

def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError('Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values')
    if timesteps is not None:
        accepts_timesteps = 'timesteps' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = 'sigmas' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return (timesteps, num_inference_steps)

class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int=1000, solver_order: int=2, prediction_type: str='flow_prediction', shift: Optional[float]=1.0, use_dynamic_shifting=False, thresholding: bool=False, dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0, algorithm_type: str='dpmsolver++', solver_type: str='midpoint', lower_order_final: bool=True, euler_at_final: bool=False, final_sigmas_type: Optional[str]='zero', lambda_min_clipped: float=-float('inf'), variance_type: Optional[str]=None, invert_sigmas: bool=False):
        if algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
            deprecation_message = f'algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead'
            deprecate('algorithm_types dpmsolver and sde-dpmsolver', '1.0.0', deprecation_message)
        if algorithm_type not in ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver', 'sde-dpmsolver++']:
            if algorithm_type == 'deis':
                self.register_to_config(algorithm_type='dpmsolver++')
            else:
                raise NotImplementedError(f'{algorithm_type} is not implemented for {self.__class__}')
        if solver_type not in ['midpoint', 'heun']:
            if solver_type in ['logrho', 'bh1', 'bh2']:
                self.register_to_config(solver_type='midpoint')
            else:
                raise NotImplementedError(f'{solver_type} is not implemented for {self.__class__}')
        if algorithm_type not in ['dpmsolver++', 'sde-dpmsolver++'] and final_sigmas_type == 'zero':
            raise ValueError(f'`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead.')
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int=0):
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: Union[int, None]=None, device: Union[str, torch.device]=None, sigmas: Optional[List[float]]=None, mu: Optional[Union[float, None]]=None, shift: Optional[Union[float, None]]=None):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(' you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`')
        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        if self.config.final_sigmas_type == 'sigma_min':
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == 'zero':
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")
        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)
        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return (1 - sigma, sigma)

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(self, model_output: torch.Tensor, *args, sample: torch.Tensor=None, **kwargs) -> torch.Tensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop('timestep', None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError('missing `sample` as a required keyward argument')
        if timestep is not None:
            deprecate('timesteps', '1.0.0', 'Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        if self.config.algorithm_type in ['dpmsolver++', 'sde-dpmsolver++']:
            if self.config.prediction_type == 'flow_prediction':
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, `v_prediction`, or `flow_prediction` for the FlowDPMSolverMultistepScheduler.')
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred
        elif self.config.algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
            if self.config.prediction_type == 'flow_prediction':
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, `v_prediction` or `flow_prediction` for the FlowDPMSolverMultistepScheduler.')
            if self.config.thresholding:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred
            return epsilon

    def dpm_solver_first_order_update(self, model_output: torch.Tensor, *args, sample: torch.Tensor=None, noise: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop('timestep', None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep', None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(' missing `sample` as a required keyward argument')
        if timestep is not None:
            deprecate('timesteps', '1.0.0', 'Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        if prev_timestep is not None:
            deprecate('prev_timestep', '1.0.0', 'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        sigma_t, sigma_s = (self.sigmas[self.step_index + 1], self.sigmas[self.step_index])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        if self.config.algorithm_type == 'dpmsolver++':
            x_t = sigma_t / sigma_s * sample - alpha_t * (torch.exp(-h) - 1.0) * model_output
        elif self.config.algorithm_type == 'dpmsolver':
            x_t = alpha_t / alpha_s * sample - sigma_t * (torch.exp(h) - 1.0) * model_output
        elif self.config.algorithm_type == 'sde-dpmsolver++':
            assert noise is not None
            x_t = sigma_t / sigma_s * torch.exp(-h) * sample + alpha_t * (1 - torch.exp(-2.0 * h)) * model_output + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
        elif self.config.algorithm_type == 'sde-dpmsolver':
            assert noise is not None
            x_t = alpha_t / alpha_s * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
        return x_t

    def multistep_dpm_solver_second_order_update(self, model_output_list: List[torch.Tensor], *args, sample: torch.Tensor=None, noise: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        timestep_list = args[0] if len(args) > 0 else kwargs.pop('timestep_list', None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep', None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(' missing `sample` as a required keyward argument')
        if timestep_list is not None:
            deprecate('timestep_list', '1.0.0', 'Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        if prev_timestep is not None:
            deprecate('prev_timestep', '1.0.0', 'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        sigma_t, sigma_s0, sigma_s1 = (self.sigmas[self.step_index + 1], self.sigmas[self.step_index], self.sigmas[self.step_index - 1])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        m0, m1 = (model_output_list[-1], model_output_list[-2])
        h, h_0 = (lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        r0 = h_0 / h
        D0, D1 = (m0, 1.0 / r0 * (m0 - m1))
        if self.config.algorithm_type == 'dpmsolver++':
            if self.config.solver_type == 'midpoint':
                x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
            elif self.config.solver_type == 'heun':
                x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1
        elif self.config.algorithm_type == 'dpmsolver':
            if self.config.solver_type == 'midpoint':
                x_t = alpha_t / alpha_s0 * sample - sigma_t * (torch.exp(h) - 1.0) * D0 - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
            elif self.config.solver_type == 'heun':
                x_t = alpha_t / alpha_s0 * sample - sigma_t * (torch.exp(h) - 1.0) * D0 - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1
        elif self.config.algorithm_type == 'sde-dpmsolver++':
            assert noise is not None
            if self.config.solver_type == 'midpoint':
                x_t = sigma_t / sigma_s0 * torch.exp(-h) * sample + alpha_t * (1 - torch.exp(-2.0 * h)) * D0 + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1 + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            elif self.config.solver_type == 'heun':
                x_t = sigma_t / sigma_s0 * torch.exp(-h) * sample + alpha_t * (1 - torch.exp(-2.0 * h)) * D0 + alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0) * D1 + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
        elif self.config.algorithm_type == 'sde-dpmsolver':
            assert noise is not None
            if self.config.solver_type == 'midpoint':
                x_t = alpha_t / alpha_s0 * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0 - sigma_t * (torch.exp(h) - 1.0) * D1 + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            elif self.config.solver_type == 'heun':
                x_t = alpha_t / alpha_s0 * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0 - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1 + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
        return x_t

    def multistep_dpm_solver_third_order_update(self, model_output_list: List[torch.Tensor], *args, sample: torch.Tensor=None, **kwargs) -> torch.Tensor:
        timestep_list = args[0] if len(args) > 0 else kwargs.pop('timestep_list', None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep', None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(' missing`sample` as a required keyward argument')
        if timestep_list is not None:
            deprecate('timestep_list', '1.0.0', 'Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        if prev_timestep is not None:
            deprecate('prev_timestep', '1.0.0', 'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`')
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (self.sigmas[self.step_index + 1], self.sigmas[self.step_index], self.sigmas[self.step_index - 1], self.sigmas[self.step_index - 2])
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)
        m0, m1, m2 = (model_output_list[-1], model_output_list[-2], model_output_list[-3])
        h, h_0, h_1 = (lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2)
        r0, r1 = (h_0 / h, h_1 / h)
        D0 = m0
        D1_0, D1_1 = (1.0 / r0 * (m0 - m1), 1.0 / r1 * (m1 - m2))
        D1 = D1_0 + r0 / (r0 + r1) * (D1_0 - D1_1)
        D2 = 1.0 / (r0 + r1) * (D1_0 - D1_1)
        if self.config.algorithm_type == 'dpmsolver++':
            x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1 - alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5) * D2
        elif self.config.algorithm_type == 'dpmsolver':
            x_t = alpha_t / alpha_s0 * sample - sigma_t * (torch.exp(h) - 1.0) * D0 - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1 - sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5) * D2
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output: torch.Tensor, timestep: Union[int, torch.Tensor], sample: torch.Tensor, generator=None, variance_noise: Optional[torch.Tensor]=None, return_dict: bool=True) -> Union[SchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
        if self.step_index is None:
            self._init_step_index(timestep)
        lower_order_final = self.step_index == len(self.timesteps) - 1 and (self.config.euler_at_final or (self.config.lower_order_final and len(self.timesteps) < 15) or self.config.final_sigmas_type == 'zero')
        lower_order_second = self.step_index == len(self.timesteps) - 2 and self.config.lower_order_final and (len(self.timesteps) < 15)
        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        sample = sample.to(torch.float32)
        if self.config.algorithm_type in ['sde-dpmsolver', 'sde-dpmsolver++'] and variance_noise is None:
            noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32)
        elif self.config.algorithm_type in ['sde-dpmsolver', 'sde-dpmsolver++']:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == 'mps' and torch.is_floating_point(timesteps):
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
