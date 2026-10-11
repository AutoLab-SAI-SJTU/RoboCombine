import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, List
import random

def resize_with_padding(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_size
    original_height, original_width = frame.shape[:2]
    scale_height = target_height / original_height
    scale_width = target_width / original_width
    scale = min(scale_height, scale_width)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    return padded_frame

def load_video_frames(video_path: str, frame_indices: List[int], target_size: Tuple[int, int]=None) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frame_idx in frame_indices:
            if frame_idx >= total_frames:
                raise ValueError(f'Frame index {frame_idx} out of bounds for video {video_path} (total frames: {total_frames})')
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f'Failed to read frame {frame_idx} from {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                if frame.shape[:2] != target_size:
                    frame = resize_with_padding(frame, target_size)
            frames.append(frame)
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
        return video_tensor
    finally:
        cap.release()

def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames
    finally:
        cap.release()

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.shape[0] == 3:
        image_np = tensor.permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np, mode='RGB')
    else:
        raise ValueError(f'Unsupported tensor shape: {tensor.shape}')

def apply_image_augmentation(frame: np.ndarray, brightness_prob: float=0.5, brightness_range: Tuple[float, float]=(0.8, 1.2), flip_prob: float=0.3) -> np.ndarray:
    if random.random() < brightness_prob:
        brightness_factor = random.uniform(*brightness_range)
        frame = np.clip(frame * brightness_factor, 0, 255)
    if random.random() < flip_prob:
        frame = np.fliplr(frame)
    return frame
