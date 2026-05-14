from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Tuple
import logging
logger = logging.getLogger(__name__)

def preprocess_vlm_messages(text_instruction: str, image_pil, processor):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image_pil}, {'type': 'text', 'text': text_instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    return inputs
