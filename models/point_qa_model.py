import os
import sys
import torch
import argparse
import numpy as np
from typing import Dict, List, Any, Callable, Union, Sequence, Mapping
from collections import OrderedDict

from .base_qa_model import QAModel, QAModelInstance, load_point_cloud
from .dependence.utils import ponder_collate_fn

point_qa_models = {
    "3dllava": ("ThreeDLLava"),
    "shapellm": ("ShapeLLM"),
    "pointllm": ("PointLLM"),
}

def list_point_qa_models() -> List[str]:
    return list(point_qa_models.keys())

class PointQAModel(QAModel):
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        prompt_name: str = "default",
        prompt_func: Callable = None,
        choice_format: str = 'letter',
        cache_path: str = None,
        device: str = None,
        **kwargs,
    ):
        if prompt_func is None:
            prompt_func = self._default_prompt_func

        super().__init__(
            model_name=model_name,
            prompt_name=prompt_name,
            prompt_func=prompt_func,
            choice_format=choice_format,
            enable_choice_search=True,
            cache_path=cache_path,
        )

        if model_name not in point_qa_models:
            raise ValueError(f"Unknown point QA model: {model_name}")

        model_class_name = point_qa_models[model_name]
        print(f"Loading {model_name}...")
        if isinstance(model_class_name, (tuple, list)):
            model_class_name = model_class_name[0]
        if isinstance(model_class_name, str):
            ModelClass = globals().get(model_class_name)
            if ModelClass is None:
                raise ValueError(f"Model class '{model_class_name}' not found in globals().")
        else:
            ModelClass = model_class_name
        runtime_kwargs = dict(kwargs)
        runtime_kwargs.setdefault('checkpoint_path', checkpoint_path)
        runtime_kwargs.setdefault('device', device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model = ModelClass(**runtime_kwargs)

    @staticmethod
    def _default_prompt_func(question: str, options: List[str] = None) -> str:
        if not options:
            return question
        options_text = "\n".join(options)
        return f"{question}\n\n{options_text}\n\nPlease answer with the letter of the correct option."

    def _data_to_str(self, data: Dict[str, Any]) -> str:
        if 'point_cloud_path' in data:
            return data['point_cloud_path']
        if 'point_cloud' in data:
            pc = load_point_cloud(data['point_cloud'])
            return str(hash(pc.tobytes()))
        return "unknown"


class ThreeDLLava(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('checkpoint_path')
        if self.model_path is None:
            raise ValueError("3D-LLaVA requires --test_ckpt to point to the llava checkpoint path")
        self.model_base = kwargs.get('llava_model_base') or kwargs.get('model_base')
        self.pointcloud_tower_name = kwargs.get('llava_pointcloud_tower_name')
        self.temperature = kwargs.get('llava_temperature', 1.0)
        self.top_p = kwargs.get('llava_top_p', None)
        self.num_beams = kwargs.get('llava_num_beams', 1)
        self.max_new_tokens = kwargs.get('llava_max_new_tokens', 64)
        self.conv_mode = kwargs.get('llava_conv_mode', 'vicuna_v1')

        from models.dependence.threedllava.llava.model.builder import load_pretrained_model
        from models.dependence.threedllava.llava.mm_utils import tokenizer_special_token, get_model_name_from_path
        from models.dependence.threedllava.llava.utils import disable_torch_init
        from models.dependence.threedllava.llava.pc_utils import vqa_transform_eval, Compose

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer_special_token = tokenizer_special_token
        self.tokenizer, self.model, _, _ = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            pointcloud_tower_name=self.pointcloud_tower_name,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Use official transform pipeline
        self.transform = Compose(vqa_transform_eval)

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> Dict[str, Any]:
        """Prepare point cloud dict compatible with 3D-LLaVA pipeline."""
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Point cloud must be (N, 3+) shape for 3D-LLaVA, but got {pc.shape}")

        coord = pc[:, :3].astype(np.float32)
        if pc.shape[1] >= 6:
            color = pc[:, 3:6].astype(np.float32)
            if color.max() > 2.0:
                color = color / 255.0
        else:
            color = np.zeros_like(coord, dtype=np.float32)

        pc_dict = dict(
            coord=coord,
            color=color,
            superpoint_mask=np.zeros(coord.shape[0], dtype=np.int64),
        )

        pc_dict = self.transform(pc_dict)
        pc_dict['condition'] = 'textgen'

        # Ensure tensors
        for key in ['coord', 'grid_coord', 'feat', 'offset']:
            if key in pc_dict and not torch.is_tensor(pc_dict[key]):
                pc_dict[key] = torch.tensor(pc_dict[key])
        if not torch.is_tensor(pc_dict['superpoint_mask']):
            pc_dict['superpoint_mask'] = torch.tensor(pc_dict['superpoint_mask'])

        return pc_dict

    def _prepare_generation_inputs(self, pc_data: Dict[str, Any]):
        for key in ['coord', 'grid_coord', 'feat', 'offset']:
            pc_data[key] = ponder_collate_fn([pc_data[key]])

        grid_coords = pc_data['grid_coord']
        grid_coords = torch.cat([
            torch.zeros(grid_coords.shape[0], 1, dtype=grid_coords.dtype),
            grid_coords
        ], dim=1)
        pc_data['grid_coord'] = grid_coords

        try:
            from pointgroup_ops import voxelization_idx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pointgroup_ops is required for 3D-LLaVA evaluation."
            ) from exc

        batch_size = 1
        spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).cpu().numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

        return {
            'coord': pc_data['coord'].to(self.device),
            'grid_coord': voxel_coords.to(self.device),
            'offset': pc_data['offset'].to(self.device),
            'feat': pc_data['feat'].to(self.device),
            'p2v_map': p2v_map.to(self.device),
            'v2p_map': v2p_map.to(self.device),
            'spatial_shape': spatial_shape,
            'superpoint_mask': [pc_data['superpoint_mask'].to(self.device)],
            'conditions': [pc_data['condition']],
        }

    def _generate_text(self, prompt: str, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> str:
        pc_data = self._prepare_point_cloud(point_cloud)
        gen_inputs = self._prepare_generation_inputs(pc_data)
        input_ids = self.tokenizer_special_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                coord=gen_inputs['coord'],
                grid_coord=gen_inputs['grid_coord'],
                offset=gen_inputs['offset'],
                feat=gen_inputs['feat'],
                p2v_map=gen_inputs['p2v_map'],
                v2p_map=gen_inputs['v2p_map'],
                spatial_shape=gen_inputs['spatial_shape'],
                superpoint_mask=gen_inputs['superpoint_mask'],
                conditions=gen_inputs['conditions'],
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                min_length=1,
                max_new_tokens=self.max_new_tokens,
                tokenizer=self.tokenizer,
                click_mask=[[]],
                use_cache=True,
            )

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return decoded

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = data.get('point_cloud') or data.get('point_cloud_path')
        if point_cloud is None:
            raise ValueError('Point cloud is required for 3D-LLaVA evaluation')

        return self._generate_text(prompt, point_cloud)


class ShapeLLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('checkpoint_path')
        if self.model_path is None:
            raise ValueError("ShapeLLM requires checkpoint_path")
        
        self.model_base = kwargs.get('model_base')
        self.conv_mode = kwargs.get('conv_mode', 'llava_v1')
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_k = kwargs.get('top_k', 1)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 2048)

        try:
            from models.dependence.shapellm.llava.utils import disable_torch_init
            from models.dependence.shapellm.llava.model.builder import load_pretrained_model
            from models.dependence.shapellm.llava.mm_utils import tokenizer_point_token, get_model_name_from_path, load_pts, process_pts
            from models.dependence.shapellm.llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
            from models.dependence.shapellm.llava.conversation import conv_templates, SeparatorStyle

            self.tokenizer_point_token = tokenizer_point_token
            self.load_pts = load_pts
            self.process_pts = process_pts
            self.POINT_TOKEN_INDEX = POINT_TOKEN_INDEX
            self.DEFAULT_POINT_TOKEN = DEFAULT_POINT_TOKEN
            self.DEFAULT_PT_START_TOKEN = DEFAULT_PT_START_TOKEN
            self.DEFAULT_PT_END_TOKEN = DEFAULT_PT_END_TOKEN
            self.conv_templates = conv_templates
            self.SeparatorStyle = SeparatorStyle
        except ImportError as exc:
            raise ImportError("ShapeLLM dependencies missing. Please ensure 'llava' and related packages are installed and importable as a Python package (e.g. in models.dependence.llava)") from exc

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str], point_path: str = None):
        if point_path:
            point_cloud = point_path
        point = load_point_cloud(point_cloud)
        
        pts_tensor = self.process_pts(point, self.model.config).unsqueeze(0)
        
        return pts_tensor.to(self.device, dtype=torch.float16)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = data.get('point_cloud')
        point_path = data.get('point_cloud_path')
        if point_cloud is None and point_path is None:
            raise ValueError('Point cloud is required for ShapeLLM evaluation')

        if self.model.config.mm_use_pt_start_end:
            qs = self.DEFAULT_PT_START_TOKEN + self.DEFAULT_POINT_TOKEN + self.DEFAULT_PT_END_TOKEN + '\n' + prompt
        else:
            qs = self.DEFAULT_POINT_TOKEN + '\n' + prompt

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self.tokenizer_point_token(
            full_prompt, self.tokenizer, self.POINT_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        pts_tensor = self._prepare_point_cloud(point_cloud, point_path)
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                points=pts_tensor,
                do_sample=self.temperature > 0 and self.num_beams == 1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs.strip()


class PointLLM(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = kwargs.get('checkpoint_path')
        if self.model_path is None:
            raise ValueError("PointLLM requires checkpoint_path")
        
        self.conv_mode = kwargs.get('conv_mode', 'vicuna_v1_1')
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', None)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)

        from models.dependence.pointllm.model import PointLLMLlamaForCausalLM  
        from models.dependence.pointllm.conversation import conv_templates 
        from models.dependence.pointllm.utils import disable_torch_init  
        from models.dependence.pointllm.data import pc_norm 
        from transformers import AutoTokenizer

        self.conv_templates = conv_templates
        self.pc_norm = pc_norm
        
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = PointLLMLlamaForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        self.model.initialize_tokenizer_point_backbone_config(self.tokenizer, device=self.device, fix_llm=True)

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor, str]) -> torch.Tensor:
        pc = load_point_cloud(point_cloud)
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        
        pc = self.pc_norm(pc)
        return torch.from_numpy(pc).float().to(self.device)

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        point_cloud = data.get('point_cloud') or data.get('point_cloud_path')
        if point_cloud is None:
            raise ValueError('Point cloud is required for PointLLM evaluation')

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids.to(self.device)
        point_tensor = self._prepare_point_cloud(point_cloud)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_tensor.unsqueeze(0),
                do_sample=self.temperature > 0 and self.num_beams == 1,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        return response



def create_point_qa_model(model_name: str, checkpoint_path: str = None, **kwargs) -> PointQAModel:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for point QA models")
    return PointQAModel(model_name=model_name, checkpoint_path=checkpoint_path, **kwargs)