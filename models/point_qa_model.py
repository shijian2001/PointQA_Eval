import os
import sys
import torch
import argparse
import numpy as np
from typing import Dict, List, Any, Callable, Union, Sequence, Mapping
from collections import OrderedDict

from .base_qa_model import QAModel, QAModelInstance

DEPENDENCIES_PATH = os.path.join(os.path.dirname(__file__), 'dependence', '3d-r1')
if DEPENDENCIES_PATH not in sys.path:
    sys.path.insert(0, DEPENDENCIES_PATH)

LLAVA_DEPENDENCIES_PATH = os.path.join(os.path.dirname(__file__), 'dependence', '3d-llava', 'llava')
LLAVA_ROOT_PATH = os.path.join(os.path.dirname(__file__), 'dependence', '3d-llava')
if LLAVA_ROOT_PATH not in sys.path:
    sys.path.insert(0, LLAVA_ROOT_PATH)

SHAPELLM_PATH = os.path.join(os.path.dirname(__file__), 'dependence', 'shapellm')
if SHAPELLM_PATH not in sys.path:
    sys.path.insert(0, SHAPELLM_PATH)

point_qa_models = {
    "3dr1": None,
    "3dllava": None,
    "shapellm": None,
}

def ponder_collate_fn(batch, max_point=-1):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for batch_id, data in enumerate(batch):
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches, max_point)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch_dict = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch_dict.keys():
            if "offset" in key:
                batch_dict[key] = torch.cumsum(batch_dict[key], dim=0)
        return batch_dict
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

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
            enable_choice_search=False,
            cache_path=cache_path,
        )

        if model_name not in point_qa_models:
            raise ValueError(f"Unknown point QA model: {model_name}")

        ModelClass = point_qa_models[model_name]
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
            pc = data['point_cloud']
            if isinstance(pc, torch.Tensor):
                pc = pc.cpu().numpy()
            return str(hash(pc.tobytes()))
        return "unknown"


class ThreeDR1(QAModelInstance):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = kwargs.get('checkpoint_path')
        
        try:
            import importlib.util
            
            model_path = os.path.join(DEPENDENCIES_PATH, 'models', 'model_general.py')
            spec = importlib.util.spec_from_file_location("model_general", model_path)
            model_general = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_general)
            self.CaptionNet = model_general.CaptionNet
            
            dataset_path = os.path.join(DEPENDENCIES_PATH, 'dataset', 'scannet_base_dataset.py')
            spec = importlib.util.spec_from_file_location("scannet_base_dataset", dataset_path)
            scannet_base_dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scannet_base_dataset)
            self.DatasetConfig = scannet_base_dataset.DatasetConfig
        except Exception as e:
            raise ImportError(f"Failed to import 3D-R1 modules: {e}. Ensure code is in: {DEPENDENCIES_PATH}")

        args = self._build_args(**kwargs)
        dataset_config = self._safe_dataset_config()
        self.model = self.CaptionNet(args, dataset_config, None).to(self.device)
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._load_checkpoint(self.model, self.checkpoint_path)
        else:
            print(f"[WARNING] Checkpoint not found: {self.checkpoint_path}")
            
        self.model.eval()
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, trust_remote_code=True)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab, trust_remote_code=True)

    def _build_args(self, **kwargs):
        args = argparse.Namespace(
            dataset=kwargs.get('dataset', 'scannet'),
            test_only=kwargs.get('test_only', True),
            use_color=kwargs.get('use_color', True),
            use_normal=kwargs.get('use_normal', True),
            no_height=kwargs.get('no_height', False),
            use_multiview=kwargs.get('use_multiview', False),
            detector=kwargs.get('detector', 'point_encoder'),
            captioner=kwargs.get('captioner', '3dr1'),
            vocab=kwargs.get('vocab', 'Qwen/Qwen2.5-7B'),
            qformer_vocab=kwargs.get('qformer_vocab', 'google-bert/bert-base-uncased'),
            use_additional_encoders=kwargs.get('use_additional_encoders', False),
            use_depth=kwargs.get('use_depth', False),
            use_image=kwargs.get('use_image', False),
            depth_encoder_dim=kwargs.get('depth_encoder_dim', 256),
            image_encoder_dim=kwargs.get('image_encoder_dim', 256),
            enable_dynamic_views=kwargs.get('enable_dynamic_views', False),
            view_selection_weight=kwargs.get('view_selection_weight', 0.1),
            use_pytorch3d_rendering=kwargs.get('use_pytorch3d_rendering', False),
            use_multimodal_model=kwargs.get('use_multimodal_model', False),
            max_des_len=kwargs.get('max_des_len', 256),
            max_gen_len=kwargs.get('max_gen_len', 512),
            use_beam_search=kwargs.get('use_beam_search', False),
            freeze_detector=kwargs.get('freeze_detector', True),
            freeze_llm=kwargs.get('freeze_llm', False),
            checkpoint_dir=kwargs.get('checkpoint_dir', './results'),
        )
        return args

    def _safe_dataset_config(self):
        try:
            return self.DatasetConfig()
        except FileNotFoundError as exc:
            print(f"[WARNING] ScanNet metadata not found: {exc}. Using fallback dataset config.")
            return self._fallback_dataset_config()

    def _fallback_dataset_config(self):
        type2class = {
            'cabinet': 0,
            'bed': 1,
            'chair': 2,
            'sofa': 3,
            'table': 4,
            'door': 5,
            'window': 6,
            'bookshelf': 7,
            'picture': 8,
            'counter': 9,
            'desk': 10,
            'curtain': 11,
            'refrigerator': 12,
            'shower curtain': 13,
            'toilet': 14,
            'sink': 15,
            'bathtub': 16,
            'others': 17,
        }
        class MinimalConfig:
            def __init__(self):
                self.num_semcls = 18
                self.num_angle_bin = 1
                self.max_num_obj = 128
                self.type2class = type2class.copy()
                self.class2type = {v: k for k, v in self.type2class.items()}
                self.nyu40ids = np.array([
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 36, 37, 38, 39, 40,
                ])
                self.nyu40id2class = {nyu40id: self.type2class.get('others', 17) for nyu40id in self.nyu40ids}
        return MinimalConfig()

    def _load_checkpoint(self, model, path):
        print(f"[INFO] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] Checkpoint loaded")

    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        self.model.eval()
        model_inp = self._prepare_input(data, prompt)
        
        with torch.no_grad():
            outputs = self.model(model_inp, is_eval=True, task_name="qa")
        
        output_ids = outputs.get("output_ids", None)
        if output_ids is None:
            return ""
        
        decoded = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return decoded[0] if decoded else ""
    
    def _prepare_input(self, data: Dict[str, Any], prompt: str) -> Dict[str, torch.Tensor]:
        point_cloud = data.get('point_cloud')
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float()
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        point_cloud = point_cloud.to(self.device)
        
        if point_cloud.shape[-1] >= 6:
            xyz, rgb = point_cloud[..., :3], point_cloud[..., 3:6]
        else:
            xyz, rgb = point_cloud[..., :3], torch.zeros_like(point_cloud[..., :3])
        
        qformer_inputs = self.qformer_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        instruction_inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        
        return {
            'point_clouds': xyz,
            'point_clouds_color': rgb,
            'point_cloud_dims_min': xyz.min(dim=1)[0],
            'point_cloud_dims_max': xyz.max(dim=1)[0],
            'qformer_input_ids': qformer_inputs['input_ids'].to(self.device),
            'qformer_attention_mask': qformer_inputs['attention_mask'].to(self.device),
            'instruction': instruction_inputs['input_ids'].to(self.device),
            'instruction_mask': instruction_inputs['attention_mask'].to(self.device),
        }


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
        self.voxel_size = kwargs.get('llava_voxel_size', 0.02)
        self.conv_mode = kwargs.get('llava_conv_mode', 'vicuna_v1')

        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_special_token, get_model_name_from_path
            from llava.utils import disable_torch_init
        except ImportError as exc:
            raise ImportError("3D-LLaVA dependencies are missing. Please ensure the llava repo is available under models/dependence/3d-llava") from exc

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

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float()
        if point_cloud.dim() != 2 or point_cloud.shape[1] < 3:
            raise ValueError("Point cloud must be (N, 3+) shape for 3D-LLaVA")
        coord = point_cloud[:, :3]
        if point_cloud.shape[1] >= 6:
            color = point_cloud[:, 3:6]
            if color.max() > 2:
                color = color / 255.0
        else:
            color = torch.zeros_like(coord)

        grid_coord = torch.floor(coord / self.voxel_size).long()
        superpoint_mask = torch.zeros(coord.shape[0], dtype=torch.int64)
        feat = torch.cat([coord, color], dim=1)
        offset = torch.tensor([coord.shape[0]], dtype=torch.int64)
        return {
            'coord': coord,
            'color': color,
            'grid_coord': grid_coord,
            'superpoint_mask': superpoint_mask,
            'condition': 'textgen',
            'feat': feat,
            'offset': offset,
        }

    def _build_prompt(self, question_text: str) -> str:
        return question_text

    def _prepare_generation_inputs(self, pc_data: Dict[str, torch.Tensor]):
        for key in ['coord', 'grid_coord', 'feat', 'offset']:
            pc_data[key] = ponder_collate_fn([pc_data[key]])

        grid_coords = pc_data['grid_coord']
        grid_coords = torch.cat([torch.zeros(grid_coords.shape[0], 1, dtype=grid_coords.dtype), grid_coords], dim=1)
        pc_data['grid_coord'] = grid_coords

        try:
            from pointgroup_ops import voxelization_idx
        except ImportError as exc:
            raise ImportError("pointgroup_ops is required for 3D-LLaVA evaluation") from exc

        batch_size = 1
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, batch_size, 4)

        spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).cpu().numpy(), 128, None)

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

    def _generate_text(self, prompt: str, point_cloud: Union[np.ndarray, torch.Tensor]) -> str:
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
        question_text = data.get('question_text', prompt)
        point_cloud = data.get('point_cloud')
        if point_cloud is None:
            raise ValueError('Point cloud is required for 3D-LLaVA evaluation')

        final_prompt = self._build_prompt(question_text)
        return self._generate_text(final_prompt, point_cloud)


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
            from llava.utils import disable_torch_init
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_point_token, get_model_name_from_path, load_pts, process_pts
            from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            
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
            raise ImportError(f"ShapeLLM dependencies missing. Ensure llava is in: {SHAPELLM_PATH}") from exc

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_answer(self, response: str, options: List[str]) -> str:
        response = response.strip().upper()
        if response in ['A', 'B', 'C', 'D']:
            return response
        for i, option in enumerate(options):
            parts = option.split('. ', 1)
            if len(parts) > 1:
                option_text = parts[1].strip().lower()
                if option_text in response.lower():
                    return chr(ord('A') + i)
        for i, option in enumerate(options):
            words = option.lower().split()[1:] if len(option.split()) > 1 else []
            if any(word in response.lower() for word in words):
                return chr(ord('A') + i)
        return ""

    def _prepare_point_cloud(self, point_cloud: Union[np.ndarray, torch.Tensor], point_path: str = None):
        if point_path and os.path.exists(point_path):
            point = self.load_pts(point_path)
        elif isinstance(point_cloud, np.ndarray):
            point = point_cloud
        elif isinstance(point_cloud, torch.Tensor):
            point = point_cloud.cpu().numpy()
        else:
            raise ValueError("Invalid point cloud input")
        
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


point_qa_models["3dr1"] = ThreeDR1
point_qa_models["3dllava"] = ThreeDLLava
point_qa_models["shapellm"] = ShapeLLM


def create_point_qa_model(model_name: str, checkpoint_path: str = None, **kwargs) -> PointQAModel:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for point QA models")
    return PointQAModel(model_name=model_name, checkpoint_path=checkpoint_path, **kwargs)