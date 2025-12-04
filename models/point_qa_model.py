import os
import sys
import numpy as np
import torch
from typing import Dict, List, Any
from collections import OrderedDict

from .base_qa_model import QAModel, QAModelInstance

DEPENDENCIES_PATH = os.path.join(os.path.dirname(__file__), 'dependence', '3d-r1')
if DEPENDENCIES_PATH not in sys.path:
    sys.path.insert(0, DEPENDENCIES_PATH)


# ==================== Model Instances ====================

class ThreeDR1Instance(QAModelInstance):
    
    def __init__(self, model, tokenizer, qformer_tokenizer, device, max_gen_len=512):
        self.model = model
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        self.device = device
        self.max_gen_len = max_gen_len
    
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


# class NewModelInstance(QAModelInstance):
#     def __init__(self, model, tokenizer, device):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#     
#     def qa(self, data: Dict[str, Any], prompt: str) -> str:
#         # 实现模型推理逻辑
#         pass


# ==================== Point QA Model ====================

class PointQAModel(QAModel):
    
    SUPPORTED_MODELS = {
        "3dr1": {
            "instance_class": ThreeDR1Instance,
            "vocab": "Qwen/Qwen2.5-7B",
            "qformer_vocab": "google-bert/bert-base-uncased",
        },
        # "new_model": {
        #     "instance_class": NewModelInstance,
        #     "vocab": "path/to/vocab",
        # },
    }
    
    def __init__(
        self,
        model_name: str = "3dr1",
        checkpoint_path: str = None,
        prompt_name: str = "default",
        prompt_func: callable = None,
        choice_format: str = 'letter',
        cache_path: str = None,
        device: torch.device = None,
        **model_kwargs,
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
        
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs
        self.tokenizer = None
        self.qformer_tokenizer = None
        self._model_name = model_name
        
        self._load_model(**model_kwargs)
    
    @staticmethod
    def _default_prompt_func(question: str, options: List[str] = None) -> str:
        if options is None:
            return question
        options_text = "\n".join(options)
        return f"{question}\n\n{options_text}\n\nPlease answer with the letter of the correct option."
    
    def _load_model(self, **kwargs):
        print(f"[INFO] Loading model: {self._model_name}")
        print(f"[INFO] Device: {self.device}")
        
        if self._model_name == "3dr1":
            self._load_3dr1(**kwargs)
        # elif self._model_name == "new_model":
        #     self._load_new_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model: {self._model_name}")
    
    def _load_3dr1(self, **kwargs):
        try:
            # 确保 dependencies 路径在 sys.path 中，用于子模块导入
            if DEPENDENCIES_PATH not in sys.path:
                sys.path.insert(0, DEPENDENCIES_PATH)
            
            import importlib.util
            
            # 导入 CaptionNet
            model_path = os.path.join(DEPENDENCIES_PATH, 'models', 'model_general.py')
            spec = importlib.util.spec_from_file_location("model_general", model_path)
            model_general = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_general)
            CaptionNet = model_general.CaptionNet
            
            # 导入 DatasetConfig
            dataset_path = os.path.join(DEPENDENCIES_PATH, 'dataset', 'scannet_base_dataset.py')
            spec = importlib.util.spec_from_file_location("scannet_base_dataset", dataset_path)
            scannet_base_dataset = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scannet_base_dataset)
            DatasetConfig = scannet_base_dataset.DatasetConfig
        except Exception as e:
            raise ImportError(f"Failed to import 3D-R1 modules: {e}. Ensure code is in: {DEPENDENCIES_PATH}")
        
        args = self._build_3dr1_args(**kwargs)
        dataset_config = DatasetConfig()
        model = CaptionNet(args, dataset_config, None).to(self.device)
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._load_checkpoint(model)
        
        model.eval()
        self._load_tokenizers(args)
        
        self.model = ThreeDR1Instance(
            model=model,
            tokenizer=self.tokenizer,
            qformer_tokenizer=self.qformer_tokenizer,
            device=self.device,
            max_gen_len=kwargs.get('max_gen_len', 512),
        )
        print(f"[INFO] Model loaded successfully")
    
    def _build_3dr1_args(self, **kwargs):
        import argparse
        config = self.SUPPORTED_MODELS.get("3dr1", {})
        
        return argparse.Namespace(
            use_color=kwargs.get('use_color', True),
            use_normal=kwargs.get('use_normal', True),
            no_height=kwargs.get('no_height', False),
            use_multiview=kwargs.get('use_multiview', False),
            detector=kwargs.get('detector', 'point_encoder'),
            captioner=kwargs.get('captioner', '3dr1'),
            vocab=kwargs.get('vocab', config.get('vocab', 'Qwen/Qwen2.5-7B')),
            qformer_vocab=kwargs.get('qformer_vocab', config.get('qformer_vocab', 'google-bert/bert-base-uncased')),
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
    
    def _load_checkpoint(self, model: torch.nn.Module):
        print(f"[INFO] Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
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
    
    def _load_tokenizers(self, args):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, trust_remote_code=True)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab, trust_remote_code=True)
        print(f"[INFO] Tokenizers loaded")
    
    def _data_to_str(self, data: Dict[str, Any]) -> str:
        if 'point_cloud_path' in data:
            return data['point_cloud_path']
        if 'point_cloud' in data:
            pc = data['point_cloud']
            if isinstance(pc, torch.Tensor):
                pc = pc.cpu().numpy()
            return str(hash(pc.tobytes()))
        return "unknown"



def create_point_qa_model(model_name: str, checkpoint_path: str = None, **kwargs) -> PointQAModel:
    return PointQAModel(model_name=model_name, checkpoint_path=checkpoint_path, **kwargs)


def list_supported_models() -> List[str]:
    return list(PointQAModel.SUPPORTED_MODELS.keys())