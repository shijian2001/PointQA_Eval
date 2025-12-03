import os
import sys
import time
import torch
from typing import Dict, List, Any, Optional
from collections import OrderedDict

from .base_qa_model import (
    BaseQAModel,
    QAModelInstance,
    extract_answer,
    extract_think,
    compute_em,
    compute_f1,
    build_qa_prompt,
)

DEPENDENCIES_PATH = os.path.join(os.path.dirname(__file__), 'dependencies', '3d-r1')
if DEPENDENCIES_PATH not in sys.path:
    sys.path.insert(0, DEPENDENCIES_PATH)


# ==================== 3D-R1 Model Instance ====================

class ThreeDR1Instance(QAModelInstance):
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_gen_len: int = 32,
        use_beam_search: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_len = max_gen_len
        self.use_beam_search = use_beam_search
    
    def qa(self, data: Dict[str, Any], prompt: str) -> str:
        # 将单个样本转换为 batch 格式
        batch_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                      for k, v in data.items()}
        results = self.batch_qa(batch_data, [prompt])
        return results[0] if results else ""
    
    def batch_qa(
        self,
        batch_data: Dict[str, torch.Tensor],
        prompts: List[str],
    ) -> List[str]:
        self.model.eval()
        
        model_inp = self._prepare_model_input(batch_data)
        
        with torch.no_grad():
            outputs = self.model(model_inp, is_eval=True, task_name="qa")
        
        output_ids = outputs.get("output_ids", outputs.get("output", None))
        if output_ids is None:
            return [""] * len(prompts)
        
        decoded = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        
        return decoded
    
    def _prepare_model_input(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_inp = {}
        
        key_mapping = {
            'point_clouds': 'point_clouds',
            'pcl_color': 'point_clouds_color',
            'point_cloud_dims_min': 'point_cloud_dims_min',
            'point_cloud_dims_max': 'point_cloud_dims_max',
            'qformer_input_ids': 'qformer_input_ids',
            'qformer_attention_mask': 'qformer_attention_mask',
            'instruction': 'instruction',
            'instruction_mask': 'instruction_mask',
        }
        
        for src_key, dst_key in key_mapping.items():
            if src_key in batch_data:
                val = batch_data[src_key]
                if isinstance(val, torch.Tensor):
                    model_inp[dst_key] = val.to(self.device, non_blocking=True)
                else:
                    model_inp[dst_key] = val
        
        return model_inp


# ==================== 3D-R1 QA Model ====================

class PointQAModel(BaseQAModel):
    SUPPORTED_MODELS = {
        "3dr1": {
            "model_class": "CaptionNet",
            "vocab": "qwen/Qwen2.5-7B",
            "qformer_vocab": "google-bert/bert-base-uncased",
        },
        # 未来可以添加更多模型
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
        super().__init__(
            model_name=model_name,
            prompt_name=prompt_name,
            prompt_func=prompt_func or build_qa_prompt,
            choice_format=choice_format,
            cache_path=cache_path,
        )
        
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs
        self.tokenizer = None
        
        self._load_model(**model_kwargs)
    
    def _load_model(self, **kwargs):
        print(f"[INFO] Loading model: {self.model_name}")
        print(f"[INFO] Checkpoint: {self.checkpoint_path}")
        print(f"[INFO] Device: {self.device}")
        
        try:
            from models.model_general import CaptionNet
            from dataset.scannet_base_dataset import DatasetConfig
        except ImportError as e:
            print(f"[ERROR] Failed to import 3D-R1 modules: {e}")
            print(f"[INFO] Please ensure the 3D-R1 code is in: {DEPENDENCIES_PATH}")
            raise
        
        args = self._build_args(**kwargs)
        
        dataset_config = DatasetConfig()
        
        model = CaptionNet(args, dataset_config, None)
        model = model.to(self.device)
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._load_checkpoint(model)
        else:
            print(f"[WARNING] Checkpoint not found: {self.checkpoint_path}")
        
        model.eval()
        
        self.tokenizer = self._get_tokenizer(args)
        
        self.model = ThreeDR1Instance(
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_gen_len=kwargs.get('max_gen_len', 32),
            use_beam_search=kwargs.get('use_beam_search', False),
        )
        
        print(f"Model loaded successfully")
    
    def _build_args(self, **kwargs) -> Any:
        import argparse
        
        args = argparse.Namespace(
            use_color=kwargs.get('use_color', True),
            use_normal=kwargs.get('use_normal', True),
            no_height=kwargs.get('no_height', False),
            use_multiview=kwargs.get('use_multiview', False),
            
            detector=kwargs.get('detector', 'point_encoder'),
            
            captioner=kwargs.get('captioner', '3dr1'),
            vocab=kwargs.get('vocab', 'qwen/Qwen2.5-7B'),
            qformer_vocab=kwargs.get('qformer_vocab', 'google-bert/bert-base-uncased'),
            
            use_additional_encoders=kwargs.get('use_additional_encoders', True),
            use_depth=kwargs.get('use_depth', True),
            use_image=kwargs.get('use_image', True),
            depth_encoder_dim=kwargs.get('depth_encoder_dim', 256),
            image_encoder_dim=kwargs.get('image_encoder_dim', 256),
            
            enable_dynamic_views=kwargs.get('enable_dynamic_views', True),
            view_selection_weight=kwargs.get('view_selection_weight', 0.1),
            use_pytorch3d_rendering=kwargs.get('use_pytorch3d_rendering', True),
            use_multimodal_model=kwargs.get('use_multimodal_model', True),
            
            max_des_len=kwargs.get('max_des_len', 128),
            max_gen_len=kwargs.get('max_gen_len', 32),
            use_beam_search=kwargs.get('use_beam_search', False),
            
            freeze_detector=kwargs.get('freeze_detector', True),
            freeze_llm=kwargs.get('freeze_llm', False),
            
            checkpoint_dir=kwargs.get('checkpoint_dir', './results'),
        )
        
        return args
    
    def _load_checkpoint(self, model: torch.nn.Module):
        print(f"[INFO] Loading checkpoint from {self.checkpoint_path}")
        
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
    
    def _get_tokenizer(self, args) -> Any:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.vocab,
                trust_remote_code=True,
            )
            return tokenizer
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            return None
    
    def _data_to_str(self, data: Dict[str, Any]) -> str:
        if 'point_clouds' in data:
            pc = data['point_clouds']
            if isinstance(pc, torch.Tensor):
                pc_hash = hash(pc.cpu().numpy().tobytes())
            else:
                pc_hash = hash(str(pc))
        else:
            pc_hash = "no_pc"
        
        key_parts = [f"pc_{pc_hash}"]
        
        for k in ['scene_id', 'scan_idx', 'object_id']:
            if k in data:
                key_parts.append(f"{k}_{data[k]}")
        
        return "_".join(key_parts)
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Any,
        annotations: List[Dict],
        output_dir: str = None,
        metrics_module: Any = None,
        logout: callable = print,
    ) -> Dict[str, float]:
        corpus, cand = {}, {}
        em_total, f1_total, n_samples = 0, 0, 0
        
        logout(f"[INFO] Starting evaluation...")
        logout(f"[INFO] Total batches: {len(dataloader)}")
        
        for bi, batch in enumerate(dataloader):
            tic = time.time()
            
            batch = {k: v.to(self.device, non_blocking=True) 
                     if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model.batch_qa(batch, [])
            
            if hasattr(outputs, 'get'):
                output_ids = outputs.get('output_ids', outputs)
            else:
                output_ids = outputs
            
            decoded = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ) if self.tokenizer else []
            
            batch_indices = batch.get('scan_idx', range(len(decoded)))
            if isinstance(batch_indices, torch.Tensor):
                batch_indices = batch_indices.cpu().numpy()
            
            for i, txt in enumerate(decoded):
                global_idx = batch_indices[i] if i < len(batch_indices) else i
                anno = annotations[global_idx]
                
                key = f"{anno.get('scene_id', 'unknown')}-{global_idx}"
                
                pred_ans = extract_answer(txt)
                gold_cot = anno.get('cot', anno.get('answer', ''))
                gold_ans = extract_think(gold_cot) + extract_answer(gold_cot)
                
                em_total += compute_em(pred_ans, gold_ans)
                f1_total += compute_f1(pred_ans, gold_ans)
                n_samples += 1
                
                cand[key] = [pred_ans]
                corpus[key] = [gold_ans]
            
            elapsed = time.time() - tic
            if bi % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                logout(f"[Eval] Batch [{bi}/{len(dataloader)}] Time: {elapsed:.2f}s Mem: {mem:.1f}MB")
            
            torch.cuda.empty_cache()
        
        metrics = OrderedDict(
            EM=round(em_total / n_samples * 100, 2) if n_samples > 0 else 0.0,
            F1=round(f1_total / n_samples * 100, 2) if n_samples > 0 else 0.0,
        )
        
        if metrics_module is not None:
            corpus_metrics = self._compute_corpus_metrics(cand, corpus, metrics_module)
            metrics.update(corpus_metrics)
        
        logout("\n" + "=" * 50)
        logout("QA Evaluation Results")
        logout("=" * 50)
        for k, v in metrics.items():
            logout(f"{k:<10}: {v:.2f}")
        
        if output_dir:
            self.save_results(cand, corpus, output_dir)
        
        return metrics



def create_point_qa_model(
    model_name: str,
    checkpoint_path: str,
    **kwargs,
) -> PointQAModel:
    if model_name not in PointQAModel.SUPPORTED_MODELS:
        print(f"[WARNING] Unknown model: {model_name}, using default config")
    
    return PointQAModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


def list_supported_models() -> List[str]:
    return list(PointQAModel.SUPPORTED_MODELS.keys())