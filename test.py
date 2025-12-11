import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from collections import defaultdict
import fire


# Compatibility shim: ensure huggingface_hub.cached_download exists
try:
    from huggingface_hub import cached_download  # noqa: F401
except Exception:
    import huggingface_hub as _hf
    if hasattr(_hf, "hf_hub_download"):
        _hf.cached_download = _hf.hf_hub_download
    else:
        def _missing_cached_download(*args, **kwargs):
            raise ImportError(
                "huggingface_hub.cached_download and hf_hub_download are both unavailable; "
                "install a compatible huggingface_hub version or update sentence-transformers."
            )
        _hf.cached_download = _missing_cached_download


def load_tasks(tasks_file: str) -> List[Dict]:
    tasks = []
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    print(f"[INFO] Loaded {len(tasks)} tasks")
    return tasks


def load_point_cloud(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        for key in ['point_cloud', 'points', 'xyz', 'data']:
            if key in data:
                return data[key]
        return data[list(data.keys())[0]]
    elif ext in ['.ply', '.pcd']:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            return np.concatenate([points, np.asarray(pcd.colors)], axis=1)
        return points
    raise ValueError(f"Unsupported format: {ext}")


def parse_options(options: List[str]) -> List[str]:
    return [opt.split('. ', 1)[1] if '. ' in opt else opt for opt in options]


def extract_answer_letter(response: str) -> str:
    import re
    response = response.strip().upper()
    if response in 'ABCDEF':
        return response
    if response and response[0] in 'ABCDEF':
        return response[0]
    
    patterns = [
        r'answer\s*(?:is|:)\s*([A-F])',
        r'\b([A-F])\b\s*(?:is\s+)?(?:the\s+)?(?:correct|right)',
        r'(?:option|choice)\s*([A-F])',
        r'\(([A-F])\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def evaluate(model, tasks: List[Dict], point_cloud_dir: str, output_dir: str = None) -> Dict[str, Any]:
    results = []
    correct, total = 0, 0
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for task in tqdm(tasks, desc="Evaluating"):
        pc_path = os.path.join(point_cloud_dir, task['point'])
        if not os.path.exists(pc_path):
            print(f"[WARNING] Not found: {pc_path}")
            continue
        
        try:
            point_cloud = load_point_cloud(pc_path)
        except Exception as e:
            print(f"[ERROR] Load failed {pc_path}: {e}")
            continue
        
        choices = parse_options(task['options'])
        data = {
            'point_cloud': point_cloud,
            'point_cloud_path': pc_path,
            'question_text': task['question'],
            'choices': choices,
        }
        
        result = model.multiple_choice_qa(
            data=data, question=task['question'], choices=choices, answer=task['answer']
        )
        
        pred_letter = extract_answer_letter(result['free_form_answer'])
        is_correct = (result['multiple_choice_answer'] == task['answer']) or (pred_letter == task.get('answer_id', ''))
        
        total += 1
        correct += int(is_correct)
        category = task.get('category', 'unknown')
        category_stats[category]['total'] += 1
        category_stats[category]['correct'] += int(is_correct)
        
        results.append({
            'question_id': task['question_id'],
            'category': category,
            'question': task['question'],
            'ground_truth': task['answer'],
            'ground_truth_id': task.get('answer_id', ''),
            'prediction': result['multiple_choice_answer'],
            'prediction_letter': pred_letter,
            'free_form_answer': result['free_form_answer'],
            'correct': is_correct,
        })
    
    accuracy = correct / total * 100 if total > 0 else 0
    category_accuracy = {
        cat: {'accuracy': round(s['correct'] / s['total'] * 100, 2), **s}
        for cat, s in category_stats.items()
    }
    
    summary = {'total': total, 'correct': correct, 'accuracy': round(accuracy, 2), 'category_accuracy': category_accuracy}
    
    print(f"\n{'='*60}\nResults: {correct}/{total} = {accuracy:.2f}%")
    for cat, s in category_accuracy.items():
        print(f"  {cat}: {s['accuracy']:.2f}% ({s['correct']}/{s['total']})")
    print('='*60)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved to {output_dir}")
    
    return {'summary': summary, 'results': results}


def run(
    tasks_file: str,
    point_cloud_dir: str,
    output_dir: str = './eval_results',
    model_name: str = '3dr1',
    checkpoint: str = None,
    test_ckpt: str = None,
    test_only: bool = False,
    dataset: str = 'scannet',
    cache_path: str = None,
    use_color: bool = False,
    use_normal: bool = False,
    vocab: str = 'Qwen/Qwen2.5-7B',
    qformer_vocab: str = 'google-bert/bert-base-uncased',
    detector: str = 'point_encoder',
    captioner: str = '3dr1',
    checkpoint_dir: str = './results',
    use_additional_encoders: bool = False,
    use_depth: bool = False,
    use_image: bool = False,
    depth_encoder_dim: int = 256,
    image_encoder_dim: int = 256,
    enable_dynamic_views: bool = False,
    view_selection_weight: float = 0.1,
    use_pytorch3d_rendering: bool = False,
    use_multimodal_model: bool = False,
    device: str = 'cuda',
    llava_model_base: str = None,
    llava_pointcloud_tower_name: str = None,
    llava_conv_mode: str = 'vicuna_v1',
    llava_temperature: float = 1.0,
    llava_top_p: float = None,
    llava_num_beams: int = 1,
    llava_max_new_tokens: int = 64,
    llava_voxel_size: float = 0.02,
):
    checkpoint_path = test_ckpt or checkpoint
    if checkpoint_path is None:
        raise ValueError("Please provide --test_ckpt (official flag) or --checkpoint")

    from models.point_qa_model import create_point_qa_model

    model = create_point_qa_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        device=device,
        dataset=dataset,
        test_only=test_only,
        use_color=use_color,
        use_normal=use_normal,
        vocab=vocab,
        qformer_vocab=qformer_vocab,
        detector=detector,
        captioner=captioner,
        checkpoint_dir=checkpoint_dir,
        use_additional_encoders=use_additional_encoders,
        use_depth=use_depth,
        use_image=use_image,
        depth_encoder_dim=depth_encoder_dim,
        image_encoder_dim=image_encoder_dim,
        enable_dynamic_views=enable_dynamic_views,
        view_selection_weight=view_selection_weight,
        use_pytorch3d_rendering=use_pytorch3d_rendering,
        use_multimodal_model=use_multimodal_model,
        llava_model_base=llava_model_base,
        llava_pointcloud_tower_name=llava_pointcloud_tower_name,
        llava_conv_mode=llava_conv_mode,
        llava_temperature=llava_temperature,
        llava_top_p=llava_top_p,
        llava_num_beams=llava_num_beams,
        llava_max_new_tokens=llava_max_new_tokens,
        llava_voxel_size=llava_voxel_size,
    )

    tasks = load_tasks(tasks_file)
    evaluate(model, tasks, point_cloud_dir, output_dir)


if __name__ == '__main__':
    fire.Fire(run)