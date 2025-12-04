import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from collections import defaultdict


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
        
        data = {'point_cloud': point_cloud, 'point_cloud_path': pc_path}
        choices = parse_options(task['options'])
        
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


def main():
    parser = argparse.ArgumentParser(description="3D Point Cloud QA Evaluation")
    parser.add_argument('--tasks_file', type=str, required=True)
    parser.add_argument('--point_cloud_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--model_name', type=str, default='3dr1')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--use_color', action='store_true')
    parser.add_argument('--use_normal', action='store_true')
    parser.add_argument('--vocab', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--qformer_vocab', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--detector', type=str, default='point_encoder')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    from models.point_qa_model import create_point_qa_model
    model = create_point_qa_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        cache_path=args.cache_path,
        device=args.device,
        use_color=args.use_color,
        use_normal=args.use_normal,
        vocab=args.vocab,
        qformer_vocab=args.qformer_vocab,
        detector=args.detector,
    )
    
    tasks = load_tasks(args.tasks_file)
    evaluate(model, tasks, args.point_cloud_dir, args.output_dir)


if __name__ == '__main__':
    main()