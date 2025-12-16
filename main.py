import fire
import torch
import os
import json
from models.point_qa_model import PointQAModel
from models.base_qa_model import make_options

def run_eval(
    model_name: str,
    checkpoint_path: str = None,
    test_ckpt: str = None,
    question: str = None,
    tasks_file: str = None,
    point_cloud_dir: str = None,
    point_cloud: str = None,
    point_cloud_path: str = None,
    choices: str = None,
    answer: str = None,
    prompt_template: str = None,
    output_dir: str = None,
    device: str = 'cuda',
    **kwargs
):
    real_ckpt = checkpoint_path or test_ckpt
    if not real_ckpt:
        raise ValueError("必须提供 checkpoint_path 或 test_ckpt")

    choices_list = None
    if choices is not None:
        try:
            if choices.strip().startswith("["):
                choices_list = json.loads(choices)
            else:
                choices_list = [c.strip() for c in choices.split(",") if c.strip()]
        except Exception as e:
            raise ValueError(f"choices参数解析失败: {choices}") from e

    def default_prompt_func(q, opts=None):
        if opts:
            return f"{q}\n{chr(10).join(opts)}\n\nAnswer with the option's letter from the given choices directly."
        return q

    prompt_func = default_prompt_func
    if prompt_template:
        def prompt_func(q, opts=None):
            if opts:
                return prompt_template.format(question=q, choices=" ".join(opts))
            return prompt_template.format(question=q)

    model = PointQAModel(
        model_name=model_name,
        checkpoint_path=real_ckpt,
        prompt_func=prompt_func,
        device=device,
        **kwargs
    )

    if tasks_file:
        if not point_cloud_dir:
            raise ValueError("批量评测模式下必须提供 --point_cloud_dir")
        with open(tasks_file, 'r') as f:
            tasks = [json.loads(line) for line in f]
        results = []
        correct = 0
        total = 0
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'results.jsonl')
            f_out = open(output_file, 'w')
        else:
            f_out = None
        for task in tasks:
            q = task['question']
            choices_task = task.get('options') or task.get('choices')
            ans = task.get('answer')
            point_file = task.get('point') or task.get('point_cloud')
            pc_path = os.path.join(point_cloud_dir, point_file)
            data = {
                'question_text': q,
                'point_cloud_path': pc_path
            }
            if choices_task:
                res = model.multiple_choice_qa(data, q, choices_task, answer=ans)
            else:
                res = model.qa(data, q)
                res = {'free_form_answer': res}
            task_result = {**task, **res}
            results.append(task_result)
            if 'accuracy' in res:
                total += 1
                correct += res['accuracy']
            if f_out:
                f_out.write(json.dumps(task_result, ensure_ascii=False) + '\n')
                f_out.flush()
        if f_out:
            f_out.close()
        if total > 0:
            print(f"Overall Accuracy: {correct/total:.2%} ({correct}/{total})")
    elif question:
        data = {
            'question_text': question,
        }
        if point_cloud_path:
            data['point_cloud_path'] = point_cloud_path
        elif point_cloud:
            data['point_cloud'] = point_cloud
        else:
            raise ValueError("单个评测模式下必须提供 --point_cloud 或 --point_cloud_path")
        if choices_list:
            result = model.multiple_choice_qa(
                data,
                question,
                choices_list,
                answer=answer
            )
        else:
            result = model.qa(data, question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Usage: 请提供 --question 进行单条测试，或提供 --tasks_file 进行批量测试。")

def main():
    fire.Fire(run_eval)

if __name__ == '__main__':
    main()
