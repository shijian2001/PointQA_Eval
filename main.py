import fire
import torch
from models.point_qa_model import PointQAModel
from models.base_qa_model import make_options
import json


def run_eval(
    model_name: str,
    checkpoint_path: str,
    question: str,
    point_cloud: str = None,
    point_cloud_path: str = None,
    choices: str = None,
    answer: str = None,
    prompt_template: str = None,
    device: str = 'cuda',
    **kwargs
):
    if choices is not None:
        try:
            if choices.strip().startswith("["):
                choices_list = json.loads(choices)
            else:
                choices_list = [c.strip() for c in choices.split(",") if c.strip()]
        except Exception as e:
            raise ValueError(f"choices参数解析失败: {choices}") from e
    else:
        choices_list = None

    if prompt_template:
        def prompt_func(q, opts=None):
            if opts:
                return prompt_template.format(question=q, choices=" ".join(opts))
            return prompt_template.format(question=q)
    else:
        prompt_func = None

    model = PointQAModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        prompt_func=prompt_func,
        device=device,
        **kwargs
    )

    data = {
        'question_text': question,
    }
    if point_cloud_path:
        data['point_cloud_path'] = point_cloud_path
    elif point_cloud:
        data['point_cloud'] = point_cloud

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


def main():
    fire.Fire(run_eval)

if __name__ == '__main__':
    main()
