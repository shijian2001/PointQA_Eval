"""
 Copyright (c) 2026, pjlab.
 All rights reserved.
"""

import re

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("multiple_choice3d")
class MultipleChoice3dTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples, return_before_evaluation=None):
        results = []

        outputs = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        question_ids = samples.get("question_id")
        options_batch = samples.get("options")
        answer_ids = samples.get("answer_id")

        for idx, output in enumerate(outputs):
            options = options_batch[idx] if options_batch is not None else []
            pred_id = self._extract_choice_id(output, options)
            results.append(
                {
                    "question_id": question_ids[idx] if question_ids is not None else idx,
                    "prediction": output,
                    "prediction_id": pred_id,
                    "answer_id": answer_ids[idx] if answer_ids is not None else None,
                }
            )
        return results

    @staticmethod
    def _extract_choice_id(prediction, options):
        pred = prediction.strip()
        letters = [chr(ord("A") + i) for i in range(len(options))]

        for letter in letters:
            if pred.startswith(f"{letter}.") or pred.startswith(f"({letter})") or pred == letter:
                return letter

        match = re.search(r"\b([A-Z])\b", pred)
        if match and match.group(1) in letters:
            return match.group(1)

        for letter, option in zip(letters, options):
            option_text = re.sub(r"^[A-Z]\.?\s*", "", option).strip()
            if option in pred or option_text in pred:
                return letter

        return ""

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        total = 0
        correct = 0
        for item in val_result:
            answer_id = item.get("answer_id")
            pred_id = item.get("prediction_id")
            if answer_id is None:
                continue
            total += 1
            if pred_id == answer_id:
                correct += 1

        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy, "total": total, "correct": correct}
