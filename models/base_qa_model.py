import os
import re
import json
import torch
import diskcache
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any, Union
from collections import OrderedDict, Counter


# ==================== Utility Functions ====================

def make_options(choices: List[str], format: str = 'letter'):
    assert format in ['numeric', 'letter']
    if format == 'numeric':
        prefix1 = [str(i + 1) for i in range(len(choices))]
    else:
        prefix1 = [chr(ord("A") + i) for i in range(len(choices))]
    prefix2 = [f"({p})" for p in prefix1]
    return prefix1, prefix2, [f'{p} {c}' for p, c in zip(prefix2, choices)]


def check_contain(answer: str, options: List[str]) -> int:

    contains = [option in answer for option in options]
    if sum(contains) == 1:
        return contains.index(True)
    return -1


# ==================== Answer Extraction ====================

TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)


def extract_answer(txt: str) -> str:
    txt = THINK_RE.sub("", txt)
    m = TAG_RE.search(txt)
    ans = m.group(1) if m else txt
    return " ".join(ans.strip().lower().split())


def extract_think(txt: str) -> str:
    m = THINK_RE.search(txt)
    if not m:
        return ""
    return " ".join(m.group(1).strip().lower().split())


# ==================== Metrics ====================

def compute_f1(pred: str, gold: str) -> float:
    pc, gc = pred.split(), gold.split()
    if not pc or not gc:
        return 0.0
    common = Counter(pc) & Counter(gc)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pc)
    r = num_same / len(gc)
    return 2 * p * r / (p + r)


def compute_em(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


# ==================== Prompt Builder ====================

def build_qa_prompt(question: str, options: List[str] = None) -> str:
    if options is None:
        return question
    options_text = "\n".join(options)
    return f"{question}\n\nOptions:\n{options_text}"


def build_cot_prompt(question: str, options: List[str] = None) -> str:
    base_prompt = build_qa_prompt(question, options)
    return f"{base_prompt}\n\nPlease think step by step and provide your answer in <answer></answer> tags."


# ==================== Base Model Classes ====================

class QAModelInstance(ABC):
    
    @abstractmethod
    def qa(self, data: Dict[str, Any], prompt: str) -> str:

        pass
    
    @abstractmethod
    def batch_qa(self, batch_data: Dict[str, torch.Tensor], prompts: List[str]) -> List[str]:
        pass


class BaseQAModel(ABC):
    
    def __init__(
        self,
        model_name: str,
        prompt_name: str,
        prompt_func: Callable = None,
        choice_format: str = 'letter',
        cache_path: str = None,
    ):
        """
        Args:
            model_name: 模型名称
            prompt_name: prompt 模板名称
            prompt_func: prompt 构建函数
            choice_format: 选项格式 ('letter' 或 'numeric')
            cache_path: 缓存路径，None 表示禁用缓存
        """
        self.model: QAModelInstance = None
        self.model_name = f'{model_name} ({prompt_name})'
        self.prompt_func = prompt_func or build_qa_prompt
        self.format = choice_format
        self.cache_path = cache_path
        
        if self.cache_path is None:
            print("[INFO] Model cache is disabled")
        else:
            print(f"[INFO] Model cache enabled, path: {cache_path}")
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """加载模型（子类实现）"""
        pass
    
    @abstractmethod
    def _data_to_str(self, data: Dict[str, Any]) -> str:
        """
        将数据转换为字符串（用于缓存 key）
        子类需要根据具体的数据类型实现
        """
        pass
    
    @torch.no_grad()
    def _qa_with_cache(self, data: Dict[str, Any], prompt: str) -> str:
        """带缓存的问答"""
        if self.cache_path is None:
            return self.model.qa(data, prompt)
        
        cache_key = f"{self._data_to_str(data)}_{prompt}"
        with diskcache.Cache(self.cache_path, size_limit=10 * (2 ** 30)) as cache:
            if cache_key in cache:
                return cache[cache_key]
            result = self.model.qa(data, prompt)
            cache[cache_key] = result
            return result
    
    @torch.no_grad()
    def qa(self, data: Dict[str, Any], question: str) -> str:
        """
        单次问答
        
        Args:
            data: 输入数据
            question: 问题文本
        
        Returns:
            模型回答
        """
        prompt = self.prompt_func(question)
        return self._qa_with_cache(data, prompt)
    
    @torch.no_grad()
    def multiple_choice_qa(
        self,
        data: Dict[str, Any],
        question: str,
        choices: List[str],
        answer: str = None,
    ) -> Dict[str, Any]:
        """
        多选题问答
        
        Args:
            data: 输入数据
            question: 问题文本
            choices: 选项列表
            answer: 标准答案（可选，用于计算正确率）
        
        Returns:
            包含 free_form_answer, multiple_choice_answer, choices 的字典
        """
        prefix1, prefix2, options = make_options(choices, self.format)
        prompt = self.prompt_func(question, options)
        free_form_answer = self._qa_with_cache(data, prompt).strip()
        
        # 尝试将自由形式的回答匹配到选项
        multiple_choice_answer = self._match_to_choice(
            free_form_answer, choices, options, prefix1, prefix2
        )
        
        result = {
            "free_form_answer": free_form_answer,
            "multiple_choice_answer": multiple_choice_answer,
            "choices": choices.copy(),
        }
        
        if answer is not None:
            result["correct"] = (multiple_choice_answer == answer)
            result["ground_truth"] = answer
        
        return result
    
    def _match_to_choice(
        self,
        answer: str,
        choices: List[str],
        options: List[str],
        prefix1: List[str],
        prefix2: List[str],
    ) -> str:
        """将自由形式的回答匹配到选项"""
        # 直接匹配
        if answer in choices:
            return answer
        if answer in options:
            return choices[options.index(answer)]
        if answer in prefix1:
            return choices[prefix1.index(answer)]
        if answer in prefix2:
            return choices[prefix2.index(answer)]
        
        # 检查包含关系
        for to_check, to_return in [
            (choices, choices),
            (options, choices),
            (prefix1, choices),
            (prefix2, choices),
        ]:
            idx = check_contain(answer, to_check)
            if idx != -1:
                return to_return[idx]
        
        return ""
    
    @torch.no_grad()
    def batch_qa(
        self,
        batch_data: Dict[str, torch.Tensor],
        questions: List[str],
    ) -> List[str]:
        """
        批量问答
        
        Args:
            batch_data: 批量输入数据
            questions: 问题列表
        
        Returns:
            回答列表
        """
        prompts = [self.prompt_func(q) for q in questions]
        return self.model.batch_qa(batch_data, prompts)
    
    def evaluate_qa(
        self,
        predictions: Dict[str, List[str]],
        references: Dict[str, List[str]],
        metrics_module: Any = None,
    ) -> Dict[str, float]:
        """
        评估 QA 结果
        
        Args:
            predictions: {key: [pred_answer]}
            references: {key: [gold_answer]}
            metrics_module: 可选的 metrics 模块（包含 BLEU, CiDEr 等）
        
        Returns:
            评估指标字典
        """
        em_total, f1_total, n_samples = 0, 0, 0
        
        for key in predictions:
            if key not in references:
                continue
            pred = predictions[key][0]
            gold = references[key][0]
            
            em_total += compute_em(pred, gold)
            f1_total += compute_f1(pred, gold)
            n_samples += 1
        
        metrics = OrderedDict(
            EM=round(em_total / n_samples * 100, 2) if n_samples > 0 else 0.0,
            F1=round(f1_total / n_samples * 100, 2) if n_samples > 0 else 0.0,
        )
        
        # 如果提供了 metrics_module，计算额外指标
        if metrics_module is not None:
            try:
                corpus_metrics = self._compute_corpus_metrics(
                    predictions, references, metrics_module
                )
                metrics.update(corpus_metrics)
            except Exception as e:
                print(f"[WARNING] Failed to compute corpus metrics: {e}")
        
        return metrics
    
    def _compute_corpus_metrics(
        self,
        predictions: Dict[str, List[str]],
        references: Dict[str, List[str]],
        metrics_module: Any,
    ) -> Dict[str, float]:
        """计算语料级别的指标（BLEU, CiDEr, ROUGE, METEOR）"""
        bleu = metrics_module.bleu.Bleu(4).compute_score(references, predictions)
        cider = metrics_module.cider.Cider().compute_score(references, predictions)
        rouge = metrics_module.rouge.Rouge().compute_score(references, predictions)
        meteor = metrics_module.meteor.Meteor().compute_score(references, predictions)
        
        return OrderedDict(
            BLEU1=round(bleu[0][0] * 100, 2),
            BLEU4=round(bleu[0][3] * 100, 2),
            CiDEr=round(cider[0] * 100, 2),
            ROUGE_L=round(rouge[0] * 100, 2),
            METEOR=round(meteor[0] * 100, 2),
        )
    
    def save_results(
        self,
        predictions: Dict[str, List[str]],
        references: Dict[str, List[str]],
        output_dir: str,
        prefix: str = "qa",
    ):
        """保存预测结果和标准答案"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"{prefix}_pred.json"), "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, f"{prefix}_gt.json"), "w") as f:
            json.dump(references, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Results saved to {output_dir}")