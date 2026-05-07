#!/usr/bin/env python3
"""Convert dataset annotations into the PointQA_Eval tasks.jsonl schema."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import fire


class BaseTaskConverter(ABC):
    """Shared converter contract for dataset-specific task exporters."""

    category = ""

    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def convert(self) -> list[dict[str, Any]]:
        source = self.load_data()
        rows = [self.normalize_record(index, record) for index, record in enumerate(self.iter_records(source))]
        self.write_rows(rows)
        return rows

    def load_data(self) -> Any:
        with self.input_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @abstractmethod
    def iter_records(self, source: Any) -> list[dict[str, Any]] | Any:
        """Yield dataset-native records to normalize."""

    @abstractmethod
    def build_point(self, record: dict[str, Any]) -> str:
        """Return the target point field."""

    def build_question(self, record: dict[str, Any]) -> str:
        return str(record["question"])

    def build_answer(self, record: dict[str, Any]) -> Any:
        return record["answer"]

    def build_options(self, record: dict[str, Any]) -> list[Any]:
        return []

    def build_answer_id(self, record: dict[str, Any]) -> str:
        return ""

    def normalize_record(self, question_id: int, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": question_id,
            "point": self.build_point(record),
            "category": self.category,
            "question": self.build_question(record),
            "options": self.build_options(record),
            "answer": self.build_answer(record),
            "answer_id": self.build_answer_id(record),
        }

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CLEVR3DRealConverter(BaseTaskConverter):
    category = "clevr3d_real"

    def iter_records(self, source: Any) -> list[dict[str, Any]]:
        if not isinstance(source, dict) or "questions" not in source:
            raise ValueError("CLEVR3D-REAL input must be a JSON object with a 'questions' field")
        questions = source["questions"]
        if not isinstance(questions, list):
            raise ValueError("CLEVR3D-REAL 'questions' field must be a list")
        return questions

    def build_point(self, record: dict[str, Any]) -> str:
        return str(record["scan"])


class ConverterCLI:
    """Thin wrapper that keeps dataset-specific entrypoints easy to extend."""

    def clevr3d_real(self, input_path: str, output_path: str) -> list[dict[str, Any]]:
        converter = CLEVR3DRealConverter(input_path=input_path, output_path=output_path)
        return converter.convert()

if __name__ == "__main__":
    fire.Fire(ConverterCLI)
