# -----------------------------------------------------------------------------
# File:     src/lab/eval/runner.py
# Purpose:  Run a strategy over the evaluation questions, in parallel, with a cost cap.
# Author:   Alexandre
# Created:  2026-09-26
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Runner de avaliação: uma estratégia, todas as perguntas, mesmas regras.

Trata toda estratégia igual (só usa a interface :class:`Strategy`), roda as
perguntas em paralelo e devolve os resultados **na ordem das perguntas**,
independentemente de qual thread terminou primeiro.

Falhas têm dois destinos:

- **Erro de uma pergunta** (API fora do ar depois das retentativas, etc.): vira
  um resultado com ``error`` e a rodada continua. Uma pergunta ruim não
  derruba as outras 79.
- **Limite de custo estourado** (``BudgetExceededError``) ou configuração
  inválida (``LLMConfigError``): a rodada inteira aborta, cancelando o que não
  começou. É a regra da SPEC 7.4: o runner aborta acima do teto.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from pydantic import BaseModel

from lab.eval.dataset import Question
from lab.eval.metrics import Scores, score
from lab.llm import BudgetExceededError, LLMConfigError
from lab.strategies.base import Answer, Strategy


class QuestionResult(BaseModel):
    """Resultado de uma pergunta: a resposta e as métricas, ou o erro."""

    question_id: str
    category: str
    answer: Answer | None = None
    scores: Scores | None = None
    error: str | None = None


def _run_one(strategy: Strategy, question: Question) -> QuestionResult:
    try:
        answer = strategy.answer(question.question)
    except (BudgetExceededError, LLMConfigError):
        raise  # aborta a rodada inteira
    except Exception as exc:
        return QuestionResult(
            question_id=question.id,
            category=question.category,
            error=f"{type(exc).__name__}: {exc}",
        )
    return QuestionResult(
        question_id=question.id,
        category=question.category,
        answer=answer,
        scores=score(answer, question),
    )


def run_eval(
    strategy: Strategy,
    questions: Sequence[Question],
    *,
    workers: int = 4,
    on_result: Callable[[QuestionResult], None] | None = None,
) -> list[QuestionResult]:
    """Responder todas as perguntas com ``strategy``.

    :param workers: perguntas em paralelo (``1`` = sequencial).
    :param on_result: chamada a cada resultado pronto, na thread principal,
        na ordem de conclusão (para mostrar progresso).
    :return: um :class:`QuestionResult` por pergunta, na ordem de ``questions``.
    :raises BudgetExceededError: o custo estimado passou do limite do cliente.
    :raises LLMConfigError: papel, provedor ou chave inválidos.
    """
    if workers < 1:
        raise ValueError("workers deve ser >= 1")
    results: dict[str, QuestionResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[QuestionResult], Question] = {
            pool.submit(_run_one, strategy, q): q for q in questions
        }
        try:
            for future in as_completed(futures):
                result = future.result()  # relança Budget/Config
                results[result.question_id] = result
                if on_result is not None:
                    on_result(result)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [results[q.id] for q in questions]


__all__ = ["QuestionResult", "run_eval"]
