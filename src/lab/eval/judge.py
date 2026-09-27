# -----------------------------------------------------------------------------
# File:     src/lab/eval/judge.py
# Purpose:  LLM-as-judge: correctness of an answer against the reference (0/1/2 rubric).
# Author:   Alexandre
# Created:  2026-09-27
# License:  Apache-2.0
# -----------------------------------------------------------------------------

"""Juiz LLM (SPEC 7.3, "Correção").

O juiz compara a resposta de uma estratégia com a ``reference_answer`` da
pergunta e dá uma nota 0/1/2 pela rubrica fixa do prompt versionado
(``prompts/judge_v1.md``). A rubrica é a mesma para todas as estratégias, então
as notas são comparáveis entre runs que usam o mesmo prompt e o mesmo modelo de
juiz (os dois ficam registrados no ``config.yaml`` da run).

O juiz deve ser de família diferente do gerador (SPEC 4.2), para reduzir o viés
de autoavaliação: :func:`check_judge_independence` recusa o mesmo modelo nos
dois papéis. Famílias diferentes com nomes distintos (ex.: um modelo derivado do
outro) não são detectáveis por nome; isso fica registrado na escolha de modelos.

Falha do juiz numa pergunta (resposta fora do schema, erro de rede) não derruba
a run: a nota fica ``None`` e o motivo vai para o resultado. Estouro de custo e
configuração inválida continuam abortando tudo, como no resto da avaliação.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lab.eval.dataset import Question
from lab.llm import BudgetExceededError, LLMClient, LLMConfigError
from lab.llm.costs import Usage
from lab.settings import ModelsConfig
from lab.strategies.base import Answer, load_prompt


class JudgeOutput(BaseModel):
    """Schema JSON que o juiz deve devolver."""

    score: Literal[0, 1, 2]
    rationale: str


class Verdict(BaseModel):
    """Resultado do juiz para uma resposta.

    ``score`` é ``None`` quando o juiz falhou (``error`` diz por quê); ``usage``
    é o custo da chamada ao juiz, contado à parte do custo da estratégia.
    """

    score: Literal[0, 1, 2] | None = None
    rationale: str = ""
    error: str | None = None
    usage: Usage = Usage()

    @property
    def correctness(self) -> float | None:
        """Nota normalizada de 0 a 1 (``score / 2``)."""
        return None if self.score is None else self.score / 2


class JudgeError(ValueError):
    """Configuração do juiz inválida (ex.: mesmo modelo que o gerador)."""


def check_judge_independence(models: ModelsConfig, *, judge_role: str, generator_role: str) -> None:
    """Recusar juiz e gerador com o mesmo modelo (SPEC 4.2).

    :raises JudgeError: os dois papéis apontam para o mesmo modelo.
    :raises LLMConfigError: algum dos papéis não existe em ``models.yaml``.
    """
    for role in (judge_role, generator_role):
        if role not in models.roles:
            raise LLMConfigError(f"unknown role: {role!r}")
    judge_model = models.roles[judge_role].model
    if judge_model == models.roles[generator_role].model:
        raise JudgeError(
            f"o juiz ({judge_role}) usa o mesmo modelo do gerador ({generator_role}): "
            f"{judge_model}. A SPEC exige famílias diferentes."
        )


def _user_message(question: Question, answer: Answer) -> str:
    kind = (
        "sem resposta no corpus (o correto é se abster)"
        if question.category == "sem_resposta"
        else question.category
    )
    return (
        f"Categoria da pergunta: {kind}\n\n"
        f"Pergunta:\n{question.question}\n\n"
        f"Resposta de referência:\n{question.reference_answer}\n\n"
        f"Resposta a avaliar:\n{answer.text.strip() or '(vazia)'}"
    )


class Judge:
    """Avalia respostas com um papel de modelo e a rubrica de ``prompts/<prompt>.md``.

    :param client: cliente LLM (o mesmo da run: cache e limite de custo compartilhados).
    :param role: papel do juiz em ``models.yaml``.
    :param prompt: nome do prompt da rubrica (a versão faz parte do nome).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        role: str = "judge",
        prompt: str = "judge_v1",
        prompts_dir: Path = Path("prompts"),
        max_tokens: int = 2000,
    ) -> None:
        self._client = client
        self.role = role
        self.prompt = prompt
        self._system = load_prompt(prompt, prompts_dir)
        self._max_tokens = max_tokens

    def evaluate(self, question: Question, answer: Answer) -> Verdict:
        """Nota da resposta. Nunca levanta, exceto por custo ou configuração."""
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": _user_message(question, answer)},
        ]
        try:
            parsed = self._client.chat_parsed(
                self.role, messages, JudgeOutput, temperature=0, max_tokens=self._max_tokens
            )
        except (BudgetExceededError, LLMConfigError):
            raise
        except Exception as exc:
            return Verdict(error=f"{type(exc).__name__}: {exc}")
        usage = parsed.result.usage
        if parsed.value is None:
            return Verdict(error=f"juiz fora do schema: {parsed.error}", usage=usage)
        return Verdict(
            score=parsed.value.score, rationale=parsed.value.rationale.strip(), usage=usage
        )


__all__ = [
    "Judge",
    "JudgeError",
    "JudgeOutput",
    "Verdict",
    "check_judge_independence",
]
