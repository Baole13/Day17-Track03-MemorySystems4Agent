from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for item in expected if item.lower() in lowered)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""
    score = recall_points(answer, expected)
    lowered = answer.lower()
    if len(answer.split()) <= 40:
        score += 0.2
    if any(marker in lowered for marker in ("- ", "ví dụ", "ngắn gọn")):
        score += 0.1
    return min(score, 1.0)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    agent_tokens_only = 0
    prompt_tokens_processed = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions = 0
    memory_growth_bytes = 0
    measured_users: set[str] = set()

    for conversation in conversations:
        thread_id = conversation["id"]
        user_id = conversation["user_id"]

        for turn in conversation["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            agent_tokens_only += int(result.get("agent_tokens", 0))

        prompt_tokens_processed += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for index, recall in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread_id = f"{thread_id}-recall-{index}"
            answer = agent.reply(user_id, recall_thread_id, recall["question"])["response"]
            recall_scores.append(recall_points(answer, recall["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, recall["expected_contains"]))
            agent_tokens_only += agent.token_usage(recall_thread_id)
            prompt_tokens_processed += agent.prompt_token_usage(recall_thread_id)

        if hasattr(agent, "memory_file_size") and user_id not in measured_users:
            memory_growth_bytes += agent.memory_file_size(user_id)
            measured_users.add(user_id)

    recall_score = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    response_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_only,
        prompt_tokens_processed=prompt_tokens_processed,
        recall_score=recall_score,
        response_quality=response_quality,
        memory_growth_bytes=memory_growth_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""
    header = (
        "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall "
        "| Response quality | Memory growth (bytes) | Compactions |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    body = [
        (
            f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} "
            f"| {row.recall_score:.2f} | {row.response_quality:.2f} "
            f"| {row.memory_growth_bytes} | {row.compactions} |"
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    standard_data = load_conversations(config.data_dir / "conversations.json")
    stress_data = load_conversations(config.data_dir / "advanced_long_context.json")

    standard_rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), standard_data, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), standard_data, config),
    ]
    stress_rows = [
        run_agent_benchmark("Baseline", BaselineAgent(config=config, force_offline=True), stress_data, config),
        run_agent_benchmark("Advanced", AdvancedAgent(config=config, force_offline=True), stress_data, config),
    ]

    print("## Standard Benchmark")
    print(format_rows(standard_rows))
    print()
    print("## Long-Context Stress Benchmark")
    print(format_rows(stress_rows))


if __name__ == "__main__":
    main()
