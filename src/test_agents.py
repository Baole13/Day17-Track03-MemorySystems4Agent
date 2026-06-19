from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import uuid

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@pytest.fixture
def tmp_path() -> Path:
    root = Path(__file__).resolve().parent.parent / ".test_tmp" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(tmp_path)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    (config.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        compact_threshold_tokens=80,
        compact_keep_messages=4,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    result = agent.reply("alice", "thread-1", "Mình tên là Alice. Mình ở Huế và đang làm MLOps engineer.")
    assert "response" in result

    content = agent.profile_store.read_text("alice")
    assert "Alice" in content
    assert "Huế" in content

    changed = agent.profile_store.edit_text("alice", "Huế", "Đà Nẵng")
    assert changed is True
    assert "Đà Nẵng" in agent.profile_store.read_text("alice")


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    for index in range(8):
        agent.reply("alice", "thread-compact", f"Turn {index} Mình đang thêm nhiều ngữ cảnh để compact memory sớm hoạt động.")

    assert agent.compaction_count("thread-compact") > 0
    context = agent.compact_memory.context("thread-compact")
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    message = "Mình tên là DũngCT. Mình ở Huế và đang làm MLOps engineer. Mình muốn bạn trả lời ngắn gọn."
    baseline.reply("dungct", "seed-thread", message)
    advanced.reply("dungct", "seed-thread", message)

    baseline_answer = baseline.reply("dungct", "fresh-thread", "Mình tên gì và hiện tại mình ở đâu?")["response"]
    advanced_answer = advanced.reply("dungct", "fresh-thread", "Mình tên gì và hiện tại mình ở đâu?")["response"]

    assert "chưa" in baseline_answer.lower() or "thread này" in baseline_answer.lower()
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    for index in range(12):
        turn = (
            f"Lượt {index}: Mình đang kéo dài hội thoại để benchmark prompt load. "
            "Mình thích Python, AI ứng dụng và muốn câu trả lời ngắn gọn có ví dụ thực tế."
        )
        baseline.reply("dungct", "long-thread", turn)
        advanced.reply("dungct", "long-thread", turn)

    assert advanced.compaction_count("long-thread") > 0
    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")


def test_profile_correction_and_noise_handling(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)

    advanced.reply("dungct", "thread-a", "Mình ở Huế và đang làm backend engineer.")
    advanced.reply(
        "dungct",
        "thread-a",
        "Mình đính chính: giờ chuyển sang MLOps engineer và hiện tại mình đang ở Đà Nẵng.",
    )
    advanced.reply(
        "dungct",
        "thread-a",
        "Hà Nội chỉ là nơi mình họp hai ngày chứ không phải nơi ở hiện tại, còn product manager chỉ là câu đùa.",
    )

    answer = advanced.reply("dungct", "thread-b", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in answer
    assert "Đà Nẵng" in answer
    assert "Hà Nội" not in advanced.profile_store.read_text("dungct")
