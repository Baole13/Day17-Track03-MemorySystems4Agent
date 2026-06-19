from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        safe_user_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_id).strip("._") or "user"
        return self.root_dir / safe_user_id / "User.md"

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            key, value = stripped[2:].split(":", 1)
            facts[key.strip()] = value.strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        facts[key] = value
        lines = ["# User Profile", ""]
        for fact_key in sorted(facts):
            lines.append(f"- {fact_key}: {facts[fact_key]}")
        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    lowered = message.lower()
    declarative_markers = (
        "mình tên là",
        "tên mình là",
        "mình ở ",
        "hiện tại mình đang ở",
        "mình đang làm",
        "giờ chuyển sang",
        "mình muốn bạn",
        "hãy trả lời",
        "mình thích",
        "mình nuôi",
        "đồ uống yêu thích",
        "món ăn yêu thích",
        "mình vẫn uống",
        "mình đính chính",
    )
    if "?" in message and not any(marker in lowered for marker in declarative_markers):
        return {}

    noise_markers = (
        "chỉ là câu đùa",
        "chỉ là câu nói đùa",
        "chỉ là ví dụ",
        "không phải nơi ở hiện tại",
        "không còn",
    )
    updates: dict[str, str] = {}
    invalid_values = {
        "gì",
        "đâu",
        "là gì",
        "như thế nào",
        "ra sao",
        "bao nhiêu",
    }

    def pick(patterns: list[tuple[str, str]], key: str) -> None:
        for pattern, group_name in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                candidate = match.group(group_name).strip(" .,!?:;")
                candidate_lower = candidate.lower()
                if (
                    candidate
                    and candidate_lower not in invalid_values
                    and not any(token in candidate_lower for token in ("gì", "đâu", "nào"))
                ):
                    updates[key] = candidate
                    return

    pick(
        [
            (r"mình tên là (?P<value>[^.!,;\n]+)", "value"),
            (r"tên mình(?: là)? (?P<value>[^.!,;\n]+)", "value"),
        ],
        "name",
    )
    pick(
        [
            (r"hiện(?: tại)? (?:mình )?(?:đang )?ở (?P<value>[^.!,;\n]+)", "value"),
            (r"mình (?:đang )?ở (?P<value>[^.!,;\n]+)", "value"),
            (r"giờ mình (?:đang )?ở (?P<value>[^.!,;\n]+)", "value"),
        ],
        "location",
    )
    pick(
        [
            (r"giờ chuyển sang (?P<value>[^.!,;\n]+)", "value"),
            (r"giờ mình đang làm (?P<value>[^.!,;\n]+)", "value"),
            (r"mình đang làm (?P<value>[^.!,;\n]+)", "value"),
            (r"nghề nghiệp hiện tại(?: vẫn)? là (?P<value>[^.!,;\n]+)", "value"),
            (r"công việc mới(?: vẫn)? (?:là|liên quan tới)? (?P<value>[^.!,;\n]+)", "value"),
        ],
        "profession",
    )
    pick(
        [
            (r"đồ uống yêu thích(?: là)? (?P<value>[^.!,;\n]+)", "value"),
            (r"mình vẫn uống (?P<value>[^.!,;\n]+)", "value"),
        ],
        "favorite_drink",
    )
    pick(
        [
            (r"món ăn yêu thích(?: là)? (?P<value>[^.!,;\n]+)", "value"),
            (r"buổi trưa mình ăn (?P<value>[^.!,;\n]+)", "value"),
        ],
        "favorite_food",
    )
    pick(
        [
            (r"mình muốn bạn trả lời (?P<value>[^.!\n]+)", "value"),
            (r"hãy trả lời (?P<value>[^.!\n]+)", "value"),
            (r"style trả lời(?: mình thích)?(?: là)? (?P<value>[^.!\n]+)", "value"),
            (r"mình vẫn muốn style trả lời (?P<value>[^.!\n]+)", "value"),
        ],
        "response_style",
    )
    pick(
        [
            (r"mình nuôi (?:một|1)? ?(?P<value>[^.!,;\n]+)", "value"),
        ],
        "pet",
    )

    interests = set()
    for phrase in (
        "python",
        "ai ứng dụng",
        "ai agent",
        "benchmark memory",
        "mlops",
        "rag",
        "evaluation",
        "async python",
        "memory architecture",
        "memory compaction",
    ):
        if phrase in lowered:
            interests.add(phrase)
    if interests:
        updates["interests"] = ", ".join(sorted(interests))

    if any(marker in lowered for marker in noise_markers):
        if "không phải nơi ở hiện tại" in lowered:
            updates.pop("location", None)
        if "không còn" in lowered and "giờ chuyển sang" not in lowered:
            updates.pop("profession", None)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""

    selected = messages[-max_items:]
    summary_lines: list[str] = []
    for item in selected:
        role = item.get("role", "unknown")
        content = " ".join(item.get("content", "").split())
        if not content:
            continue
        snippet = content[:80].rstrip()
        if len(content) > 80:
            snippet += "..."
        summary_lines.append(f"- {role}: {snippet}")
    return "\n".join(summary_lines[:max_items])


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})

        total_tokens = estimate_tokens(str(context["summary"])) + sum(
            estimate_tokens(message["content"]) for message in messages
        )
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older_messages = messages[:-self.keep_messages]
        kept_messages = messages[-self.keep_messages :]
        prior_summary = str(context["summary"]).strip()
        summary_inputs: list[dict[str, str]] = []
        if prior_summary:
            summary_inputs.append({"role": "summary", "content": prior_summary})
        summary_inputs.extend(older_messages)
        context["summary"] = summarize_messages(summary_inputs, max_items=max(3, self.keep_messages))
        context["messages"] = kept_messages
        context["compactions"] = int(context["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        return int(self.context(thread_id)["compactions"])
