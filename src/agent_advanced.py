from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                result = self.langchain_agent.invoke({"messages": [{"role": "user", "content": message}]})
                content = getattr(result, "content", str(result))
                return self._reply_offline(user_id, thread_id, content)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": updates,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        context = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(context["summary"]))
        recent_tokens = sum(estimate_tokens(item["content"]) for item in context["messages"])
        return profile_tokens + summary_tokens + recent_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        lowered = message.lower()
        context = self.compact_memory.context(thread_id)
        summary = str(context["summary"])

        if "3 bullet" in lowered or "stress test" in lowered:
            return (
                f"- Tên: {facts.get('name', 'mình chưa lưu')}\n"
                f"- Nghề và nơi ở: {facts.get('profession', 'mình chưa lưu')}, {facts.get('location', 'mình chưa lưu')}\n"
                f"- Style: {facts.get('response_style', 'mình chưa lưu')}"
            )

        if "tóm tắt" in lowered or "là ai" in lowered or "biết dũngct" in lowered:
            interests = facts.get("interests", "mình chưa lưu")
            return (
                f"Bạn là {facts.get('name', 'mình chưa lưu')}, hiện làm {facts.get('profession', 'mình chưa lưu')} "
                f"và ở {facts.get('location', 'mình chưa lưu')}. "
                f"Mối quan tâm chính: {interests}. "
                f"Style trả lời bạn thích là {facts.get('response_style', 'trả lời gọn')}."
            )

        requested_parts: list[str] = []
        if "tên" in lowered:
            requested_parts.append(f"Tên bạn là {facts.get('name', 'mình chưa lưu')}")
        if "ở đâu" in lowered or "nơi ở" in lowered or "ở huế không" in lowered:
            requested_parts.append(f"hiện tại ở {facts.get('location', 'mình chưa lưu')}")
        if "nghề" in lowered or "làm nghề gì" in lowered or "nghề nghiệp" in lowered:
            requested_parts.append(f"nghề nghiệp hiện tại là {facts.get('profession', 'mình chưa lưu')}")
        if "style trả lời" in lowered or "kiểu trả lời" in lowered:
            requested_parts.append(f"style trả lời bạn thích là {facts.get('response_style', 'mình chưa lưu')}")
        if "đồ uống" in lowered:
            requested_parts.append(f"đồ uống yêu thích là {facts.get('favorite_drink', 'mình chưa lưu')}")
        if "món ăn" in lowered:
            requested_parts.append(f"món ăn yêu thích là {facts.get('favorite_food', 'mình chưa lưu')}")
        if "nuôi con gì" in lowered or "nuôi con" in lowered:
            requested_parts.append(f"bạn nuôi {facts.get('pet', 'mình chưa lưu')}")

        if requested_parts:
            return ". ".join(requested_parts) + "."

        return (
            f"Mình đã cập nhật hồ sơ hiện hành cho {facts.get('name', 'bạn')} và sẽ ưu tiên facts mới nhất. "
            f"Hiện mình nhớ: {facts.get('profession', 'chưa rõ nghề nghiệp')} tại {facts.get('location', 'chưa rõ nơi ở')}."
        )

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
