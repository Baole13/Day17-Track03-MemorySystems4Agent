from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None and not self.force_offline:
            try:
                result = self.langchain_agent.invoke({"messages": [{"role": "user", "content": message}]})
                content = getattr(result, "content", str(result))
                return self._reply_offline(thread_id, content)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        response = "Mình chỉ nhớ trong thread hiện tại. Bạn cứ hỏi tiếp trong cuộc trò chuyện này nhé."
        lowered = message.lower()

        if any(phrase in lowered for phrase in ("tên mình", "mình tên gì", "mình là ai")):
            name = None
            for item in reversed(session.messages):
                content = item["content"]
                if item["role"] == "user" and "mình tên" in content.lower():
                    name = content.split("mình tên", 1)[-1].replace("là", "", 1).strip(" .!?")
                    break
            response = f"Trong thread này mình nhớ tên bạn là {name}." if name else "Trong thread này mình chưa thấy bạn nói tên."
        elif "đồ uống yêu thích" in lowered:
            response = "Trong thread này mình chỉ nhớ nếu bạn đã nói rõ đồ uống yêu thích ngay trước đó."
        elif "style trả lời" in lowered:
            response = "Trong thread này mình nhớ bạn thích câu trả lời ngắn gọn nếu bạn vừa nhắc trong cuộc chat này."
        elif "ở đâu" in lowered or "nơi ở" in lowered:
            response = "Trong thread này mình chỉ nhắc lại được nơi ở nếu nó đã xuất hiện ở chính thread này."

        session.messages.append({"role": "assistant", "content": response})
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
