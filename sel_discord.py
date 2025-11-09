#!/usr/bin/env python3
"""
Discord interface for SEL.

The bot watches full channel context, keeps timestamped memory, interprets
images through an optional vision model, and can proactively rejoin quiet
conversations based on relationship cadence instead of randomness.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import defaultdict
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import discord

try:  # pragma: no cover
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:  # pragma: no cover
    load_dotenv()

from sel import build_brain, build_vision_model  # noqa: E402


DEFAULT_RESPONSIVENESS = float(os.environ.get("SEL_DISCORD_RESPONSIVENESS", "0.25"))
MONITOR_INTERVAL = int(os.environ.get("SEL_MONITOR_INTERVAL", "60"))
IDLE_THRESHOLD = int(os.environ.get("SEL_IDLE_THRESHOLD", "900"))
EMOJI_REGEX = re.compile(
    r"[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF\U00002700-\U000027BF\U00002600-\U000026FF]"
)
TOKEN_REGEX = re.compile(r"[A-Za-z0-9']+")
AMBIENT_EMOJIS = ["✨", "👀", "😄", "❤️", "🫶", "😊"]


class SELDiscordClient(discord.Client):
    def __init__(self, *, responsiveness: float = DEFAULT_RESPONSIVENESS) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.default_responsiveness = max(0.05, min(0.95, responsiveness))
        self.monitor_interval = max(30, MONITOR_INTERVAL)
        self.idle_threshold = max(300, IDLE_THRESHOLD)
        self.brain = build_brain()
        self.vision = build_vision_model()
        self.channel_styles = self.brain.channel_styles
        self.histories: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        self.channel_state: Dict[int, Dict[str, object]] = {}
        self.monitor_task: Optional[asyncio.Task] = None

    async def setup_hook(self) -> None:
        self.monitor_task = asyncio.create_task(self._monitor_channels())

    async def close(self) -> None:
        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
        await super().close()

    async def on_ready(self) -> None:
        assert self.user is not None
        vision_state = "enabled" if self.vision and getattr(self.vision, "available", False) else "disabled"
        print(f"SEL Discord client ready as {self.user} (vision={vision_state})")

    def _effective_responsiveness(self, user_id: str) -> float:
        stats = self.brain.relationships.users.get(user_id)
        if not stats:
            return self.default_responsiveness
        base = stats.preferred_cadence
        if stats.jealousy > 0.3:
            base *= 0.9
        if stats.trust > 0.75:
            base = min(0.95, base + 0.15)
        return max(0.05, min(0.98, base))

    async def _describe_attachments(self, message: discord.Message) -> List[str]:
        if not message.attachments or not self.vision or not getattr(self.vision, "available", False):
            return []
        descriptions: List[str] = []
        for attachment in message.attachments:
            content_type = attachment.content_type or ""
            if not content_type.startswith("image"):
                continue
            try:
                blob = await attachment.read()
            except Exception:
                continue
            desc = await asyncio.to_thread(self.vision.describe_image, blob)
            if desc:
                descriptions.append(desc)
        return descriptions

    @staticmethod
    def _mentions_memory(text: str) -> bool:
        lowered = text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "remember when",
                "as you said",
                "like you mentioned",
                "last time you",
                "the meme you liked",
            )
        )

    @staticmethod
    def _extract_emoji(text: str) -> tuple[int, Optional[str]]:
        matches = EMOJI_REGEX.findall(text)
        return len(matches), (matches[-1] if matches else None)

    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        tokens = []
        for match in TOKEN_REGEX.findall(text.lower()):
            if len(match) < 4:
                continue
            if match.startswith("http"):
                continue
            tokens.append(match)
        return tokens

    @staticmethod
    def _deterministic_reaction(message_id: int) -> str:
        idx = abs(hash(message_id)) % len(AMBIENT_EMOJIS)
        return AMBIENT_EMOJIS[idx]

    @staticmethod
    def _quick_priority_hint(text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ("urgent", "emergency", "panic", "help me")):
            return "high"
        if "?" in text or any(word in lowered for word in ("please", "can you")):
            return "medium"
        return "normal"

    def _should_reply(self, context_payload: Dict[str, object]) -> bool:
        return self.brain.should_participate(context_payload)

    async def _maybe_ambient_react(self, message: discord.Message, context_payload: Dict[str, object]) -> None:
        snapshot = self.brain.snapshot_hormones()
        hormones = snapshot.get("hormone_map", {})
        if not hormones:
            return
        dopamine = hormones.get("dopamine", 0.5)
        oxytocin = hormones.get("oxytocin", 0.5)
        cortisol = hormones.get("cortisol", 0.5)
        if dopamine + oxytocin < 1.1 or cortisol > 0.5:
            return
        emoji = self._deterministic_reaction(message.id)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji)
        with contextlib.suppress(Exception):
            await message.channel.trigger_typing()

    async def _maybe_bored_ping(self, now: datetime) -> None:
        snapshot = self.brain.snapshot_hormones()
        hormones = snapshot.get("hormone_map", {})
        if not hormones:
            return
        dopamine = hormones.get("dopamine", 0.5)
        oxytocin = hormones.get("oxytocin", 0.5)
        cortisol = hormones.get("cortisol", 0.5)
        melatonin = hormones.get("melatonin", 0.5)
        if not (dopamine < 0.45 and oxytocin < 0.55 and cortisol < 0.55 and melatonin < 0.62):
            return
        for channel_id, state in self.channel_state.items():
            last_sel: Optional[datetime] = state.get("last_sel")  # type: ignore[assignment]
            last_user: Optional[datetime] = state.get("last_user")  # type: ignore[assignment]
            if not last_user:
                continue
            if last_sel and (now - last_sel).total_seconds() < 900:
                continue
            if (now - last_user).total_seconds() < 600:
                continue
            channel = self.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(channel_id)
                except Exception:
                    continue
            text = "Brain’s idling over here—anyone up for a quick tangent?"
            try:
                await channel.send(text)
            except Exception:
                continue
            state["last_sel"] = now
            break

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        assert self.user is not None

        channel_id = message.channel.id
        author_id = str(message.author.id)
        now = message.created_at
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        content = message.content.strip()
        descriptions = await self._describe_attachments(message)
        if descriptions:
            image_lines = "\n".join(f"[Image] {desc}" for desc in descriptions)
            content = f"{content}\n{image_lines}" if content else image_lines

        state = self.channel_state.setdefault(channel_id, {})
        state["last_user"] = now
        state["last_user_id"] = author_id
        state["last_user_name"] = message.author.display_name

        entry = {"role": "user", "text": content, "timestamp": now.isoformat()}
        self.histories[channel_id].append(entry)

        priority_hint = self._quick_priority_hint(content)
        recent_msgs = state.setdefault("recent_msgs", [])
        recent_msgs.append(now.timestamp())
        cutoff = now.timestamp() - 900
        state["recent_msgs"] = [ts for ts in recent_msgs if ts > cutoff]
        recent_users = state.setdefault("recent_users", [])
        recent_users.append((author_id, now.timestamp()))
        state["recent_users"] = [(uid, ts) for uid, ts in recent_users if now.timestamp() - ts < 900]

        style_metrics = state.setdefault(
            "style",
            {
                "avg_len": max(len(content), 1),
                "emoji_rate": 0.0,
                "sample_emoji": "",
                "last_update": now.timestamp(),
                "signature_terms": [],
            },
        )
        text_len = max(len(content), 1)
        style_metrics["avg_len"] = style_metrics["avg_len"] * 0.8 + text_len * 0.2
        emoji_count, sample_emoji = self._extract_emoji(content)
        rate = emoji_count / text_len
        style_metrics["emoji_rate"] = style_metrics["emoji_rate"] * 0.8 + rate * 0.2
        if sample_emoji:
            style_metrics["sample_emoji"] = sample_emoji
        style_metrics["last_update"] = now.timestamp()
        token_counts = state.setdefault("token_counts", {})
        for token in self._extract_tokens(content):
            token_counts[token] = token_counts.get(token, 0) + 1
        top_terms = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)[:3]
        style_metrics["signature_terms"] = [term for term, _ in top_terms]
        self.channel_styles.update(str(channel_id), style_metrics)

        pinged = isinstance(message.channel, discord.DMChannel)
        if not pinged and self.user:
            pinged = self.user in message.mentions
        if not pinged and message.reference and isinstance(message.reference.resolved, discord.Message):
            ref = message.reference.resolved
            if ref.author == self.user:
                pinged = True

        lowered = content.lower()
        direct_reference = pinged or "sel" in lowered

        context_payload = {
            "channel_density": min(1.0, len(state["recent_msgs"]) / 40.0),
            "active_users": len({uid for uid, _ in state["recent_users"]}),
            "channel_name": getattr(message.channel, "name", "DM"),
            "pinged": pinged,
            "priority_hint": priority_hint,
            "mentions_memory": self._mentions_memory(content),
            "direct_reference": direct_reference,
            "channel_id": str(channel_id),
            "channel_profile": dict(style_metrics),
        }

        should_reply = self._should_reply(context_payload)
        if not should_reply and priority_hint != "high":
            await self._maybe_ambient_react(message, context_payload)
            return

        result = await asyncio.to_thread(
            self.brain.respond,
            content,
            self.histories[channel_id],
            author_id,
            now.timestamp(),
            context_payload,
            descriptions,
        )

        priority_label = result.get("priority", "normal")
        if priority_label != "high" and not should_reply and priority_hint != "high":
            return

        if result.get("silent"):
            self.histories[channel_id].append({"role": "assistant", "text": "", "timestamp": now.isoformat()})
            return

        answer = (result.get("answer") or "").strip()
        if not answer:
            return

        await message.channel.send(answer)
        self.histories[channel_id].append(
            {"role": "assistant", "text": result.get("raw_answer", answer), "timestamp": datetime.now(timezone.utc).isoformat()}
        )
        state["last_sel"] = datetime.now(timezone.utc)

    async def _monitor_channels(self) -> None:
        try:
            while not self.is_closed():
                await asyncio.sleep(self.monitor_interval)
                now = datetime.now(timezone.utc)
                for channel_id, state in list(self.channel_state.items()):
                    last_user: Optional[datetime] = state.get("last_user")  # type: ignore[assignment]
                    last_user_id = state.get("last_user_id")
                    if not last_user or not last_user_id:
                        continue
                    idle_seconds = (now - last_user).total_seconds()
                    threshold = self.idle_threshold
                    stats = self.brain.relationships.users.get(last_user_id)
                    if stats:
                        threshold = max(300, self.idle_threshold * (1.0 - stats.preferred_cadence * 0.4))
                    if idle_seconds < threshold:
                        continue
                    last_sel: Optional[datetime] = state.get("last_sel")  # type: ignore[assignment]
                    if last_sel and (now - last_sel).total_seconds() < threshold / 2:
                        continue
                    channel = self.get_channel(channel_id)
                    if channel is None:
                        try:
                            channel = await self.fetch_channel(channel_id)
                        except Exception:
                            continue

                    minutes = max(1, int(idle_seconds // 60))
                    prompt = (
                        f"It's been about {minutes} minutes since {state.get('last_user_name', 'a friend')} "
                        "last spoke here. Offer a gentle, human check-in that fits the current emotional rhythm."
                    )
                    history_copy = list(self.histories[channel_id])
                    result = await asyncio.to_thread(
                        self.brain.respond,
                        prompt,
                        history_copy,
                        last_user_id,
                        now.timestamp(),
                        {
                            "channel_density": 0.1,
                            "active_users": 1,
                            "channel_name": getattr(channel, "name", "DM"),
                            "pinged": False,
                            "priority_hint": "medium",
                            "mentions_memory": False,
                            "direct_reference": False,
                            "channel_id": str(channel_id),
                            "channel_profile": self.channel_styles.get(str(channel_id)),
                            "message_id": f"checkin-{channel_id}-{int(now.timestamp())}",
                            "message_ts": now.timestamp(),
                        },
                    )
                    if result.get("silent"):
                        continue
                    answer = (result.get("answer") or "").strip()
                    if not answer:
                        continue
                    try:
                        await channel.send(answer)
                    except Exception:
                        continue
                    self.histories[channel_id].append(
                        {"role": "assistant", "text": result.get("raw_answer", answer), "timestamp": now.isoformat()}
                    )
                    state["last_sel"] = now
                await self._maybe_bored_ping(now)
        except asyncio.CancelledError:  # pragma: no cover
            pass


async def _amain() -> None:
    token = os.environ.get("SEL_DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("Set SEL_DISCORD_TOKEN (or DISCORD_BOT_TOKEN) in your environment.")
    responsiveness = float(os.environ.get("SEL_DISCORD_RESPONSIVENESS", DEFAULT_RESPONSIVENESS))
    client = SELDiscordClient(responsiveness=responsiveness)
    async with client:
        await client.start(token)


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
