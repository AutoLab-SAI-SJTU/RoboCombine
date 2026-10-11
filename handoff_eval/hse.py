from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def parse_clear_cache(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid clear_cache value: {value!r}")


HSEVariant = str
FULL_INST = "full_inst"
ORACLE_HSE = "oracle_hse"
ORDERED_HSE = "ordered_hse"


@dataclass(frozen=True)
class InstructionSwitchConfig:
    clear_action_cache: bool = True

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> "InstructionSwitchConfig":
        return cls(
            clear_action_cache=parse_clear_cache(args.get("clear_cache", True)),
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "switch_cache_reset": bool(self.clear_action_cache),
        }


def resolve_hse_variant(args: dict[str, Any]) -> HSEVariant:
    variant = str(args.get("hse_variant", FULL_INST) or FULL_INST).strip().lower()
    if variant not in {FULL_INST, ORACLE_HSE, ORDERED_HSE}:
        raise ValueError(f"Unsupported HSE variant: {variant}")
    return variant


def apply_instruction_switch(
    model: Any,
    *,
    config: InstructionSwitchConfig,
) -> dict[str, Any]:
    if hasattr(model, "prepare_instruction_switch"):
        switch_info = model.prepare_instruction_switch(
            clear_action_cache=config.clear_action_cache,
        )
    else:
        raise TypeError("Policy must implement prepare_instruction_switch")
    switch_info = dict(switch_info or {})
    switch_info.setdefault("cache_cleared", bool(config.clear_action_cache))
    return switch_info


@dataclass
class HSEEpisodeState:
    current_stage_idx: int
    stage_count: int
    switch_events: list[dict[str, Any]] = field(default_factory=list)
    predictor_switch_steps: list[int | None] = field(default_factory=list)
    pending_switch: dict[str, Any] | None = None

    @classmethod
    def create(cls, stage_count: int) -> "HSEEpisodeState":
        return cls(
            current_stage_idx=0,
            stage_count=max(1, int(stage_count)),
            predictor_switch_steps=[None] * max(1, int(stage_count)),
        )


class HSEController:
    def __init__(
        self,
        *,
        variant: HSEVariant,
        switch_config: InstructionSwitchConfig,
        stage_instructions: list[str],
        real_switch_session: Any | None = None,
    ) -> None:
        self.variant = resolve_hse_variant({"hse_variant": variant})
        self.switch_config = switch_config
        self.stage_instructions = list(stage_instructions)
        self.real_switch_session = real_switch_session

    @property
    def uses_stage_instructions(self) -> bool:
        return self.variant in {ORACLE_HSE, ORDERED_HSE}

    def create_episode_state(self) -> HSEEpisodeState:
        stage_count = max(1, len(self.stage_instructions))
        return HSEEpisodeState.create(stage_count=stage_count)

    def initial_instruction(self, *, full_instruction: str) -> str:
        if self.uses_stage_instructions:
            return self.stage_instructions[0]
        return str(full_instruction)

    def begin_episode(
        self,
        *,
        env: Any,
        state: HSEEpisodeState,
        initial_instruction: str,
        step_count: int,
    ) -> dict[str, Any] | None:
        env.set_instruction(instruction=initial_instruction)
        if self.variant != ORDERED_HSE or self.real_switch_session is None:
            return None
        return self.real_switch_session.begin_stage(
            instruction=initial_instruction,
            stage_index=state.current_stage_idx,
            episode_step=step_count,
            reset_reason="episode_start",
            can_switch=state.current_stage_idx < state.stage_count - 1,
        )

    def _switch_to_next_stage(
        self,
        *,
        env: Any,
        model: Any,
        state: HSEEpisodeState,
        step_count: int,
        source: str,
        extra_event: dict[str, Any] | None = None,
        predictor_reset_reason: str | None = None,
    ) -> dict[str, Any] | None:
        if state.current_stage_idx >= state.stage_count - 1:
            return None
        if not self.switch_config.clear_action_cache and len(getattr(model, "action_cache", ())):
            if state.pending_switch is None:
                state.pending_switch = dict(
                    source=source, extra_event={"cue_step": int(step_count), **(extra_event or {})},
                    predictor_reset_reason=predictor_reset_reason,
                )
            return None
        from_stage_idx = state.current_stage_idx
        to_stage_idx = from_stage_idx + 1
        new_instruction = self.stage_instructions[to_stage_idx]
        env.set_instruction(instruction=new_instruction)
        switch_info = apply_instruction_switch(model, config=self.switch_config)
        reset_event = None
        if self.variant == ORDERED_HSE and self.real_switch_session is not None:
            reset_event = self.real_switch_session.begin_stage(
                instruction=new_instruction,
                stage_index=to_stage_idx,
                episode_step=step_count,
                reset_reason=str(predictor_reset_reason or "stage_switch"),
                can_switch=to_stage_idx < state.stage_count - 1,
            )
        event = {
            "source": source,
            "switch_source": source,
            "from_subtask_idx": from_stage_idx,
            "to_subtask_idx": to_stage_idx,
            "completed_subtask_idx": from_stage_idx,
            "completed_subtask_name": self.stage_instructions[from_stage_idx],
            "new_instruction": new_instruction,
            "switch_step": int(step_count),
            **switch_info,
        }
        if extra_event:
            event.update(extra_event)
        if reset_event is not None:
            event["predictor_reset_event"] = reset_event
        state.current_stage_idx = to_stage_idx
        state.switch_events.append(event)
        return event

    def flush_pending_switch(self, *, env, model, state, step_count):
        if state.pending_switch is None or len(getattr(model, "action_cache", ())):
            return None
        pending = state.pending_switch
        state.pending_switch = None
        return self._switch_to_next_stage(
            env=env, model=model, state=state, step_count=step_count, **pending
        )

    def handle_oracle_hse(
        self,
        *,
        env: Any,
        model: Any,
        state: HSEEpisodeState,
        step_count: int,
    ) -> dict[str, Any] | None:
        if self.variant != ORACLE_HSE:
            return None
        current_stage_idx = state.current_stage_idx
        stage_status = list(getattr(env, "subtask_completed", []))
        if current_stage_idx >= len(stage_status) or not stage_status[current_stage_idx]:
            return None
        return self._switch_to_next_stage(
            env=env,
            model=model,
            state=state,
            step_count=step_count,
            source="gt",
        )

    def observe_real_switch(
        self,
        *,
        env: Any,
        model: Any,
        state: HSEEpisodeState,
        observation: dict[str, Any],
        step_count: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.variant != ORDERED_HSE or self.real_switch_session is None:
            return (None, None)
        if state.pending_switch is not None:
            return (None, None)
        metadata = self.real_switch_session.observe(observation, env_step=step_count)
        if not metadata.get("should_switch", False):
            return (metadata, None)
        completed_stage_idx = state.current_stage_idx
        if completed_stage_idx < len(state.predictor_switch_steps) and state.predictor_switch_steps[completed_stage_idx] is None:
            state.predictor_switch_steps[completed_stage_idx] = int(step_count)
        env_status_at_switch = [bool(c) for c in getattr(env, "subtask_completed", [])]
        env_completed_at_switch = (
            completed_stage_idx < len(env_status_at_switch)
            and bool(env_status_at_switch[completed_stage_idx])
        )
        event = self._switch_to_next_stage(
            env=env,
            model=model,
            state=state,
            step_count=step_count,
            source="real_switch_predictor",
            predictor_reset_reason="real_switch_trigger",
            extra_event={
                "backend": "xvla",
                "predictor_switch_step": int(step_count),
                "env_stage_status_at_switch": env_status_at_switch,
                "env_subtask_completed_at_switch": env_completed_at_switch,
                "trigger_score": metadata.get("trigger_score"),
                "trigger_raw_score": metadata.get("trigger_raw_score"),
                "trigger_reason": metadata.get("trigger_reason"),
                "predictor_frame": metadata.get("trigger_frame"),
                "decision_rule": metadata.get("decision_rule"),
                "camera_key": metadata.get("camera_key"),
                "predictor_bundle_dir": metadata.get("predictor_bundle_dir"),
                "switch_mode": metadata.get("switch_mode"),
            },
        )
        return (metadata, event)

    def export_real_switch_trace(self) -> dict[str, Any] | None:
        if self.variant != ORDERED_HSE or self.real_switch_session is None:
            return None
        return self.real_switch_session.export_payload()

    def close(self) -> None:
        if self.real_switch_session is not None:
            self.real_switch_session.close()
