from __future__ import annotations

import re
from collections.abc import Mapping


def natural_command_key(value: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)(.*)$", value)
    return (int(match.group(1)), match.group(2)) if match else (999, value)


def sorted_command_keys(commands: Mapping[str, object]) -> list[str]:
    return sorted(commands, key=natural_command_key)


def resolve_alias(aliases: Mapping[str, str], command: str) -> str:
    try:
        return aliases[command.strip().lower()]
    except KeyError:
        raise ValueError(f"Neznámý příkaz: {command!r}") from None
