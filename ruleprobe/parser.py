"""Parse instruction files (AGENTS.md, CLAUDE.md, system prompts) into atomic Rules."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from .models import Rule


# ---------------------------------------------------------------------------
# Heuristics for splitting compound rules
# ---------------------------------------------------------------------------

SPLIT_PATTERNS = [
    r"\s+and never\s+",
    r"\s+but never\s+",
    r"\s+and always\s+",
    r"\s+;\s+(?=[A-Z])",
    r"\s+\.\s+(?=[A-Z])",
]

NEGATION_MAP = {
    "always": "never",
    "never": "always",
    "must": "must not",
    "must not": "must",
    "do": "do not",
    "do not": "do",
    "should": "should not",
    "should not": "should",
    "will": "will not",
    "will not": "will",
    "can": "cannot",
    "cannot": "can",
    "allow": "disallow",
    "disallow": "allow",
    "include": "exclude",
    "exclude": "include",
    "use": "avoid",
    "avoid": "use",
    "prefer": "avoid",
    "require": "not require",
    "ensure": "do not ensure",
    "provide": "do not provide",
    "reveal": "do not reveal",
    "keep": "do not keep",
}


class InstructionParser:
    """Parse a markdown instruction file into atomic Rule objects."""

    def __init__(self, split_compound: bool = True, pinned_ids: Sequence[str] | None = None):
        self.split_compound = split_compound
        self.pinned_ids: set[str] = set(pinned_ids or [])

    def parse(self, text: str) -> list[Rule]:
        """Parse instruction text → list of Rule objects."""
        rules: list[Rule] = []
        current_section = "root"
        position = 0

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            # Track markdown headings as section context
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                current_section = heading_match.group(2).strip()
                i += 1
                continue

            # Bullet points: -, *, +
            bullet_match = re.match(r"^(\s*)[-*+]\s+(.+)", line)
            if bullet_match:
                content = bullet_match.group(2).strip()
                # Collect continuation lines (indented more)
                indent = len(bullet_match.group(1))
                while i + 1 < len(lines):
                    next_line = lines[i + 1]
                    cont = re.match(r"^(\s+)(.+)", next_line)
                    if cont and len(cont.group(1)) > indent and not re.match(r"^\s*[-*+]\s+", next_line):
                        content += " " + cont.group(2).strip()
                        i += 1
                    else:
                        break
                new_rules = self._make_rules(content, current_section, position)
                rules.extend(new_rules)
                position += len(new_rules)
                i += 1
                continue

            # Numbered lists: 1. 2. etc.
            numbered_match = re.match(r"^\s*\d+[.)]\s+(.+)", line)
            if numbered_match:
                content = numbered_match.group(1).strip()
                new_rules = self._make_rules(content, current_section, position)
                rules.extend(new_rules)
                position += len(new_rules)
                i += 1
                continue

            # Plain paragraphs (non-empty, not a heading, not a code block)
            if line.strip() and not line.startswith("```") and not line.startswith("    "):
                # Only treat as rule if it looks like an instruction
                if self._looks_like_instruction(line.strip()):
                    new_rules = self._make_rules(line.strip(), current_section, position)
                    rules.extend(new_rules)
                    position += len(new_rules)

            i += 1

        # Apply pinned flag
        for rule in rules:
            if rule.id in self.pinned_ids or rule.text in self.pinned_ids:
                rule.pinned = True

        return rules

    def _looks_like_instruction(self, text: str) -> bool:
        """Heuristic: does this line look like an instruction vs prose/metadata?"""
        instruction_keywords = [
            r"\b(always|never|must|should|do not|don't|ensure|avoid|prefer|keep|use|provide|include|exclude|require|when|if)\b",
        ]
        for pat in instruction_keywords:
            if re.search(pat, text, re.IGNORECASE):
                return True
        # Short imperative sentences
        if len(text.split()) <= 15 and text[0].isupper():
            return True
        return False

    def _make_rules(self, text: str, section: str, start_position: int) -> list[Rule]:
        """Optionally split compound rules, return list."""
        texts = [text]
        if self.split_compound:
            texts = self._split_compound(text)

        rules = []
        for idx, t in enumerate(texts):
            t = t.strip().strip(".,;")
            if not t:
                continue
            pos = start_position + idx
            raw_hash = hashlib.sha1(f"{pos}:{t}".encode()).hexdigest()[:12]
            rule_id = f"R{pos}_{raw_hash[:6]}"
            rules.append(Rule(
                id=rule_id,
                text=t,
                section=section,
                position=pos,
                hash=raw_hash,
            ))
        return rules

    def _split_compound(self, text: str) -> list[str]:
        """Split 'Always do X and never do Y' into two rules."""
        for pat in SPLIT_PATTERNS:
            parts = re.split(pat, text, maxsplit=1)
            if len(parts) > 1:
                return [p.strip() for p in parts if p.strip()]
        return [text]

    def rebuild_prompt(self, original: str, rules: list[Rule]) -> str:
        """Rebuild a prompt from a list of rules (used by operators)."""
        lines = []
        current_section = None
        for rule in rules:
            if rule.section != current_section:
                if rule.section and rule.section != "root":
                    lines.append(f"\n## {rule.section}\n")
                current_section = rule.section
            lines.append(f"- {rule.text}")
        return "\n".join(lines)

    @staticmethod
    def negate_rule_text(text: str) -> str:
        """Best-effort negation of a rule's text."""
        lower = text.lower()
        # Try negation map on first matching keyword
        for word, opposite in NEGATION_MAP.items():
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, lower):
                return re.sub(pattern, opposite, text, count=1, flags=re.IGNORECASE)
        # Fallback: prepend "Do NOT follow this rule:"
        return f"[NEGATED — do NOT follow] {text}"


def parse_file(path: str, **kwargs) -> list[Rule]:
    """Convenience function: parse a file path."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    parser = InstructionParser(**kwargs)
    return parser.parse(text)
