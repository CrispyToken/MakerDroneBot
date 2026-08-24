# services/skills.py
import re
import logging
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger("rag-bot")

@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: Path

class SkillManager:
    def __init__(self, dirs: list[Path]):
        self.dirs = dirs
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self):
        self.skills.clear()
        for d in self.dirs:
            if not d.exists():
                continue
            # rglob catches skills installed in subdirectories (e.g., .agents/skills/unlazy/SKILL.md)
            for path in d.rglob("SKILL.md"):
                try:
                    text = path.read_text(encoding="utf-8")
                    # Simple regex to split YAML frontmatter from Markdown content
                    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
                    if not match:
                        log.warning(f"No valid frontmatter found in {path}")
                        continue
                    
                    meta = match.group(1)
                    content = match.group(2).strip()
                    
                    name = ""
                    description = ""
                    # Lightweight YAML parsing for standard Agent Skills frontmatter
                    for line in meta.split('\n'):
                        if line.startswith('name:'):
                            name = line.split(':', 1)[1].strip().strip('"\'')
                        elif line.startswith('description:'):
                            description = line.split(':', 1)[1].strip().strip('"\'')
                    
                    if name:
                        self.skills[name.lower()] = Skill(name, description, content, path)
                        log.info(f"Loaded Agent Skill: {name}")
                except Exception as e:
                    log.warning(f"Failed to parse {path}: {e}")

    def get_skill(self, name: str) -> Skill | None:
        return self.skills.get(name.lower())

    def get_skills_prompt(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "[Available Agent Skills]",
            "You have access to a library of installed Agent Skills. If a user's request aligns with a skill's purpose, "
            "or if they explicitly ask to use one, you MUST use the `activate_skill` tool to load its instructions."
        ]
        for skill in self.skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

# Global instance initialized in bot.py or main.py
skill_manager: SkillManager | None = None