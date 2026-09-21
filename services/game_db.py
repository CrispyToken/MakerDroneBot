import csv
import json
import logging
from pathlib import Path
from config import EH_DATABASE_DIR

log = logging.getLogger("rag-bot")

AVAILABILITY_NAMES = {0: "Unobtainable", 1: "Common", 2: "Rare", 3: "Special", 4: "Hidden"}
DAMAGE_TYPE_NAMES = {0: "Kinetic", 1: "Energy", 2: "Heat", 3: "Direct"}
SIZE_CLASS_NAMES = {1: "Destroyer", 2: "Cruiser", 3: "Battleship", 4: "Capital Ship", 5: "Drone"}
VETERAN_NAMES = {0: "Non-veteran", 1: "Veteran", 2: "Double veteran", 3: "Triple veteran"}
WEAPON_CLASS_NAMES = {0: "Common", 1: "Manageable", 2: "Continuous", 3: "Machine gun", 4: "Multishot", 5: "Charged"}

QUALITY_EFFECTS = {
    1: {0: "+100% weight", 1: "+50% weight", 2: "+20% weight", 3: "-20% weight", 4: "-40% weight", 5: "-50% weight"},
    2: {0: "+100% energy cost", 1: "+50% energy cost", 2: "+20% energy cost", 3: "-10% energy cost", 4: "-25% energy cost", 5: "-50% energy cost"},
    3: {0: "-50% defense", 1: "-30% defense", 2: "-20% defense", 3: "+20% defense", 4: "+50% defense", 5: "+100% defense"},
    4: {0: "-3 hit points", 1: "-2 hit points", 2: "-1 hit point", 3: "+1 hit point", 4: "+3 hit points", 5: "+5 hit points"},
    5: {0: "-50% damage", 1: "-30% damage", 2: "-20% damage", 3: "+20% damage", 4: "+50% damage", 5: "+100% damage"},
    6: {0: "+100% cooldown time", 1: "+50% cooldown time", 2: "+20% cooldown time", 3: "-10% cooldown time", 4: "-25% cooldown time", 5: "-50% cooldown time"},
    7: {0: "-50% range", 1: "-30% range", 2: "-20% range", 3: "+10% range", 4: "+25% range", 5: "+50% range"},
    8: {0: "-50% projectile speed", 1: "-30% projectile speed", 2: "-20% projectile speed", 3: "+10% projectile speed", 4: "+25% projectile speed", 5: "+50% projectile speed"},
    9: {0: "-50% energy capacity", 1: "-30% energy capacity", 2: "-20% energy capacity", 3: "+20% energy capacity", 4: "+50% energy capacity", 5: "+100% energy capacity"},
    10: {0: "-50% repair rate", 1: "-30% repair rate", 2: "-20% repair rate", 3: "+10% repair rate", 4: "+25% repair rate", 5: "+50% repair rate"},
    11: {0: "-50% engine power", 1: "-30% engine power", 2: "-20% engine power", 3: "+10% engine power", 4: "+25% engine power", 5: "+50% engine power"},
    12: {0: "-50% recharge rate", 1: "-30% recharge rate", 2: "-20% recharge rate", 3: "+10% recharge rate", 4: "+25% recharge rate", 5: "+50% recharge rate"},
    13: {0: "-30% projectile speed, -20% damage", 1: "-20% projectile speed, -15% damage", 2: "-10% projectile speed, -10% damage", 3: "+20% projectile speed, -10% damage", 4: "+50% projectile speed, -20% damage", 5: "+100% projectile speed, -25% damage"},
    14: {0: "-50% area of effect", 1: "-30% area of effect", 2: "-20% area of effect", 3: "+25% area of effect", 4: "+60% area of effect", 5: "+100% area of effect"},
    15: {0: "-50% shield power", 1: "-30% shield power", 2: "-20% shield power", 3: "+10% shield power", 4: "+25% shield power", 5: "+50% shield power"},
    16: {0: "-60% damage, -30% cooldown time", 1: "-40% damage, -20% cooldown time", 2: "-15% damage, -10% cooldown time", 3: "+40% damage, +10% cooldown time", 4: "+100% damage, +25% cooldown time", 5: "+200% damage, +50% cooldown time"},
    17: {0: "-50% drone damage", 1: "-30% drone damage", 2: "-20% drone damage", 3: "+30% drone damage", 4: "+80% drone damage", 5: "+150% drone damage"},
    18: {0: "-50% drone defense", 1: "-30% drone defense", 2: "-20% drone defense", 3: "+30% drone defense", 4: "+80% drone defense", 5: "+150% drone defense"},
    19: {0: "-50% drone speed", 1: "-30% drone speed", 2: "-20% drone speed", 3: "+20% drone speed", 4: "+50% drone speed", 5: "+80% drone speed"},
    20: {0: "-50% drone range", 1: "-30% drone range", 2: "-20% drone range", 3: "+20% drone range", 4: "+50% drone range", 5: "+80% drone range"},
    21: {0: "+150% projectile weight", 1: "+100% projectile weight", 2: "+50% projectile weight", 3: "-20% projectile weight", 4: "-50% projectile weight", 5: "-80% projectile weight"},
}

_STAT_FIELDS = (
    ("ArmorPoints", "Armor points"),
    ("ArmorRepairRate", "Armor repair rate"),
    ("HullPoints", "Hull points"),
    ("HullRepairRate", "Hull repair rate"),
    ("HullRepairCooldownModifier", "Hull repair cooldown modifier"),
    ("EnergyPoints", "Energy points"),
    ("EnergyRechargeRate", "Energy recharge rate"),
    ("EnergyRechargeCooldownModifier", "Energy recharge cooldown modifier"),
    ("ShieldPoints", "Shield points"),
    ("ShieldRechargeRate", "Shield recharge rate"),
    ("Weight", "Weight"),
    ("RammingDamage", "Ramming damage"),
    ("EnergyAbsorption", "Energy absorption"),
    ("KineticResistance", "Kinetic resistance"),
    ("EnergyResistance", "Energy resistance"),
    ("ThermalResistance", "Thermal resistance"),
    ("EnginePower", "Engine power"),
    ("TurnRate", "Turn rate"),
    ("DroneRangeModifier", "Drone range modifier"),
    ("DroneDamageModifier", "Drone damage modifier"),
    ("DroneDefenseModifier", "Drone defense modifier"),
    ("DroneSpeedModifier", "Drone speed modifier"),
    ("DronesBuiltPerSecond", "Drones built per second"),
    ("DroneBuildTimeModifier", "Drone build time modifier"),
    ("WeaponFireRateModifier", "Weapon fire rate modifier"),
    ("WeaponDamageModifer", "Weapon damage modifier"),
    ("WeaponDamageModifier", "Weapon damage modifier"),
    ("WeaponRangeModifier", "Weapon range modifier"),
    ("WeaponEnergyCostModifier", "Weapon energy cost modifier"),
)

_MAX_RESULT_CHARS = 12000
_MAX_MATCHES = 6
_MAX_BUILDS_SHOWN = 8
_MAX_BUILD_LINES = 40


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.warning("Game DB: could not parse %s", path)
        return None


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [row for row in csv.reader(f) if row and any(cell.strip() for cell in row)]
    except FileNotFoundError:
        log.warning("Game DB: lookup table not found: %s", path)
        return []


class GameDatabase:
    """Read-only, in-memory index of the Event Horizon vanilla database.
    Loaded once at startup; every lookup is pure dict access."""

    def __init__(self) -> None:
        self.loaded = False
        self.root: Path | None = None
        self.base_armor_points = 0.0
        self.armor_points_per_cell = 0.5
        self.default_weight_per_cell = 20.0
        self.minimum_weight_per_cell = 10.0
        self.modules: dict[str, str] = {}
        self.module_names: dict[str, list[str]] = {}
        self.ships: dict[str, dict] = {}
        self.ship_names: dict[str, list[str]] = {}
        self.ship_files: dict[str, dict] = {}
        self.component_files: dict[str, dict] = {}
        self.components_by_id: dict[int, dict] = {}
        self.component_name_by_file: dict[str, str] = {}
        self.stats_by_id: dict[int, dict] = {}
        self.ammo_by_id: dict[int, dict] = {}
        self.weapons_by_id: dict[int, dict] = {}
        self.techs_by_id: dict[int, dict] = {}
        self.tech_by_stem: dict[str, dict] = {}
        self.tech_by_name: dict[str, str] = {}
        self.factions: dict[str, str] = {}
        self.modifications: dict[int, str] = {}
        self.builds_by_ship: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        root = EH_DATABASE_DIR / "Database" if (EH_DATABASE_DIR / "Database").is_dir() else EH_DATABASE_DIR
        if not root.is_dir():
            log.warning("Game DB: directory not found at %s; game_lookup will be unavailable.", EH_DATABASE_DIR)
            return
        self.root = root

        self._scan_components(root / "Component")
        self._scan_stats(root / "Component" / "Stats")
        self._scan_by_id(root / "Ammunition", self.ammo_by_id)
        self._scan_by_id(root / "Weapon", self.weapons_by_id)
        self._scan_technology(root / "Technology")
        self._scan_ships(root / "Ship")
        self._scan_builds(root / "Ship" / "Builds")
        self._load_ship_settings(root / "Settings" / "Ships.json")

        csv_dir = root if (root / "modulelookuptable.csv").exists() else root.parent
        self._load_factions(csv_dir / "factionlookuptable.csv")
        self._load_modifications(csv_dir / "modificationlookuptable.csv")
        self._load_modules(csv_dir / "modulelookuptable.csv")
        self._load_ships(csv_dir / "shiplookuptable.csv")
        self._load_tech_table(csv_dir / "techlookuptable.csv")

        self.loaded = True
        log.info(
            "Game DB ready at %s: %d modules, %d ships, %d components, %d stats, %d ammo, %d weapons, %d techs, %d builds.",
            root, len(self.module_names), len(self.ship_names), len(self.components_by_id),
            len(self.stats_by_id), len(self.ammo_by_id), len(self.weapons_by_id),
            len(self.techs_by_id), sum(len(v) for v in self.builds_by_ship.values()),
        )

    def _scan_components(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is None:
                continue
            self.component_files[path.name] = data
            if isinstance(data.get("Id"), int):
                self.components_by_id[data["Id"]] = {"filename": path.name, "data": data}

    def _scan_stats(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is not None and isinstance(data.get("Id"), int):
                self.stats_by_id[data["Id"]] = data

    def _scan_by_id(self, directory: Path, target: dict[int, dict]) -> None:
        for folder in (directory / "Obsolete", directory):
            if not folder.is_dir():
                continue
            for path in folder.glob("*.json"):
                data = _read_json(path)
                if data is not None and isinstance(data.get("Id"), int):
                    target[data["Id"]] = data

    def _scan_technology(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is None:
                continue
            self.tech_by_stem[path.stem] = data
            if isinstance(data.get("Id"), int):
                self.techs_by_id[data["Id"]] = data

    def _scan_ships(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is not None:
                self.ship_files[path.name] = data

    def _scan_builds(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            data = _read_json(path)
            if data is None:
                continue
            ship_id = data.get("ShipId")
            if isinstance(ship_id, int):
                self.builds_by_ship.setdefault(ship_id, []).append(data)

    def _load_ship_settings(self, path: Path) -> None:
        data = _read_json(path)
        if data is None:
            log.warning("Game DB: Settings/Ships.json not found; derived ship stats use vanilla defaults.")
            return
        self.base_armor_points = float(data.get("BaseArmorPoints", self.base_armor_points))
        self.armor_points_per_cell = float(data.get("ArmorPointsPerCell", self.armor_points_per_cell))
        self.default_weight_per_cell = float(data.get("DefaultWeightPerCell", self.default_weight_per_cell))
        self.minimum_weight_per_cell = float(data.get("MinimumWeightPerCell", self.minimum_weight_per_cell))

    def _load_factions(self, path: Path) -> None:
        for row in _read_csv_rows(path)[1:]:
            if len(row) < 2:
                continue
            key, name = row[0].strip(), row[1].strip()
            if key and name:
                self.factions[key] = name

    def _load_modifications(self, path: Path) -> None:
        for row in _read_csv_rows(path)[1:]:
            if len(row) < 3:
                continue
            try:
                self.modifications[int(row[0])] = row[2].strip()
            except ValueError:
                continue

    def _load_modules(self, path: Path) -> None:
        for row in _read_csv_rows(path)[1:]:
            if len(row) < 2:
                continue
            name, filename = row[0].strip(), row[1].strip()
            if not name or not filename or name == "BREAK":
                continue
            if filename not in self.component_files:
                continue
            self.modules[name] = filename
            self.component_name_by_file.setdefault(filename, name)
            keys = self.module_names.setdefault(_norm(name), [])
            if name not in keys:
                keys.append(name)

    def _load_ships(self, path: Path) -> None:
        for row in _read_csv_rows(path)[1:]:
            if len(row) < 2:
                continue
            name, filename = row[0].strip(), row[1].strip()
            description = row[2].strip() if len(row) > 2 else ""
            if not name or not filename or name == "BREAK" or name.startswith("MODDED SHIPS BELOW"):
                continue
            if filename not in self.ship_files:
                continue
            self.ships[name] = {"filename": filename, "description": description}
            keys = self.ship_names.setdefault(_norm(name), [])
            if name not in keys:
                keys.append(name)

    def _load_tech_table(self, path: Path) -> None:
        for row in _read_csv_rows(path)[1:]:
            if len(row) < 2:
                continue
            name, filename = row[0].strip(), row[1].strip()
            if not name or name.startswith("//") or name == "BREAK":
                continue
            if not filename or filename.startswith("//"):
                continue
            if not filename.endswith(".json"):
                filename += ".json"
            stem = Path(filename).stem
            if stem in self.tech_by_stem:
                self.tech_by_name.setdefault(_norm(name), stem)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def _faction_name(self, raw) -> str:
        if raw is None:
            return "Unknown"
        return self.factions.get(str(raw), "Unknown")

    def _tech_for_name(self, name: str) -> dict | None:
        stem = self.tech_by_name.get(_norm(name))
        return self.tech_by_stem.get(stem) if stem else None

    def _tech_tree_sum(self, tech_id: int, seen: set[int]) -> int | None:
        tech = self.techs_by_id.get(tech_id)
        if tech is None:
            return None
        total = int(tech.get("Price") or 0)
        for dep_id in tech.get("Dependencies") or []:
            if not isinstance(dep_id, int) or dep_id in seen:
                continue
            seen.add(dep_id)
            sub = self._tech_tree_sum(dep_id, seen)
            if sub is None:
                return None
            total += sub
        return total

    def _workshop_level(self, entity_name: str) -> int | None:
        tech = self._tech_for_name(entity_name)
        if tech is None or not isinstance(tech.get("Id"), int):
            return None
        return self._tech_tree_sum(tech["Id"], {tech["Id"]})

    def _component_display_name(self, component_id) -> str:
        record = self.components_by_id.get(component_id) if isinstance(component_id, int) else None
        if record is None:
            return f"Unknown component (id {component_id})"
        name = self.component_name_by_file.get(record["filename"])
        if name:
            return name
        raw = record["data"].get("Name") or record["filename"]
        return str(raw).lstrip("$")

    def _quality_effect(self, modification, quality) -> str:
        if not isinstance(modification, int) or not isinstance(quality, int) or modification == 0:
            return ""
        return QUALITY_EFFECTS.get(modification, {}).get(quality, "")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(self, query: str) -> str:
        if not self.loaded:
            return "The game database is not loaded. game_lookup is unavailable."
        q = _norm(query or "")
        if not q:
            return "Provide the name of a ship or module to look up."

        matches = self._find_matches(q)
        if not matches:
            return (
                f"No ship or module matching '{query}' was found in the game database "
                f"({len(self.ship_names)} ships, {len(self.module_names)} modules indexed). "
                "Try the exact in-game name."
            )

        sections = [self._render_ship(key) if kind == "ship" else self._render_module(key)
                    for kind, key in matches]
        result = ("\n\n" + "=" * 60 + "\n\n").join(sections)
        if len(result) > _MAX_RESULT_CHARS:
            result = result[:_MAX_RESULT_CHARS] + "\n\n[Result truncated. Ask a more specific question for full data.]"
        return result

    def _find_matches(self, q: str) -> list[tuple[str, str]]:
        matches: list[tuple[str, str]] = []

        def add(kind: str, keys: list[str]) -> None:
            for key in keys:
                if (kind, key) not in matches:
                    matches.append((kind, key))

        for kind, index in (("ship", self.ship_names), ("module", self.module_names)):
            if q in index:
                add(kind, index[q])
        if matches:
            return matches[:_MAX_MATCHES]

        scored: list[tuple[int, str, str]] = []
        for kind, index in (("ship", self.ship_names), ("module", self.module_names)):
            for name in index:
                if name in q or (len(q) >= 3 and (q in name or name.startswith(q) or q.startswith(name))):
                    scored.append((len(name), kind, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, kind, name in scored:
            index = self.ship_names if kind == "ship" else self.module_names
            add(kind, index[name])
            if len(matches) >= _MAX_MATCHES:
                break
        return matches[:_MAX_MATCHES]

    # ------------------------------------------------------------------
    # Rendering — ships
    # ------------------------------------------------------------------
    def _render_ship(self, key: str) -> str:
        entry = self.ships[key]
        data = self.ship_files.get(entry["filename"]) or {}
        lines = [f"SHIP: {key.title()}"]
        lines.append(f"Class: {SIZE_CLASS_NAMES.get(data.get('SizeClass'), 'Frigate')}")
        lines.append(f"Faction: {self._faction_name(data.get('Faction'))}")
        description = entry.get("description", "").strip()
        if description:
            lines.append(f"Description: {description}")

        layout = data.get("Layout") or ""
        cells = sum(1 for ch in layout if ch != "0")
        features = data.get("Features") or {}

        def ship_stat(name: str):
            return data[name] if name in data else features.get(name)

        weight_modifier = ship_stat("BaseWeightModifier") or 0
        hp = round(cells * self.armor_points_per_cell + self.base_armor_points, 1)
        base_weight = int(cells * self.default_weight_per_cell * (1 + weight_modifier))
        min_weight = int(cells * self.minimum_weight_per_cell * (1 + weight_modifier))
        cost = 15 * cells ** 2 if data.get("SizeClass") == 4 else 5 * cells ** 2
        lines.append(f"Hull cells: {cells}")
        lines.append(f"Hitpoints: {_fmt(hp)}")
        lines.append(f"Base weight: {base_weight} (minimum {min_weight})")
        lines.append(f"Crafting cost: {cost}")

        for raw_key, label in (("KineticResistance", "Kinetic resistance"),
                               ("HeatResistance", "Heat resistance"),
                               ("EnergyResistance", "Energy resistance")):
            value = ship_stat(raw_key)
            if isinstance(value, (int, float)):
                lines.append(f"{label}: {int(100 - 100 / (value + 1))}%")
        velocity_bonus = ship_stat("VelocityBonus")
        if isinstance(velocity_bonus, (int, float)):
            lines.append(f"Velocity bonus: {int(velocity_bonus * 100)}%")
        if ship_stat("Regeneration"):
            lines.append("Living ship: regenerates its own HP")

        barrels = data.get("Barrels") or []
        if barrels:
            counts: dict[str, int] = {}
            for barrel in barrels:
                slot = str(barrel.get("WeaponClass", "?"))
                counts[slot] = counts.get(slot, 0) + 1
            summary = ", ".join(f"{count}x {slot}" for slot, count in sorted(counts.items()))
            lines.append(f"Weapon barrels: {summary}")

        workshop = self._workshop_level(key)
        lines.append(f"Workshop level: {workshop if workshop is not None else 'Unknown'}")

        lines.extend(self._render_builds(data.get("Id")))
        return "\n".join(lines)

    def _render_builds(self, ship_id) -> list[str]:
        if not isinstance(ship_id, int) or ship_id not in self.builds_by_ship:
            return []
        lines = ["Builds:"]
        builds = sorted(self.builds_by_ship[ship_id],
                        key=lambda b: (b.get("DifficultyClass", 0), b.get("Id", 0)))
        shown = 0
        for build in builds:
            if shown >= _MAX_BUILDS_SHOWN:
                lines.append(f"  (+{len(builds) - shown} more builds not shown)")
                break
            shown += 1
            label = VETERAN_NAMES.get(build.get("DifficultyClass", 0), f"Veteran {build.get('DifficultyClass')}")
            flags = []
            if build.get("NotAvailableInGame"):
                flags.append("not in game")
            if build.get("AvailableForPlayer"):
                flags.append("usable by player")
            if build.get("AvailableForEnemy"):
                flags.append("used by enemies")
            suffix = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"  {label}{suffix}:")
            counts: dict[str, int] = {}
            for component in build.get("Components") or []:
                name = self._component_display_name(component.get("ComponentId"))
                effect = self._quality_effect(component.get("Modification"), component.get("Quality"))
                entry_name = f"{name} ({effect})" if effect else name
                counts[entry_name] = counts.get(entry_name, 0) + 1
            items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            for name, count in items[:_MAX_BUILD_LINES]:
                lines.append(f"    {count}x {name}")
            if len(items) > _MAX_BUILD_LINES:
                lines.append(f"    (+{len(items) - _MAX_BUILD_LINES} more module types)")
        return lines

    # ------------------------------------------------------------------
    # Rendering — modules
    # ------------------------------------------------------------------
    def _render_module(self, key: str) -> str:
        data = self.component_files.get(self.modules[key]) or {}
        lines = [f"MODULE: {key.title()}"]
        lines.append(f"Availability: {AVAILABILITY_NAMES.get(data.get('Availability'), 'Unknown')}")

        faction = self._faction_name(data.get("Faction"))
        if faction == "Unknown":
            tech = self._tech_for_name(key)
            if tech is not None:
                faction = self._faction_name(tech.get("Faction"))
        lines.append(f"Faction: {faction}")

        slot_type = data.get("WeaponSlotType")
        if slot_type:
            lines.append(f"Weapon slot type: {slot_type}")

        layout = data.get("Layout") or ""
        if layout:
            size = int(len(layout) ** 0.5)
            filled = sum(1 for ch in layout if ch != "0")
            lines.append(f"Size: {size}x{size} ({filled} cells)")

        level = data.get("Level")
        if isinstance(level, int):
            lines.append(f"Workshop level requirement: {level}")
            base_price = 50 + level * 20
            if isinstance(data.get("WeaponId"), int):
                base_price *= 2
            lines.append(f"Crafting cost (base quality): {base_price}")
        workshop = self._workshop_level(key)
        if workshop is not None:
            lines.append(f"Total research price (tech tree): {workshop}")

        modification_ids = data.get("PossibleModifications") or []
        mod_names = [self.modifications.get(m, f"Unknown ({m})")
                     for m in modification_ids if isinstance(m, int) and m != 0]
        if mod_names:
            lines.append("Possible modifications: " + ", ".join(mod_names))

        lines.extend(self._render_weapon(data))
        lines.extend(self._render_ammunition(data))
        lines.extend(self._render_component_stats(data, layout))
        return "\n".join(lines)

    def _render_weapon(self, data: dict) -> list[str]:
        weapon_id = data.get("WeaponId")
        weapon = self.weapons_by_id.get(weapon_id) if isinstance(weapon_id, int) else None
        if weapon is None:
            return []
        lines = ["Weapon:"]
        weapon_class = WEAPON_CLASS_NAMES.get(weapon.get("WeaponClass"))
        if weapon_class:
            lines.append(f"  Class: {weapon_class}")
        fire_rate = weapon.get("FireRate")
        if isinstance(fire_rate, (int, float)) and fire_rate > 0:
            lines.append(f"  Reload time: {round(1 / fire_rate, 2)}s (fire rate {_fmt(fire_rate)})")
        for field, label in (("Magazine", "Magazine"), ("Spread", "Spread")):
            if field in weapon:
                lines.append(f"  {label}: {_fmt(weapon[field])}")
        return lines

    def _render_ammunition(self, data: dict) -> list[str]:
        ammo_id = data.get("AmmunitionId")
        ammo = self.ammo_by_id.get(ammo_id) if isinstance(ammo_id, int) else None
        if ammo is None:
            return []
        lines = ["Ammunition:"]
        body = ammo.get("Body") or {}
        for effect in ammo.get("Effects") or []:
            damage_type = DAMAGE_TYPE_NAMES.get(effect.get("DamageType"), "Unknown")
            power = effect.get("Power")
            if power is not None:
                lines.append(f"  Damage: {_fmt(power)} ({damage_type})")
            else:
                lines.append(f"  Damage type: {damage_type}")
        if "Damage" in ammo:
            lines.append(f"  Damage: {_fmt(ammo['Damage'])}")
        seen_fields: set[str] = set()
        for field, label, sources in (
                ("EnergyCost", "Energy cost", (ammo, body)),
                ("Range", "Range", (ammo, body)),
                ("Velocity", "Velocity", (ammo, body)),
                ("Impulse", "Impulse", (ammo, body)),
                ("Lifetime", "Lifetime", (body, ammo)),
                ("LifeTime", "Lifetime", (ammo, body)),
        ):
            if label in seen_fields:
                continue
            for source in sources:
                if field in source:
                    lines.append(f"  {label}: {_fmt(source[field])}")
                    seen_fields.add(label)
                    break
        return lines

    def _render_component_stats(self, data: dict, layout: str) -> list[str]:
        stats_id = data.get("ComponentStatsId")
        stats = self.stats_by_id.get(stats_id) if isinstance(stats_id, int) else None
        if stats is None:
            return []
        modifier = 1
        if stats.get("Type"):
            modifier = max(1, sum(1 for ch in layout if ch != "0"))
        lines = ["Stats:"]
        seen_labels: set[str] = set()
        for field, label in _STAT_FIELDS:
            if field not in stats or label in seen_labels:
                continue
            seen_labels.add(label)
            raw = stats[field]
            value = raw * modifier if isinstance(raw, (int, float)) else raw
            if field.endswith("Modifier") or field == "DronesBuiltPerSecond":
                if isinstance(value, (int, float)):
                    lines.append(f"  {label}: {round(value * 100)}%")
                    continue
            lines.append(f"  {label}: {_fmt(value)}")
        return lines


game_database = GameDatabase()


def load_game_database() -> None:
    game_database.load()