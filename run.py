#!/usr/bin/env python3
"""Interactive launcher for TextDD experiments."""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import re
import subprocess
import sys

import questionary
from questionary import Style, Choice

REPO = Path(__file__).parent.resolve()
ENV_FILE = REPO / ".env"

EXPERIMENTS: dict[str, str] = {
    "embed": "expts/embed.py",
    "eval_cls": "expts/eval_cls.py",
    "eval_nli": "expts/eval_nli.py",
    "condense": "expts/condense.py",
    "finetune": "expts/finetune.py",
    "generate": "expts/generate.py",
}

STYLE = Style(
    [
        ("qmark", "fg:#00bfff bold"),
        ("question", "bold"),
        ("answer", "fg:#00bfff bold"),
        ("pointer", "fg:#00bfff bold"),
        ("highlighted", "fg:#00bfff bold"),
        ("selected", "fg:#00bfff"),
        ("instruction", "fg:#888888"),
    ]
)

# ── ANSI ─────────────────────────────────────────────────────────────────────

SHORTCUT_KEYS = "123456789abcdefghijklmnopqrstuvwxyz"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[38;2;0;191;255m"  # #00bfff, matches questionary answer color

_printed_lines = 0  # lines printed since last redraw anchor


def _group_line(group: "Group") -> str:
    chunks = []
    for f in group.fields:
        if f.val == "?":
            chunks.append(f"{DIM}{f.name}=?{RESET}")
        else:
            chunks.append(f"{f.name}={BLUE}{BOLD}{f.val}{RESET}")
    return f"{DIM}{group.name}:{RESET} " + "  ".join(chunks)


def _section(label: str) -> str:
    return f"{DIM}[ {label} ]{RESET}"


def _redraw(
    header: list[tuple[str, str]],
    done_groups: list["Group"] | None = None,
    current_group: "Group | None" = None,
) -> None:
    """Clear back to anchor and reprint header + completed group summaries + current group."""
    global _printed_lines
    if _printed_lines > 0:
        sys.stdout.write(f"\033[{_printed_lines}A\033[J")
        sys.stdout.flush()
    _printed_lines = 0

    print(_section("JOB CONFIG"))
    _printed_lines += 1
    if header:
        summary = "  ".join(f"{k}={BLUE}{BOLD}{v}{RESET}" for k, v in header)
        print(f" > {summary}")
        _printed_lines += 1

    if done_groups is not None or current_group is not None:
        print(_section("EXPT CONFIG"))
        _printed_lines += 1
        for g in done_groups or []:
            print(_group_line(g))
            _printed_lines += 1
        if current_group is not None:
            print(_group_line(current_group))
            _printed_lines += 1


def _questionary_done() -> None:
    """Call after each questionary prompt — it printed 1 answered line."""
    global _printed_lines
    _printed_lines += 1


# ── .env loader ───────────────────────────────────────────────────────────────


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Z0-9_]+)="?([^"]*)"?$', line)
        if m:
            env[m.group(1)] = m.group(2)
    return env


# ── Field / help parser ───────────────────────────────────────────────────────


@dataclass
class Field:
    name: str
    choices: list[str] = field(default_factory=list)  # non-empty → select prompt
    val: str = "?"


@dataclass
class Group:
    name: str
    fields: list[Field] = field(default_factory=list)


_help_cache: dict[str, list[Group]] = {}


class JobConfigStep(IntEnum):
    INFRA = 0
    SPEC = 1
    DURATION = 2
    EXPERIMENT = 3


def parse_help(script: str) -> list[Group]:
    """Run script --help and return one Group per argparse group (skipping 'options')."""
    if script in _help_cache:
        return _help_cache[script]

    result = subprocess.run(
        [sys.executable, str(REPO / script), "--help"],
        capture_output=True,
        text=True,
    )
    text = result.stdout

    groups: list[Group] = []
    current: Group | None = None
    for line in text.splitlines():
        # group header: ends with colon, not indented
        if re.match(r"^[A-Za-z].*:$", line):
            if line.lower().startswith("option"):
                current = None
            else:
                current = Group(name=line.rstrip(":"))
                groups.append(current)
            continue
        if current is None:
            continue
        # argument line: "  --name {a,b,c}" or "  --name NAME"
        m = re.match(r"^\s{1,4}(--[\w.\-]+)\s*(\{[^}]+\}|[A-Z_]+)?", line)
        if m and not m.group(1).startswith("--help"):
            name = m.group(1).lstrip("-")
            typ = m.group(2) or ""
            if typ.startswith("{"):
                choices = [c.strip() for c in typ.strip("{}").split(",")]
                current.fields.append(Field(name=name, choices=choices))
            else:
                current.fields.append(Field(name=name))

    _help_cache[script] = groups
    return groups


# ── kwargs picker ─────────────────────────────────────────────────────────────


def _pick_group(
    group: Group,
    is_meta: bool,
    state: list[tuple[str, str]],
    done_groups: list[Group],
    can_go_back: bool,
    start_idx: int = 0,
) -> int | None:
    """Prompt each field in a group.

    Returns:
        - None: exit
        - -1: group completed
        - >= 0: go back to the previous group and resume at this field index
    """
    idx = start_idx
    while idx < len(group.fields):
        f = group.fields[idx]
        _redraw(state, done_groups, group)

        if f.choices:
            show_back = idx > 0 or can_go_back
            back_choice = Choice("← Back", value="0", shortcut_key="0") if show_back else None
            skippable = "None" in f.choices
            display_choices = [c for c in f.choices if c != "None"]
            default = f.val if f.val in display_choices else display_choices[0]
            # Reserve 0 for back, so real choices use 1-9 then a-z.
            use_shortcuts = len(display_choices) + (1 if show_back else 0) + (1 if skippable else 0) <= len(SHORTCUT_KEYS)
            use_search_filter = not use_shortcuts
            prompt = f.name if use_shortcuts else f"{f.name} (type to filter)"
            if use_shortcuts:
                choices = [
                    Choice(choice, shortcut_key=SHORTCUT_KEYS[index])
                    for index, choice in enumerate(display_choices)
                ]
                if skippable:
                    choices.append(Choice("[skip]", value="__skip__", shortcut_key=SHORTCUT_KEYS[len(display_choices)]))
                if back_choice is not None:
                    choices.append(back_choice)
            else:
                choices = display_choices + (["[skip]"] if skippable else []) + ([back_choice] if back_choice is not None else [])
            ans = questionary.select(
                prompt,
                choices=choices,
                default=default,
                style=STYLE,
                use_shortcuts=use_shortcuts,
                use_search_filter=use_search_filter,
                use_jk_keys=not use_search_filter,
            ).ask()
            if ans is None:
                return None
            _questionary_done()
            if ans == "0":
                if idx > 0:
                    idx -= 1
                    group.fields[idx].val = "?"
                else:
                    if not done_groups:
                        return None
                    return max(len(done_groups[-1].fields) - 1, 0)
            elif ans in ("__skip__", "[skip]"):
                f.val = "?"
                idx += 1
            else:
                f.val = ans
                idx += 1

        else:
            ans = questionary.text(
                f"{f.name} (- to go back)",
                default=f.val if f.val != "?" else "",
                style=STYLE,
            ).ask()
            if ans is None:
                return None
            _questionary_done()
            if ans == "-":
                if idx > 0:
                    idx -= 1
                    group.fields[idx].val = "?"
                else:
                    if not done_groups:
                        return None
                    return max(len(done_groups[-1].fields) - 1, 0)
            else:
                if ans == "":
                    f.val = "?"
                else:
                    f.val = ans
                idx += 1

    return -1


def pick_kwargs(
    exp: str, state: list[tuple[str, str]], prev_fields: list[Field] | None = None
) -> tuple[str, list[Field]] | None:
    groups = parse_help(EXPERIMENTS[exp])
    if not groups:
        return "", []

    # pre-fill from previous run
    if prev_fields:
        prev = {f.name: f.val for f in prev_fields}
        for g in groups:
            for f in g.fields:
                if f.name in prev and prev[f.name] != "?":
                    f.val = prev[f.name]

    gi = 0
    resume_idx = 0
    while gi < len(groups):
        result = _pick_group(
            groups[gi],
            is_meta=(gi == 0),
            state=state,
            done_groups=groups[:gi],
            can_go_back=gi > 0,
            start_idx=resume_idx,
        )
        resume_idx = 0
        if result is None:
            return None
        if result >= 0:
            gi -= 1
            if gi < 0:
                return None
            resume_idx = result
            continue
        gi += 1

    all_fields = [f for g in groups for f in g.fields]
    parts = [f"--{f.name} {f.val}" for f in all_fields if f.val != "?"]
    return " ".join(parts), all_fields


# ── preset picker ─────────────────────────────────────────────────────────────


def pick_preset_labeled(
    prefix: str, title: str, env: dict[str, str], state: list[tuple[str, str]]
) -> tuple[str, str | None]:
    names, vals = [], []
    for key in sorted(k for k in env if k.startswith(prefix)):
        label = re.sub(r"^\d+_", "", key[len(prefix) :])
        names.append(label)
        vals.append(env[key])
    _redraw(state)
    ans = questionary.select(
        f"{title} (0=Back)",
        choices=names + [Choice("← Back", value="0", shortcut_key="0")],
        style=STYLE,
        use_shortcuts=True,
    ).ask()
    if ans is None:
        return "", None
    _questionary_done()
    if ans == "0":
        return "0", ""
    return ans, vals[names.index(ans)]


# ── (infra)structure ──────────────────────────────────────────────────────────


def run_local(exp: str, py_kwargs: str) -> None:
    py_file = REPO / EXPERIMENTS[exp]
    subprocess.run([str(REPO / ".venv" / "bin" / "python"), "-u", str(py_file)] + py_kwargs.split())


def run_slurm(exp: str, slurm_kwargs: str, py_kwargs: str, env: dict[str, str]) -> None:
    log_dir = REPO / "slurm"
    log_dir.mkdir(exist_ok=True)
    py_file = REPO / EXPERIMENTS[exp]
    cmd = (
        ["sbatch", f"--job-name={exp}", f"--output={log_dir}/%j.out"]
        + slurm_kwargs.split()
        + [env["MAIN_SH"], f"--env_path={REPO / '.venv'}", f"--python_path={py_file}", "--"]
        + py_kwargs.split()
    )
    subprocess.run(cmd)
    print("Job submitted!")


def preview_local(exp: str, py_kwargs: str) -> None:
    print(f"\npython -u {REPO / EXPERIMENTS[exp]} {py_kwargs}\n")


def preview_slurm(exp: str, slurm_kwargs: str, py_kwargs: str, env: dict[str, str]) -> None:
    log_dir = REPO / "slurm"
    py_file = REPO / EXPERIMENTS[exp]
    print("\nsbatch \\")
    print(f"    --output={log_dir}/%j.out \\")
    for kw in slurm_kwargs.split():
        print(f"    {kw} \\")
    print(f"    {env['MAIN_SH']} \\")
    print(f"    --env_path={REPO / '.venv'} --python_path={py_file} -- \\")
    print(f"    {py_kwargs}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def pick_job_config(env: dict[str, str]) -> tuple[str, str, str, list[tuple[str, str]]] | None:
    infra = ""
    spec_label = spec_val = dur_label = dur_val = ""
    step = JobConfigStep.INFRA

    while step <= JobConfigStep.EXPERIMENT:
        state: list[tuple[str, str]] = []
        if infra:
            state.append(("Infra", infra))
        if spec_label:
            state.append(("Spec", spec_label))
        if dur_label:
            state.append(("Duration", dur_label))

        if step == JobConfigStep.INFRA:
            _redraw(state)
            ans = questionary.select(
                "Infra (1/2/0)",
                choices=[
                    Choice("Slurm", value="Slurm", shortcut_key="1"),
                    Choice("Local", value="Local", shortcut_key="2"),
                    Choice("Exit", value="0", shortcut_key="0"),
                ],
                style=STYLE,
                use_shortcuts=True,
            ).ask()
            if ans is None:
                return None
            _questionary_done()
            if ans == "0":
                return None
            infra = ans
            step = JobConfigStep.SPEC

        elif step == JobConfigStep.SPEC:
            if infra != "Slurm":
                step = JobConfigStep.EXPERIMENT
                continue
            label, val = pick_preset_labeled("SPEC_", "Spec", env, state)
            if label == "0":
                infra = ""
                spec_label = spec_val = ""
                step = JobConfigStep.INFRA
                continue
            if val is None:
                return None
            spec_label, spec_val = label, val
            step = JobConfigStep.DURATION

        elif step == JobConfigStep.DURATION:
            label, val = pick_preset_labeled("DUR_", "Duration", env, state)
            if label == "0":
                spec_label = spec_val = ""
                step = JobConfigStep.SPEC
                continue
            if val is None:
                return None
            dur_label, dur_val = label, val
            step = JobConfigStep.EXPERIMENT

        else:  # step == JobConfigStep.EXPERIMENT
            _redraw(state)
            ans = questionary.select(
                "Experiment (0=Back)",
                choices=[
                    Choice(name, shortcut_key=SHORTCUT_KEYS[i])
                    for i, name in enumerate(EXPERIMENTS)
                ] + [Choice("← Back", value="0", shortcut_key="0")],
                style=STYLE,
                use_shortcuts=True,
            ).ask()
            if ans is None:
                return None
            _questionary_done()
            if ans == "0":
                dur_label = dur_val = ""
                step = JobConfigStep.DURATION if infra == "Slurm" else JobConfigStep.INFRA
                if infra != "Slurm":
                    infra = ""
            else:
                state.append(("Experiment", ans))
                slurm_kwargs = f"{spec_val} {dur_val}".strip()
                return infra, slurm_kwargs, ans, state

    return None


def main() -> None:
    env = load_env(ENV_FILE)

    while True:
        job = pick_job_config(env)
        if job is None:
            sys.exit(0)
        infra, slurm_kwargs, exp, state = job

        prev_fields = None
        while True:
            result = pick_kwargs(exp, state, prev_fields)
            if result is None:
                break
            py_kwargs, prev_fields = result

            _redraw(state)
            if infra == "Local":
                preview_local(exp, py_kwargs)
            else:
                preview_slurm(exp, slurm_kwargs, py_kwargs, env)
            confirm = questionary.select(
                "Submit?",
                choices=[
                    questionary.Choice("Submit & Finish", shortcut_key="1"),
                    questionary.Choice("Submit & Repeat", shortcut_key="2"),
                    questionary.Choice("Back to Edit", shortcut_key="0"),
                    questionary.Choice("Cancel & Exit", shortcut_key="q"),
                ],
                use_shortcuts=True,
                style=STYLE,
            ).ask()
            _questionary_done()

            if confirm is None or confirm == "Cancel & Exit":
                break
            if confirm == "Back to Edit":
                continue

            if infra == "Local":
                run_local(exp, py_kwargs)
            else:
                run_slurm(exp, slurm_kwargs, py_kwargs, env)

            if confirm == "Submit & Repeat":
                continue
            break

        questionary.press_any_key_to_continue().ask()


if __name__ == "__main__":
    main()
