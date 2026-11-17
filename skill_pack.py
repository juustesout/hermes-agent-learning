"""
skill_pack.py — export / import shareable skills.

A skill is workflow.json + optional cases.jsonl + README.
No policy snapshots and no raw episode payloads (privacy).

Usage:
    python3 skill_pack.py export --task "Scrape X" --out ./skills/scrape-x
    python3 skill_pack.py import ./skills/scrape-x
    python3 skill_pack.py validate ./skills/scrape-x
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import env_setup  # noqa: F401

from workflow_dag import WorkflowDAG
from workflow_synthesizer import WorkflowSynthesizer, _slugify

SCHEMA_VERSION = "1.0.0"
SKILL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillPackError(ValueError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SkillPackError(msg)


def dag_to_workflow_json(
    dag: WorkflowDAG,
    *,
    skill_id: Optional[str] = None,
    skill_version: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    meta = dict(dag.metadata or {})
    sid = skill_id or meta.get("skill_id") or _slugify(dag.id) or dag.id
    sver = skill_version or meta.get("skill_version") or "1.0.0"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "skill_id": sid,
        "skill_version": sver,
        "name": dag.name,
        "description": description,
        "template": meta.get("template"),
        "workflow": dag.to_dict(),
    }
    validate_workflow_payload(payload)
    return payload


def workflow_json_to_dag(payload: Dict[str, Any]) -> WorkflowDAG:
    validate_workflow_payload(payload)
    return WorkflowDAG.from_dict(payload["workflow"])


def validate_workflow_payload(payload: Dict[str, Any]) -> None:
    _require(isinstance(payload, dict), "workflow.json must be an object")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION}",
    )
    skill_id = payload.get("skill_id")
    _require(isinstance(skill_id, str) and SKILL_ID_RE.match(skill_id),
             "skill_id must be a kebab-case slug")
    _require(isinstance(payload.get("skill_version"), str)
             and payload["skill_version"],
             "skill_version is required")
    wf = payload.get("workflow")
    _require(isinstance(wf, dict), "workflow is required")
    _require(isinstance(wf.get("id"), str) and wf["id"], "workflow.id required")
    _require(isinstance(wf.get("name"), str), "workflow.name required")
    nodes = wf.get("nodes")
    _require(isinstance(nodes, list) and nodes, "workflow.nodes must be non-empty")
    node_ids = set()
    for node in nodes:
        _require(isinstance(node, dict), "node must be an object")
        _require(isinstance(node.get("id"), str) and node["id"], "node.id required")
        _require(isinstance(node.get("task_id"), str) and node["task_id"],
                 "node.task_id required")
        _require(isinstance(node.get("action_ids"), list) and node["action_ids"],
                 "node.action_ids must be non-empty")
        node_ids.add(node["id"])
    edges = wf.get("edges", [])
    _require(isinstance(edges, list), "workflow.edges must be a list")
    for edge in edges:
        _require(isinstance(edge, dict), "edge must be an object")
        _require(edge.get("source") in node_ids, f"unknown edge source {edge.get('source')!r}")
        _require(edge.get("target") in node_ids, f"unknown edge target {edge.get('target')!r}")


def validate_cases(cases: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        _require(isinstance(case, dict), f"cases.jsonl line {i}: not an object")
        _require("name" in case and "correct_action" in case,
                 f"cases.jsonl line {i}: name and correct_action required")
        banned = {"episode", "policy", "user_input", "conversation_history"}
        _require(not (banned & set(case)),
                 f"cases.jsonl line {i}: raw policy/episode fields are not allowed")
        out.append(case)
    return out


def write_skill_pack(
    dest: Path,
    payload: Dict[str, Any],
    *,
    cases: Optional[List[Dict[str, Any]]] = None,
    readme: Optional[str] = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "workflow.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if cases is not None:
        validate_cases(cases)
        with (dest / "cases.jsonl").open("w", encoding="utf-8") as fh:
            for case in cases:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    if readme is None:
        readme = (
            f"# {payload.get('name') or payload['skill_id']}\n\n"
            f"skill_id: `{payload['skill_id']}`  \n"
            f"version: `{payload['skill_version']}`  \n"
            f"template: `{payload.get('template') or ''}`\n\n"
            "This pack contains `workflow.json` (DAG + stable task ids). "
            "It does not include policy snapshots or raw episodes.\n"
        )
    (dest / "README.md").write_text(readme, encoding="utf-8")
    return dest


def read_skill_pack(src: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    wf_path = src / "workflow.json" if src.is_dir() else src
    _require(wf_path.exists(), f"missing workflow.json at {wf_path}")
    payload = json.loads(wf_path.read_text(encoding="utf-8"))
    validate_workflow_payload(payload)
    cases: List[Dict[str, Any]] = []
    cases_path = wf_path.parent / "cases.jsonl"
    if cases_path.exists():
        raw = []
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                raw.append(json.loads(line))
        cases = validate_cases(raw)
    return payload, cases


def export_from_task(
    task: str,
    dest: Path,
    *,
    template: Optional[str] = None,
    skill_id: Optional[str] = None,
    skill_version: str = "1.0.0",
    description: str = "",
    cases: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    syn = WorkflowSynthesizer()
    dag = syn.synthesize(
        task=task,
        template=template,
        skill_id=skill_id,
        skill_version=skill_version,
    )
    payload = dag_to_workflow_json(
        dag,
        skill_id=skill_id or dag.metadata.get("skill_id"),
        skill_version=skill_version,
        description=description,
    )
    return write_skill_pack(dest, payload, cases=cases)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export / import skill packs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("export", help="Synthesize a DAG and write a skill pack")
    p_ex.add_argument("--task", required=True)
    p_ex.add_argument("--out", required=True)
    p_ex.add_argument("--template", default=None)
    p_ex.add_argument("--skill-id", default=None)
    p_ex.add_argument("--skill-version", default="1.0.0")
    p_ex.add_argument("--description", default="")

    p_im = sub.add_parser("import", help="Load workflow.json into a WorkflowDAG")
    p_im.add_argument("path")
    p_im.add_argument("--json", action="store_true")

    p_va = sub.add_parser("validate", help="Validate a skill pack")
    p_va.add_argument("path")

    args = parser.parse_args()
    try:
        if args.cmd == "export":
            dest = export_from_task(
                args.task,
                Path(args.out),
                template=args.template,
                skill_id=args.skill_id,
                skill_version=args.skill_version,
                description=args.description,
            )
            print(f"Wrote skill pack → {dest}")
            return 0
        if args.cmd == "import":
            payload, cases = read_skill_pack(Path(args.path))
            dag = workflow_json_to_dag(payload)
            if args.json:
                print(json.dumps({
                    "skill_id": payload["skill_id"],
                    "skill_version": payload["skill_version"],
                    "dag_id": dag.id,
                    "nodes": list(dag.nodes),
                    "n_cases": len(cases),
                }, indent=2))
            else:
                print(f"{payload['skill_id']} v{payload['skill_version']}")
                print(f"DAG {dag.id}: {' → '.join(dag.topological_sort())}")
                print(f"cases: {len(cases)}")
            return 0
        payload, cases = read_skill_pack(Path(args.path))
        print(
            f"ok {payload['skill_id']} v{payload['skill_version']} "
            f"nodes={len(payload['workflow']['nodes'])} cases={len(cases)}"
        )
        return 0
    except SkillPackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
