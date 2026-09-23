from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .providers import LLMProvider
from .util import json_dump



def _batch(entries, max_chars):
    batch=[]; n=0
    for e in entries:
        s=json.dumps(e,ensure_ascii=False)
        if batch and n+len(s)>max_chars:
            yield batch; batch=[]; n=0
        batch.append(e); n+=len(s)
    if batch: yield batch


def research(provider: LLMProvider, inventory: dict[str, Any], project_dir: Path, max_chars: int = 45000) -> dict[str, Any]:
    allfacts=[]; assets=[]
    for i,b in enumerate(_batch(inventory["evidence"],max_chars),1):
        payload={"batch":i,"evidence":b,"required_output":{"facts":[{"claim":"...","evidence_refs":["E0001"],"phase":"development|final|background|unknown","confidence":"high|medium|low"}],"assets":[{"evidence_ref":"E0002","purpose":"...","authentic_project_media":True}]}}
        system=(project_dir.parents[1]/"prompts"/"research.txt").read_text()
        r=provider.complete_json(system,json.dumps(payload,ensure_ascii=False))
        allfacts.extend(r.get("facts",[])); assets.extend(r.get("assets",[]))
    valid={e["ref"] for e in inventory["evidence"]}
    clean=[]
    for f in allfacts:
        refs=[r for r in f.get("evidence_refs",[]) if r in valid]
        if refs and f.get("claim"):
            f["evidence_refs"]=refs; clean.append(f)
    out={"version":1,"facts":clean,"assets":assets}
    json_dump(project_dir/"manifests"/"research.json",out)
    return out
