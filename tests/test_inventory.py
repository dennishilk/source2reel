from pathlib import Path
from source2reel.inventory import build_inventory


def test_document_chunking(tmp_path: Path):
    src=tmp_path/"src"; project=tmp_path/"project"; src.mkdir(); project.mkdir(); (project/"manifests").mkdir()
    (src/"README.md").write_text("\n".join(f"line {i}" for i in range(2500)))
    inv=build_inventory(src,project,chunk_chars=1200)
    docs=[e for e in inv["evidence"] if e["kind"]=="document"]
    assert len(docs)>1
    assert docs[0]["line_start"]==1
    assert docs[1]["line_start"] <= docs[0]["line_end"]
