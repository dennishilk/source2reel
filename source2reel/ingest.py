from __future__ import annotations
import html.parser, os, shutil, subprocess, urllib.parse, urllib.request
from pathlib import Path
from typing import Any
from .util import json_dump, sha256_file, slugify

DOC_EXTS = {".md", ".txt", ".rst", ".html", ".htm", ".json", ".yaml", ".yml", ".toml", ".c", ".h", ".cpp", ".hpp", ".py", ".sh", ".js", ".ts", ".css"}
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".mp4", ".mov", ".mkv", ".webm"}
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}


class _PageParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.images=[]; self.text=[]; self._skip=0
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag in {"script","style","noscript"}: self._skip += 1
        if tag == "a" and a.get("href"): self.links.append(a["href"])
        if tag in {"img","source"}:
            src=a.get("src") or a.get("srcset")
            if src: self.images.append(src.split()[0])
    def handle_endtag(self, tag):
        if tag in {"script","style","noscript"} and self._skip: self._skip -= 1
    def handle_data(self, data):
        if not self._skip:
            s=" ".join(data.split())
            if s: self.text.append(s)


def _download(url: str, path: Path, timeout=30):
    req=urllib.request.Request(url, headers={"User-Agent":"Source2Reel/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        path.write_bytes(r.read())


def _ingest_website(url: str, dest: Path, max_pages: int = 12) -> Path:
    web=dest/"web"; web.mkdir(parents=True, exist_ok=True)
    parsed0=urllib.parse.urlparse(url); origin=(parsed0.scheme, parsed0.netloc)
    queue=[url]; seen=set(); pages=[]; media_urls=[]
    while queue and len(pages) < max_pages:
        cur=queue.pop(0)
        if cur in seen: continue
        seen.add(cur)
        try:
            req=urllib.request.Request(cur, headers={"User-Agent":"Source2Reel/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                ctype=r.headers.get_content_type(); body=r.read()
            if ctype != "text/html": continue
            html=body.decode("utf-8",errors="replace"); p=_PageParser(); p.feed(html)
            idx=len(pages); html_path=web/f"page-{idx:03d}.html"; html_path.write_text(html)
            txt_path=web/f"page-{idx:03d}.txt"; txt_path.write_text("\n".join(p.text))
            pages.append({"url":cur,"html":str(html_path.relative_to(dest)),"text":str(txt_path.relative_to(dest))})
            for x in p.images: media_urls.append(urllib.parse.urljoin(cur,x))
            for x in p.links:
                nxt=urllib.parse.urljoin(cur,x).split("#",1)[0]
                q=urllib.parse.urlparse(nxt)
                if (q.scheme,q.netloc)==origin and nxt not in seen and (q.path.endswith("/") or Path(q.path).suffix.lower() in {"", ".html", ".htm", ".md"}):
                    queue.append(nxt)
        except Exception as e:
            pages.append({"url":cur,"error":str(e)})
    media=dest/"web-media"; media.mkdir(exist_ok=True)
    downloaded=[]
    for u in dict.fromkeys(media_urls):
        q=urllib.parse.urlparse(u); ext=Path(q.path).suffix.lower()
        if ext not in MEDIA_EXTS: continue
        name=slugify(Path(q.path).stem)+ext
        target=media/name
        if target.exists(): continue
        try:
            _download(u,target); downloaded.append({"url":u,"path":str(target.relative_to(dest))})
        except Exception: pass
    json_dump(dest/"website-manifest.json", {"root_url":url,"pages":pages,"media":downloaded})
    return dest


def _scan_local(root: Path) -> list[Path]:
    files=[]
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if any(part in SKIP_DIRS for part in p.parts): continue
        if p.suffix.lower() in DOC_EXTS|MEDIA_EXTS:
            files.append(p)
    return files


def ingest(source: str, project_dir: Path, max_pages: int = 12, namespace: str | None = None) -> Path:
    sources=project_dir/"sources"
    if namespace: sources=sources/namespace
    sources.mkdir(parents=True,exist_ok=True)
    if source.startswith(("http://","https://")):
        u=urllib.parse.urlparse(source)
        if u.netloc.lower() in {"github.com","www.github.com"}:
            bits=[x for x in u.path.strip("/").split("/") if x]
            if len(bits)>=2:
                repo_url=f"https://github.com/{bits[0]}/{bits[1].removesuffix('.git')}.git"
                repo=sources/"repo"
                if repo.exists(): shutil.rmtree(repo)
                subprocess.run(["git","clone","--depth=1",repo_url,str(repo)],check=True)
                commit=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
                json_dump(sources/"source.json", {"kind":"github","source":source,"repo_url":repo_url,"commit":commit,"local_path":"repo"})
                return repo
        result=_ingest_website(source,sources,max_pages=max_pages)
        json_dump(sources/"source.json", {"kind":"website","source":source,"local_path":"."})
        return result
    src=Path(source).expanduser().resolve()
    if not src.exists(): raise FileNotFoundError(source)
    manifest=[]
    for p in _scan_local(src):
        try: rel=p.relative_to(src)
        except ValueError: rel=Path(p.name)
        manifest.append({"path":str(p),"relative":str(rel),"size":p.stat().st_size,"sha256":sha256_file(p)})
    json_dump(sources/"local-source-manifest.json", {"root":str(src),"files":manifest})
    json_dump(sources/"source.json", {"kind":"local","source":str(src),"local_path":str(src)})
    return src
