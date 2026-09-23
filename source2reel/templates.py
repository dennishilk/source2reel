from __future__ import annotations
from pathlib import Path
from typing import Any
import tomllib
from PIL import Image, ImageDraw, ImageFont
from .config import profile_paths, series_label


def _rgb(x):
    x=x.lstrip("#")
    return tuple(int(x[i:i+2],16) for i in (0,2,4))

def _font(path,size):
    try: return ImageFont.truetype(path,size)
    except Exception: return ImageFont.load_default()

def _theme(root,cfg):
    theme_path,_=profile_paths(root,cfg)
    return tomllib.loads(theme_path.read_text())


def _base(root: Path, title: str, cfg: dict[str,Any]):
    t=_theme(root,cfg); F=t["frame"]; T=t["typography"]; W,H=F["width"],F["height"]
    im=Image.new("RGB",(W,H),_rgb(F["background"])); d=ImageDraw.Draw(im)
    for x in range(0,W,48): d.line((x,0,x,H),fill=_rgb(F["grid"]),width=1)
    for y in range(0,H,48): d.line((0,y,W,y),fill=_rgb(F["grid"]),width=1)
    d.line((72,110,W-72,110),fill=_rgb(F["line"]),width=1); d.line((72,988,W-72,988),fill=_rgb(F["line"]),width=1)
    hf=_font(T["heading_font"],T["heading_size"]); mf=_font(T["mono_font"],T["meta_size"])
    d.text((72,30),series_label(cfg),font=mf,fill=_rgb(F["accent"]))
    bb=d.textbbox((0,0),title,font=hf); d.text(((W-(bb[2]-bb[0]))//2,25),title,font=hf,fill=_rgb(F["text"]))
    return im,d,t


def _contain(im: Image.Image, asset: Path, box):
    x,y,w,h=box; src=Image.open(asset).convert("RGB"); scale=min(w/src.width,h/src.height); nw,nh=round(src.width*scale),round(src.height*scale); src=src.resize((nw,nh),Image.Resampling.LANCZOS); px=x+(w-nw)//2; py=y+(h-nh)//2; im.paste(src,(px,py)); return px,py,nw,nh


def _evidence_footer(d,t,label="AUTHENTIC PROJECT EVIDENCE // STATIC"):
    F=t["frame"]; mf=_font(t["typography"]["mono_font"],t["typography"]["meta_size"])
    d.text((72,1018),label,font=mf,fill=_rgb(F["muted"]))
    right="POSITION LOCK // SCALE LOCK // CROP LOCK"; rb=d.textbbox((0,0),right,font=mf); d.text((1920-72-(rb[2]-rb[0]),1018),right,font=mf,fill=_rgb(F["muted"]))


def render_evidence(scene, asset: Path, out: Path, root: Path, cfg):
    im,d,t=_base(root,scene.get("title") or "PROJECT EVIDENCE",cfg); F=t["frame"]; box=(72,154,1776,800); px,py,nw,nh=_contain(im,asset,box)
    d.rectangle((px-2,py-2,px+nw+1,py+nh+1),outline=_rgb(F["line"]),width=2); _evidence_footer(d,t)
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_video_shell(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or "PROJECT VIDEO",cfg); F=t["frame"]
    d.rectangle((72,154,1848,954),outline=_rgb(F["line"]),width=2)
    _evidence_footer(d,t,"AUTHENTIC PROJECT VIDEO // FIXED FRAME")
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_title(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or "SECTION",cfg); F=t["frame"]; hf=_font(t["typography"]["heading_font"],82); mf=_font(t["typography"]["mono_font"],28)
    title=scene.get("title") or "SECTION"; bb=d.textbbox((0,0),title,font=hf); d.text(((1920-(bb[2]-bb[0]))//2,400),title,font=hf,fill=_rgb(F["text"]))
    note=scene.get("notes",""); bb=d.textbbox((0,0),note,font=mf); d.text(((1920-(bb[2]-bb[0]))//2,530),note,font=mf,fill=_rgb(F["muted"]))
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_code(scene,asset,out,root,cfg,evidence_item=None):
    im,d,t=_base(root,scene.get("title") or "CODE",cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],27)
    if asset and asset.exists():
        lines=asset.read_text(errors="replace").splitlines()
        if evidence_item and evidence_item.get("line_start"):
            a=max(0,int(evidence_item["line_start"])-1); b=min(len(lines),int(evidence_item.get("line_end",a+26))); lines=lines[a:b]
        else: lines=lines[:26]
    else: lines=(scene.get("diagram",{}).get("code","")).splitlines()
    y=160
    for line in lines[:26]: d.text((96,y),line[:110],font=mf,fill=_rgb(F["text"])); y+=31
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def _node_labels(scene):
    data=scene.get("diagram") or {}; nodes=data.get("nodes") or data.get("steps") or []
    if nodes and isinstance(nodes[0],dict): return [str(x.get("label") or x.get("name") or x.get("id")) for x in nodes]
    return [str(x) for x in nodes]


def render_diagram(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or scene["type"],cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],27); labels=_node_labels(scene) or ["SOURCE","PROCESS","OUTPUT"]
    n=len(labels); top=220; bottom=840; x=960; gap=(bottom-top)/(max(1,n-1)); coords=[]
    for i,label in enumerate(labels):
        y=top+i*gap; coords.append((x,y)); w=780; h=86; d.rounded_rectangle((x-w/2,y-h/2,x+w/2,y+h/2),radius=10,outline=_rgb(F["line"]),width=3); bb=d.textbbox((0,0),label,font=mf); d.text((x-(bb[2]-bb[0])/2,y-(bb[3]-bb[1])/2),label,font=mf,fill=_rgb(F["text"]))
        if i: d.line((x,coords[i-1][1]+43,x,y-43),fill=_rgb(F["accent"]),width=3); d.polygon([(x,y-43),(x-9,y-60),(x+9,y-60)],fill=_rgb(F["accent"]))
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_graph(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or "GRAPH",cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],24); data=scene.get("diagram") or {}; pts=data.get("points") or []
    xy=[]
    for i,p in enumerate(pts):
        if isinstance(p,dict): xy.append((float(p.get("x",i)),float(p.get("y",0)),str(p.get("label",""))))
        elif isinstance(p,(list,tuple)) and len(p)>=2: xy.append((float(p[0]),float(p[1]),""))
    if not xy: xy=[(0,0,""),(1,1,""),(2,0.6,"")]
    left,top,right,bottom=180,180,1740,870; d.line((left,bottom,right,bottom),fill=_rgb(F["line"]),width=2); d.line((left,top,left,bottom),fill=_rgb(F["line"]),width=2)
    xs=[p[0] for p in xy]; ys=[p[1] for p in xy]; xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); xr=max(xmax-xmin,1e-9); yr=max(ymax-ymin,1e-9)
    px=[]
    for x,y,label in xy:
        sx=left+(x-xmin)/xr*(right-left); sy=bottom-(y-ymin)/yr*(bottom-top); px.append((sx,sy)); d.ellipse((sx-5,sy-5,sx+5,sy+5),fill=_rgb(F["accent"]));
        if label: d.text((sx+8,sy-28),label,font=mf,fill=_rgb(F["muted"]))
    if len(px)>1: d.line(px,fill=_rgb(F["accent"]),width=4)
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_summary(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or scene["type"],cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],31); items=scene.get("annotations") or []; y=245
    for item in items[:7]:
        text=item if isinstance(item,str) else str(item.get("text") or item); d.text((180,y),"— "+text,font=mf,fill=_rgb(F["text"])); y+=92
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_scene(scene: dict[str,Any], asset: Path|None, out: Path, root: Path, cfg: dict[str,Any], evidence_item: dict[str,Any]|None=None):
    typ=scene["type"]
    if typ in {"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE"} and asset: return render_evidence(scene,asset,out,root,cfg)
    if typ in {"SECTION_TITLE","OUTRO"}: return render_title(scene,out,root,cfg)
    if typ=="CODE": return render_code(scene,asset,out,root,cfg,evidence_item)
    if typ=="GRAPH": return render_graph(scene,out,root,cfg)
    if typ in {"ARCHITECTURE_DIAGRAM","DATA_FLOW","TIMELINE"}: return render_diagram(scene,out,root,cfg)
    if typ=="SUMMARY": return render_summary(scene,out,root,cfg)
    return render_title(scene,out,root,cfg)
