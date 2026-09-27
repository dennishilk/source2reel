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
    except OSError:
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
        except OSError: return ImageFont.load_default(size=size)

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


def render_evidence(scene, asset: Path, out: Path, root: Path, cfg, captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or "PROJECT EVIDENCE",cfg); F=t["frame"]; box=(72,154,1776,630 if captions_enabled else 800); px,py,nw,nh=_contain(im,asset,box)
    d.rectangle((px-2,py-2,px+nw+1,py+nh+1),outline=_rgb(F["line"]),width=2); _evidence_footer(d,t)
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_video_shell(scene,out,root,cfg,captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or "PROJECT VIDEO",cfg); F=t["frame"]
    d.rectangle((72,154,1848,784 if captions_enabled else 954),outline=_rgb(F["line"]),width=2)
    _evidence_footer(d,t,"AUTHENTIC PROJECT VIDEO // FIXED FRAME")
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_title(scene,out,root,cfg):
    im,d,t=_base(root,scene.get("title") or "SECTION",cfg); F=t["frame"]; hf=_font(t["typography"]["heading_font"],82); mf=_font(t["typography"]["mono_font"],28)
    title=scene.get("title") or "SECTION"; bb=d.textbbox((0,0),title,font=hf); d.text(((1920-(bb[2]-bb[0]))//2,400),title,font=hf,fill=_rgb(F["text"]))
    note=scene.get("notes",""); bb=d.textbbox((0,0),note,font=mf); d.text(((1920-(bb[2]-bb[0]))//2,530),note,font=mf,fill=_rgb(F["muted"]))
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_code(scene,asset,out,root,cfg,evidence_item=None,captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or "CODE",cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],27)
    if asset and asset.exists():
        lines=asset.read_text(errors="replace").splitlines()
        if evidence_item and evidence_item.get("line_start"):
            a=max(0,int(evidence_item["line_start"])-1); b=min(len(lines),int(evidence_item.get("line_end",a+26))); lines=lines[a:b]
        else: lines=lines[:26]
    else: lines=(scene.get("diagram",{}).get("code","")).splitlines()
    y=160
    for line in lines[:20 if captions_enabled else 26]: d.text((96,y),line[:110],font=mf,fill=_rgb(F["text"])); y+=31
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def _node_labels(scene):
    data=scene.get("diagram") or {}; nodes=data.get("nodes") or data.get("steps") or []
    if nodes and isinstance(nodes[0],dict): return [str(x.get("label") or x.get("name") or x.get("id")) for x in nodes]
    return [str(x) for x in nodes]


def render_diagram(scene,out,root,cfg,captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or scene["type"],cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],27); labels=_node_labels(scene)
    if len(labels)<2: raise ValueError(f"{scene['id']}: explicit diagram nodes required")
    n=len(labels); top=210; bottom=730 if captions_enabled else 840; x=960; gap=(bottom-top)/(n-1); coords=[]; h=min(86,gap-8)
    for i,label in enumerate(labels):
        y=top+i*gap; coords.append((x,y)); w=1300; d.rounded_rectangle((x-w/2,y-h/2,x+w/2,y+h/2),radius=10,outline=_rgb(F["line"]),width=3); font=_fit_text(d,label,t["typography"]["mono_font"],27,w-30); bb=d.textbbox((0,0),label,font=font); d.text((x-(bb[2]-bb[0])/2,y-(bb[3]-bb[1])/2),label,font=font,fill=_rgb(F["text"]))
        if i: d.line((x,coords[i-1][1]+h/2,x,y-h/2),fill=_rgb(F["accent"]),width=3)
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_graph(scene,out,root,cfg,captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or "GRAPH",cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],24); data=scene.get("diagram") or {}; pts=data.get("points") or []
    xy=[]
    for i,p in enumerate(pts):
        if isinstance(p,dict): xy.append((float(p["x"]),float(p["y"]),str(p.get("label",""))))
        elif isinstance(p,(list,tuple)) and len(p)==2: xy.append((float(p[0]),float(p[1]),""))
    if len(xy)<2: raise ValueError(f"{scene['id']}: GRAPH requires explicit points")
    left,top,right,bottom=180,180,1740,730 if captions_enabled else 870; d.line((left,bottom,right,bottom),fill=_rgb(F["line"]),width=2); d.line((left,top,left,bottom),fill=_rgb(F["line"]),width=2)
    xs=[p[0] for p in xy]; ys=[p[1] for p in xy]; xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); xr=max(xmax-xmin,1e-9); yr=max(ymax-ymin,1e-9)
    px=[]
    for x,y,label in xy:
        sx=left+(x-xmin)/xr*(right-left); sy=bottom-(y-ymin)/yr*(bottom-top); px.append((sx,sy)); d.ellipse((sx-5,sy-5,sx+5,sy+5),fill=_rgb(F["accent"]));
        if label: d.text((sx+8,sy-28),label,font=mf,fill=_rgb(F["muted"]))
    if len(px)>1: d.line(px,fill=_rgb(F["accent"]),width=4)
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_summary(scene,out,root,cfg,captions_enabled=False):
    im,d,t=_base(root,scene.get("title") or scene["type"],cfg); F=t["frame"]; mf=_font(t["typography"]["mono_font"],31); items=scene.get("annotations") or []; y=215
    for item in items[:7]:
        text=item if isinstance(item,str) else str(item.get("text") or item); d.text((180,y),"— "+text,font=mf,fill=_rgb(F["text"])); y+=78 if captions_enabled else 92
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def _fit_text(d, value, font_path, preferred, width):
    size=preferred
    while size>26 and d.textbbox((0,0),value,font=_font(font_path,size))[2]>width: size-=2
    font=_font(font_path,size)
    if d.textbbox((0,0),value,font=font)[2]>width:
        raise ValueError(f"Presentation text too wide; split into lines: {value[:60]}")
    return font


def render_outro(out: Path, root: Path, cfg: dict, presentation: dict):
    im,d,t=_base(root,"",cfg); F=t["frame"]; T=t["typography"]
    data=presentation.get("outro") or {}
    if not data: raise ValueError("OUTRO requires presentation.outro metadata (headline and links)")
    left=112; bright=_rgb(F["text"]); accent=_rgb(F["accent"])
    eyebrow=data.get("eyebrow",""); d.text((left,155),eyebrow,font=_font(T["mono_font"],30),fill=accent)
    lines=data.get("headline",[])
    if isinstance(lines,str): lines=[lines]
    if not lines or len(lines)>3: raise ValueError("OUTRO headline requires 1–3 lines")
    y=230
    for line in lines:
        font=_fit_text(d,line,T["heading_font"],90,1696)
        d.text((left,y),line,font=font,fill=bright); y+=105
    links=data.get("links",[])
    if not 1<=len(links)<=3: raise ValueError("OUTRO requires 1–3 links")
    y=max(510,y+20)
    for link in links:
        d.text((left,y),link["label"],font=_font(T["mono_font"],29),fill=accent); y+=43
        url_lines=link["url"] if isinstance(link["url"],list) else [link["url"]]
        for line in url_lines:
            font=_fit_text(d,line,T["heading_font"],49,1696)
            d.text((left,y),line,font=font,fill=bright); y+=57
        y+=32
    footer=data.get("footer","")
    if footer: d.text((left,942),footer,font=_fit_text(d,footer,T["mono_font"],25,1696),fill=_rgb(F["muted"]))
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out)


def render_scene(scene: dict[str,Any], asset: Path|None, out: Path, root: Path, cfg: dict[str,Any], evidence_item: dict[str,Any]|None=None, captions_enabled=False, presentation=None):
    typ=scene["type"]
    if typ in {"HERO","PROJECT_EVIDENCE","TERMINAL_EVIDENCE","HARDWARE_EVIDENCE"} and asset: return render_evidence(scene,asset,out,root,cfg,captions_enabled)
    if typ=="OUTRO": return render_outro(out,root,cfg,presentation or {})
    if typ=="SECTION_TITLE": return render_title(scene,out,root,cfg)
    if typ=="CODE": return render_code(scene,asset,out,root,cfg,evidence_item,captions_enabled)
    if typ=="GRAPH": return render_graph(scene,out,root,cfg,captions_enabled)
    if typ in {"ARCHITECTURE_DIAGRAM","DATA_FLOW","TIMELINE"}: return render_diagram(scene,out,root,cfg,captions_enabled)
    if typ=="SUMMARY": return render_summary(scene,out,root,cfg,captions_enabled)
    return render_title(scene,out,root,cfg)
