# STYLE_GUIDE

This document defines the default **Dennis Explainer** profile shipped with Source2Reel. Source2Reel itself is profile-driven; the reusable engine must not assume this branding.

## Series identity
- 1920×1080, 16:9, 30 fps
- near-black charcoal engineering-documentary base
- restrained 48 px engineering grid
- cool cyan instrumentation lines only; no neon overload
- Inter Display for headings when installed; DejaVu Sans Mono for instrumentation metadata
- full-frame composition: no small floating presentation card

## Permanent scene vocabulary
- HERO
- PROJECT_EVIDENCE
- TERMINAL_EVIDENCE
- HARDWARE_EVIDENCE
- ARCHITECTURE_DIAGRAM
- DATA_FLOW
- TIMELINE
- CODE
- GRAPH
- SECTION_TITLE
- SUMMARY
- OUTRO

The local AI chooses and populates templates. It does not invent new series layouts during ordinary episode production.

## Evidence rule — permanent
Real photographs and screenshots are documentary evidence. While visible they remain static:
- POSITION = CONSTANT
- SCALE = CONSTANT
- CROP = CONSTANT
- no Ken Burns
- no zoom / pan / parallax / animated reframing

Evidence uses `contain` whenever practical and remains large enough to read. Scene duration follows narration, not an arbitrary edit clock.

Authentic project video clips may play inside a fixed evidence frame. Their frame position and scale remain fixed and source audio is muted by default under the narration stem.

Cuts and restrained fades are allowed. Animation is reserved for diagrams, flows, timelines, graphs and explanatory highlights.

## Readability
Terminal screenshots, code and hardware displays must remain readable for the narration that discusses them. If one frame cannot remain legible, split the explanation into multiple evidence scenes rather than adding camera motion.

## Captions and endcard

Static evidence, terminal, code and diagram scenes use deterministic captions
derived from the approved narration WAV duration. At most two large lines are
shown in the lower information strip. The evidence frame ends above that
strip; neither the evidence nor the captions move during their display.
Moving project footage and the endcard omit captions by default. Scene-level
configuration can override this when editorially needed.

The OUTRO is a fixed full-frame card: a short series/episode eyebrow, large
project headline, dominant readable project links, and an optional compact
footer. Its copy comes from per-episode presentation metadata, not renderer
constants. Keep the card static through the last narration and padding.
