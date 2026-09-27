"""Shared operation verbs for explicit source prose and workflow signals."""
from __future__ import annotations

import re


FLOW_ACTION = re.compile(
    r"\b(?:normaliz(?:e|es|ed|ing)|enqueu(?:e|es|ed|ing)|"
    r"updat(?:e|es|ed|ing)|resolv(?:e|es|ed|ing)|"
    r"correlat(?:e|es|ed|ing)|classif(?:y|ies|ied|ying)|"
    r"writ(?:e|es|ten|ing)|stor(?:e|es|ed|ing)|"
    r"captur(?:e|es|ed|ing)|aggregat(?:e|es|ed|ing)|"
    r"convert(?:s|ed|ing)?|produc(?:e|es|ed|ing)|"
    r"emit(?:s|ted|ting)?|read(?:s|ing)?)\b", re.I,
)
