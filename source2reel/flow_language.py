"""Shared operation verbs for explicit source prose and workflow signals."""
from __future__ import annotations

import re


FLOW_ACTION = re.compile(
    r"\b(?:(?P<normalize>normaliz(?:e|es|ed|ing))|"
    r"(?P<enqueue>enqueu(?:e|es|ed|ing))|"
    r"(?P<update>updat(?:e|es|ed|ing))|"
    r"(?P<resolve>resolv(?:e|es|ed|ing))|"
    r"(?P<correlate>correlat(?:e|es|ed|ing))|"
    r"(?P<classify>classif(?:y|ies|ied|ying))|"
    r"(?P<write>writ(?:e|es|ten|ing))|"
    r"(?P<store>stor(?:e|es|ed|ing))|"
    r"(?P<capture>captur(?:e|es|ed|ing))|"
    r"(?P<aggregate>aggregat(?:e|es|ed|ing))|"
    r"(?P<convert>convert(?:s|ed|ing)?)|"
    r"(?P<produce>produc(?:e|es|ed|ing))|"
    r"(?P<emit>emit(?:s|ted|ting)?)|"
    r"(?P<read>read(?:s|ing)?))\b", re.I,
)


def distinct_flow_actions(text: str) -> set[str]:
    """Count operation kinds, so inflections and repetitions count only once."""
    return {match.lastgroup for match in FLOW_ACTION.finditer(text)
            if match.lastgroup is not None}
