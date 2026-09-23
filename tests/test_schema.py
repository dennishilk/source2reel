from source2reel.schema import validate_episode

def test_minimal_episode():
    ep={"version":1,"title":"X","scenes":[{"id":"s1","type":"PROJECT_EVIDENCE","narration":"Hello.","evidence_refs":["E0001"],"asset_ref":"E0001"}]}
    validate_episode(ep,{"E0001"})
