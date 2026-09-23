from source2reel.tts_engine import apply_pronunciations

def test_replacement():
    assert apply_pronunciations("CP-9951",{"pronunciation":{"CP-9951":"C P ninety-nine fifty-one"}})=="C P ninety-nine fifty-one"
