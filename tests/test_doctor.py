from source2reel.doctor import Check, exit_code


def test_optional_voice_failure_does_not_fail_doctor():
    checks = [
        Check("CORE", "core", True, "ok", required=True),
        Check("VOICE (optional)", "kokoro-runtime", False, "not installed", required=False),
    ]
    assert exit_code(checks) == 0


def test_required_core_failure_still_fails_doctor():
    checks = [
        Check("CORE", "core", False, "missing", required=True),
        Check("VOICE (optional)", "kokoro-runtime", True, "ok", required=False),
    ]
    assert exit_code(checks) == 1
