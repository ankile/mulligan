import pytest

from mulligan.tools.lerobot_hub import retry_transient_hf_operation


def test_retry_transient_hf_operation_retries_ssl_error(monkeypatch) -> None:
    monkeypatch.setenv("MULLIGAN_HF_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("MULLIGAN_HF_RETRY_BASE_SLEEP_S", "0")
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("SSLError: SSLEOFError UNEXPECTED_EOF_WHILE_READING")
        return "ok"

    assert retry_transient_hf_operation("test", flaky) == "ok"
    assert calls == 3


def test_retry_transient_hf_operation_does_not_retry_contract_errors(monkeypatch) -> None:
    monkeypatch.setenv("MULLIGAN_HF_RETRY_ATTEMPTS", "3")
    calls = 0

    def broken():
        nonlocal calls
        calls += 1
        raise ValueError("missing required dataset column")

    with pytest.raises(ValueError, match="missing required dataset column"):
        retry_transient_hf_operation("test", broken)
    assert calls == 1
