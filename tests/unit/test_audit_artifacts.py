"""Smoke checks para artefatos da auditoria (Task A)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_contract_document_present():
    doc = PROJECT_ROOT / "docs" / "legacy_contract.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "GET /" in text
    assert "/info" in text
    assert "/config" in text
