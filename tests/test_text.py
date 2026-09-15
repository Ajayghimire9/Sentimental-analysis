from src.text import normalize_text


def test_normalize_text_removes_urls_and_mentions():
    result = normalize_text("Hello @ajay https://example.com!!!")
    assert "http" not in result
    assert "@ajay" not in result
    assert "hello" in result


def test_normalize_text_is_deterministic():
    text = "Great product!!! #awesome"
    assert normalize_text(text) == normalize_text(text)
