"""Minimal sanity smoke test for package imports."""


def test_package_import() -> None:
    """Test that the main package imports without error."""
    import sat_qkd_lab
    assert sat_qkd_lab is not None


def test_trivial() -> None:
    """Trivial assertion to verify test framework works."""
    assert True
