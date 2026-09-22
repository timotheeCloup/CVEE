def test_build_params_daily_uses_publiee_depuis() -> None:
    from ft_client import _build_params

    params = _build_params(0, 150, 1, "NV1", 1, None, None)
    assert params["range"] == "0-149"
    assert params["niveauFormation"] == "NV1"
    assert params["publieeDepuis"] == 1
    assert "minCreationDate" not in params


def test_build_params_backfill_uses_iso_datetimes() -> None:
    from ft_client import _build_params

    params = _build_params(0, 150, 1, "NV1", 1, "2026-09-18", "2026-09-19")
    # France Travail only honours minCreationDate/maxCreationDate (ISO-8601).
    assert params["minCreationDate"] == "2026-09-18T00:00:00Z"
    assert params["maxCreationDate"] == "2026-09-19T23:59:59Z"
    assert "publieeDepuis" not in params
    assert "minDateCreation" not in params
    assert "maxDateCreation" not in params
