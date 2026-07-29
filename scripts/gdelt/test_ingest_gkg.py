import json
import urllib.error

import pytest

import ingest_gkg
from ingest_gkg import (
    build_content,
    build_payload,
    detect_columns,
    iter_gkg_rows,
    looks_like_header,
    parse_gkg_date,
    post_remember,
    run,
    _normalize_locations_field,
    _DEFAULT_COLUMNS,
)

# type#fullname#countrycode#adm1code#lat#lon#featureid (V1-shaped block, as
# accepted by v2locations.parse_v2locations)
V1_LOCATIONS = "3#Baltimore, Maryland, United States#US#USMD#39.2904#-76.6122#4347778"

# True GKG 2.1 V2Locations block: adds ADM2Code before lat, CharOffset after featureid.
V2_LOCATIONS = "3#Baltimore, Maryland, United States#US#USMD#USMD005#39.2904#-76.6122#4347778#128"


def make_row(**overrides):
    row = list(_DEFAULT_COLUMNS.keys())  # 27 placeholder cells, overwritten below
    row = [""] * len(_DEFAULT_COLUMNS)
    row[_DEFAULT_COLUMNS["gkgrecordid"]] = overrides.get("gkgrecordid", "20150326000000-0")
    row[_DEFAULT_COLUMNS["date"]] = overrides.get("date", "20150326120000")
    row[_DEFAULT_COLUMNS["sourcecommonname"]] = overrides.get("sourcecommonname", "nytimes.com")
    row[_DEFAULT_COLUMNS["documentidentifier"]] = overrides.get(
        "documentidentifier", "http://nytimes.com/article1"
    )
    row[_DEFAULT_COLUMNS["v2locations"]] = overrides.get("v2locations", V2_LOCATIONS)
    row[_DEFAULT_COLUMNS["allnames"]] = overrides.get(
        "allnames", "Francis Scott Key Bridge,10;Baltimore,40"
    )
    return row


# =============================================================================
# _normalize_locations_field: the V1/V2 shape adapter
# =============================================================================
def test_normalize_drops_adm2_and_charoffset_from_true_v2_block():
    normalized = _normalize_locations_field(V2_LOCATIONS)
    assert normalized == V1_LOCATIONS


def test_normalize_passes_through_v1_shaped_block_unchanged():
    assert _normalize_locations_field(V1_LOCATIONS) == V1_LOCATIONS


def test_normalize_handles_multiple_blocks_and_empty_field():
    combined = V2_LOCATIONS + ";" + V1_LOCATIONS
    normalized = _normalize_locations_field(combined)
    assert normalized == V1_LOCATIONS + ";" + V1_LOCATIONS
    assert _normalize_locations_field("") == ""


# =============================================================================
# parse_gkg_date
# =============================================================================
def test_parse_gkg_date_full_precision():
    assert parse_gkg_date("20150326120000") == "2015-03-26T12:00:00Z"


def test_parse_gkg_date_day_only():
    assert parse_gkg_date("20150326") == "2015-03-26T00:00:00Z"


def test_parse_gkg_date_malformed_returns_none_not_fatal():
    assert parse_gkg_date("notadate") is None
    assert parse_gkg_date("") is None
    assert parse_gkg_date("20151399120000") is None  # bad month/day


# =============================================================================
# build_payload: the row -> payload transform (the core contract)
# =============================================================================
def test_build_payload_full_row_has_geo_and_timestamp():
    row = make_row()
    payload = build_payload(row, "gdelt-user")
    assert payload["user_id"] == "gdelt-user"
    assert payload["geo_location"] == [39.2904, -76.6122, 0.0]
    assert payload["created_at"] == "2015-03-26T12:00:00Z"
    assert "gdelt" in payload["tags"]
    assert "nytimes.com" in payload["tags"]
    assert payload["content"]  # non-empty, embeddable text
    assert "Baltimore" in payload["content"] or "Francis Scott Key Bridge" in payload["content"]


def test_build_payload_resolves_true_v2_locations_not_v1_shaped():
    # This is the field-shape regression: parsing the *true* V2Locations
    # block (with ADM2Code) must still yield the correct coordinates, not
    # silently misread ADM2Code as latitude and not silently drop the block.
    row = make_row(v2locations=V2_LOCATIONS)
    payload = build_payload(row, "u")
    assert payload["geo_location"] == [39.2904, -76.6122, 0.0]


def test_build_payload_missing_locations_still_ingests_text():
    row = make_row(v2locations="")
    payload = build_payload(row, "u")
    assert payload is not None
    assert "geo_location" not in payload
    assert payload["content"]


def test_build_payload_malformed_date_omits_created_at_not_fatal():
    row = make_row(date="garbage-date")
    payload = build_payload(row, "u")
    assert payload is not None
    assert "created_at" not in payload
    assert payload["content"]


def test_build_payload_out_of_range_coords_dropped_not_fatal():
    bad_block = "3#Nowhere#US#US#US#999.0#-76.6122#id#1"
    row = make_row(v2locations=bad_block)
    payload = build_payload(row, "u")
    assert payload is not None
    assert "geo_location" not in payload


def test_build_payload_blank_row_skipped():
    assert build_payload([], "u") is None
    assert build_payload(["", "", ""], "u") is None
    assert build_payload([""] * len(_DEFAULT_COLUMNS), "u") is None


def test_build_payload_truncated_row_no_crash_and_no_geo():
    row = ["20150326000000-0", "20150326120000", "1", "src.com", "http://src.com/z"]
    payload = build_payload(row, "u")
    assert payload is not None
    assert "geo_location" not in payload
    assert payload["content"]


def test_build_content_falls_back_when_nothing_but_record_id():
    row = [""] * len(_DEFAULT_COLUMNS)
    row[_DEFAULT_COLUMNS["gkgrecordid"]] = "20150326000000-9"
    content = build_content(row, _DEFAULT_COLUMNS, [])
    assert content == "GDELT GKG record 20150326000000-9"


# =============================================================================
# header / column detection
# =============================================================================
def test_looks_like_header_true_for_column_name_false_for_record_id():
    assert looks_like_header(["GKGRECORDID", "DATE"]) is True
    assert looks_like_header(["20150326000000-0", "20150326120000"]) is False
    assert looks_like_header([]) is False


def test_detect_columns_bigquery_header_reorders_correctly():
    header = ["GKGRECORDID", "DATE", "V2Locations", "AllNames"]  # deliberately partial/reordered-looking
    columns = detect_columns(header)
    assert columns["gkgrecordid"] == 0
    assert columns["date"] == 1
    assert columns["v2locations"] == 2
    assert columns["allnames"] == 3
    # untouched canonical columns still resolve to their default position
    assert columns["sourcecommonname"] == _DEFAULT_COLUMNS["sourcecommonname"]


def test_detect_columns_no_header_falls_back_to_canonical():
    assert detect_columns(None) == _DEFAULT_COLUMNS
    assert detect_columns(["20150326000000-0", "20150326120000"]) == _DEFAULT_COLUMNS


# =============================================================================
# iter_gkg_rows: deterministic file reading, header handling
# =============================================================================
def test_iter_gkg_rows_headerless_tsv_preserves_file_order(tmp_path):
    lines = ["\t".join(make_row(gkgrecordid=str(i))) for i in range(3)]
    path = tmp_path / "sample.tsv"
    path.write_text("\n".join(lines), encoding="utf-8")
    columns, rows = iter_gkg_rows(str(path))
    ids = [r[columns["gkgrecordid"]] for r in rows]
    assert ids == ["0", "1", "2"]


def test_iter_gkg_rows_with_header_is_consumed_not_yielded_as_data(tmp_path):
    header = ",".join(_CANONICAL_HEADER_LINE())
    data_row = ",".join(f'"{c}"' if "," in c else c for c in make_row(gkgrecordid="1"))
    path = tmp_path / "sample.csv"
    path.write_text(header + "\n" + data_row, encoding="utf-8")
    columns, rows = iter_gkg_rows(str(path))
    rows = list(rows)
    assert len(rows) == 1
    assert rows[0][columns["gkgrecordid"]] == "1"


def _CANONICAL_HEADER_LINE():
    return [
        "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
        "DocumentIdentifier", "V1Counts", "V2Counts", "V1Themes", "V2Themes",
        "V1Locations", "V2Locations", "V1Persons", "V2Persons", "V1Organizations",
        "V2Organizations", "V15Tone", "Dates", "GCAM", "SharingImage",
        "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
        "AllNames", "Amounts", "TranslationInfo", "Extras",
    ]


def test_iter_gkg_rows_empty_file(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    columns, rows = iter_gkg_rows(str(path))
    assert list(rows) == []


# =============================================================================
# post_remember: HTTP layer, stubbed (never calls a live server)
# =============================================================================
class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_remember_sends_json_body_to_correct_url(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return _FakeResponse(b'{"success": true, "id": "abc-123"}')

    monkeypatch.setattr(ingest_gkg.urllib.request, "urlopen", fake_urlopen)

    result = post_remember("http://localhost:9944", {"user_id": "u", "content": "c"})

    assert captured["url"] == "http://localhost:9944/api/remember"
    assert captured["body"] == {"user_id": "u", "content": "c"}
    assert result == {"success": True, "id": "abc-123"}


def test_post_remember_sets_bearer_header_when_api_key_given(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["auth"] = request.get_header("Authorization")
        return _FakeResponse(b"{}")

    monkeypatch.setattr(ingest_gkg.urllib.request, "urlopen", fake_urlopen)
    post_remember("http://localhost:9944", {"content": "c"}, api_key="secret-key")
    assert captured["auth"] == "Bearer secret-key"


# =============================================================================
# run(): end-to-end orchestration with stubbed HTTP
# =============================================================================
def test_run_dry_run_never_calls_http(monkeypatch, tmp_path):
    def fail_urlopen(*a, **k):
        raise AssertionError("dry-run must not perform HTTP calls")

    monkeypatch.setattr(ingest_gkg.urllib.request, "urlopen", fail_urlopen)

    lines = ["\t".join(make_row(gkgrecordid=str(i))) for i in range(3)]
    path = tmp_path / "sample.tsv"
    path.write_text("\n".join(lines), encoding="utf-8")

    summary = run(["--server", "http://x", "--user-id", "u", "--input", str(path), "--dry-run"])
    assert summary == {"parsed": 3, "skipped": 0, "posted": 0, "duplicates": 0, "errors": 0, "dry_run": True}


def test_run_respects_limit(monkeypatch, tmp_path):
    lines = ["\t".join(make_row(gkgrecordid=str(i))) for i in range(5)]
    path = tmp_path / "sample.tsv"
    path.write_text("\n".join(lines), encoding="utf-8")

    summary = run(["--server", "http://x", "--user-id", "u", "--input", str(path), "--dry-run", "--limit", "2"])
    assert summary["parsed"] == 2


def test_run_posts_and_counts_success(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b'{"success": true, "id": "x"}')

    monkeypatch.setattr(ingest_gkg.urllib.request, "urlopen", fake_urlopen)

    lines = ["\t".join(make_row(gkgrecordid=str(i))) for i in range(2)]
    path = tmp_path / "sample.tsv"
    path.write_text("\n".join(lines), encoding="utf-8")

    summary = run(["--server", "http://x", "--user-id", "u", "--input", str(path)])
    assert summary["posted"] == 2
    assert summary["errors"] == 0
    assert summary["duplicates"] == 0


def test_run_counts_duplicates_separately_from_errors(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 409, "Conflict", None, None)
        raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", None, None)

    monkeypatch.setattr(ingest_gkg.urllib.request, "urlopen", fake_urlopen)

    lines = ["\t".join(make_row(gkgrecordid=str(i))) for i in range(2)]
    path = tmp_path / "sample.tsv"
    path.write_text("\n".join(lines), encoding="utf-8")

    summary = run(["--server", "http://x", "--user-id", "u", "--input", str(path)])
    assert summary["posted"] == 0
    assert summary["duplicates"] == 1
    assert summary["errors"] == 1
    assert summary["error_samples"][0]["status"] == 500


def test_run_never_drops_a_malformed_row(monkeypatch, tmp_path):
    # A row with a blank V2Locations and a garbage date is still ingested --
    # only the coordinate and timestamp fields are dropped.
    good = make_row(gkgrecordid="0")
    malformed = make_row(gkgrecordid="1", v2locations="", date="not-a-date")
    blank = [""] * len(_DEFAULT_COLUMNS)

    path = tmp_path / "sample.tsv"
    path.write_text(
        "\n".join("\t".join(r) for r in (good, malformed, blank)), encoding="utf-8"
    )

    summary = run(["--server", "http://x", "--user-id", "u", "--input", str(path), "--dry-run"])
    assert summary["parsed"] == 2  # good + malformed-but-kept
    assert summary["skipped"] == 1  # only the truly blank row
