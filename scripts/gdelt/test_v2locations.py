from v2locations import parse_v2locations, primary_location

# GKG V2Locations: ';'-separated blocks, '#'-separated fields:
# type#fullname#countrycode#adm1code#lat#lon#featureid
SAMPLE = "1#United States#US#US#39.828175#-98.5795#US;3#Baltimore, Maryland, United States#US#USMD#39.2904#-76.6122#4347778"

def test_parse_two_blocks():
    locs = parse_v2locations(SAMPLE)
    assert len(locs) == 2
    assert locs[1].loc_type == 3 and abs(locs[1].lat - 39.2904) < 1e-6

def test_primary_prefers_most_specific():
    # type 3 (US city) and 4 (world city) outrank 1 (country); first wins within a rank
    assert primary_location(SAMPLE) == (39.2904, -76.6122)

def test_malformed_blocks_skipped_not_fatal():
    assert parse_v2locations("garbage#novalues") == []
    assert primary_location("") is None
    # partial corruption: good block still parses
    locs = parse_v2locations("bad##block;3#Baltimore#US#USMD#39.2904#-76.6122#4347778")
    assert len(locs) == 1

def test_non_numeric_coords_skipped():
    assert parse_v2locations("3#X#US#USMD#notanumber#-76.6122#id") == []
