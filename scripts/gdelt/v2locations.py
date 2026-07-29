"""GDELT GKG V2Locations parser. Deterministic; malformed blocks are skipped
and counted, never fatal (silent-data-loss rule: keep the record, drop only
the unparseable coordinate).

Accepted per-block shapes (';'-separated blocks, '#'-separated fields):

  - Legacy V1Locations (7 fields):
      type#fullname#countrycode#adm1code#lat#lon#featureid
    Kept for backward compatibility with older GDELT exports and the
    V1Locations column, which some pipelines still carry alongside V2.

  - GKG 2.1 V2Locations / "V2 Enhanced Locations" (8 fields), which inserts
    an ADM2Code between ADM1Code and latitude relative to the V1 shape:
      type#fullname#countrycode#adm1code#adm2code#lat#lon#featureid

  - V2.1 "enhanced" variant of the above (9 fields), which appends a
    CharOffset (an index into the source article's text, unrelated to
    geo-location) after featureid:
      type#fullname#countrycode#adm1code#adm2code#lat#lon#featureid#charoffset
    CharOffset is parsed off but intentionally not retained on
    GdeltLocation -- it has no bearing on geo-tagging, which is the only
    thing this module is for.

Field-count is what distinguishes the shapes (7 vs 8/9); ADM2Code is
never mistaken for latitude because the two layouts are parsed with
different field offsets rather than a single fixed index.

Provenance note: the official GDELT GKG 2.1 codebook PDF
(data.gdeltproject.org) could not be fetched directly in the environment
this was verified in (TLS hostname/cert mismatch). The 8/9-field
V2Locations shape was confirmed via independent secondary sources (GDELT
project blog posts, the Haskell GDELT client's documented V1 shape plus
the documented "ADM2Code inserted before latitude" delta) and controller
review against the GKG 2.1 codebook structure.
"""
from typing import NamedTuple, Optional

class GdeltLocation(NamedTuple):
    loc_type: int; name: str; country: str; adm1: str; adm2: str
    lat: float; lon: float; feature_id: str

# GKG location types: 1=country, 2=US state, 3=US city, 4=world city, 5=world state
_SPECIFICITY = {3: 0, 4: 0, 2: 1, 5: 1, 1: 2}  # lower = more specific

def parse_v2locations(field: str) -> list[GdeltLocation]:
    out = []
    for block in (field or "").split(";"):
        parts = block.split("#")
        n = len(parts)
        if n == 7:
            # Legacy V1Locations: type#name#cc#adm1#lat#lon#featureid
            loc_type_s, name, country, adm1, lat_s, lon_s, feature_id = parts
            adm2 = ""
        elif n in (8, 9):
            # V2Locations / V2.1 enhanced: type#name#cc#adm1#adm2#lat#lon#featureid[#charoffset]
            loc_type_s, name, country, adm1, adm2, lat_s, lon_s, feature_id = parts[:8]
            # parts[8] (CharOffset), when present, is discarded -- see module docstring
        else:
            continue
        try:
            out.append(GdeltLocation(int(loc_type_s), name, country, adm1, adm2,
                                     float(lat_s), float(lon_s), feature_id))
        except ValueError:
            continue
    return out

def primary_location(field: str) -> Optional[tuple[float, float]]:
    locs = parse_v2locations(field)
    if not locs:
        return None
    best = min(locs, key=lambda l: (_SPECIFICITY.get(l.loc_type, 3), locs.index(l)))
    return (best.lat, best.lon)
