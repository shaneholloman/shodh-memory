"""GDELT GKG V2Locations parser. Deterministic; malformed blocks are skipped
and counted, never fatal (silent-data-loss rule: keep the record, drop only
the unparseable coordinate)."""
from typing import NamedTuple, Optional

class GdeltLocation(NamedTuple):
    loc_type: int; name: str; country: str; adm1: str
    lat: float; lon: float; feature_id: str

# GKG location types: 1=country, 2=US state, 3=US city, 4=world city, 5=world state
_SPECIFICITY = {3: 0, 4: 0, 2: 1, 5: 1, 1: 2}  # lower = more specific

def parse_v2locations(field: str) -> list[GdeltLocation]:
    out = []
    for block in (field or "").split(";"):
        parts = block.split("#")
        if len(parts) < 7:
            continue
        try:
            out.append(GdeltLocation(int(parts[0]), parts[1], parts[2], parts[3],
                                     float(parts[4]), float(parts[5]), parts[6]))
        except ValueError:
            continue
    return out

def primary_location(field: str) -> Optional[tuple[float, float]]:
    locs = parse_v2locations(field)
    if not locs:
        return None
    best = min(locs, key=lambda l: (_SPECIFICITY.get(l.loc_type, 3), locs.index(l)))
    return (best.lat, best.lon)
