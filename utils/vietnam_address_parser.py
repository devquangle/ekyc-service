# -*- coding: utf-8 -*-
"""
vietnam_address_parser.py
=========================
Standalone module for parsing, normalizing, and hierarchically decomposing
Vietnamese address strings into 4 administrative levels:

  Level 1 - province : Tỉnh / Thành phố trực thuộc Trung ương
  Level 2 - district : Quận / Huyện / Thị xã / Thành phố thuộc tỉnh
  Level 3 - ward     : Phường / Xã / Thị trấn
  Level 4 - street   : Số nhà, tên đường, thôn/ấp/tổ/khóm/khu phố/ngõ/hẻm

Data Source
-----------
DiaGioiHanhChinhVN  (kenzouno1)
URL: https://raw.githubusercontent.com/kenzouno1/DiaGioiHanhChinhVN/master/data.json

The data is auto-downloaded and cached locally as `vietnam_administrative_data.json`
next to this module on first run.

Dependencies
------------
  pip install rapidfuzz pydantic>=2.0 ftfy

Usage
-----
    from vietnam_address_parser import parse_address, VietnamAddressParser

    result = parse_address("Số 5 Nguyễn Trãi, Phường Khương Trung, Quận Thanh Xuân, Hà Nội")
    print(result.province)   # Thành phố Hà Nội
    print(result.district)   # Quận Thanh Xuân
    print(result.ward)       # Phường Khương Trung
    print(result.street)     # Số 5 Nguyễn Trãi
    print(result.confidence) # 0.97
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── optional fast fuzzy matching ───────────────────────────────────────────
try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# ─── Pydantic V2 ─────────────────────────────────────────────────────────────
try:
    from pydantic import BaseModel, Field as PydField, ConfigDict
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_URL = "https://raw.githubusercontent.com/kenzouno1/DiaGioiHanhChinhVN/master/data.json"
CACHE_FILENAME = "vietnam_administrative_data.json"

# Default fuzzy-match threshold (0-100 scale for rapidfuzz)
DEFAULT_THRESHOLD = 78

# Administrative prefix words to strip before core-name matching
# Ordered longest-first so greedy prefix stripping works correctly
_PROVINCE_PREFIXES = [
    "thành phố", "tp.", "tp",
    "tỉnh",
]
_DISTRICT_PREFIXES = [
    "thành phố", "tp.", "tp",
    "thị xã", "tx.", "tx",
    "quận", "q.", "q",
    "huyện", "h.", "h",
]
_WARD_PREFIXES = [
    "thị trấn", "tt.", "tt",
    "phường", "p.", "p",
    "xã", "x.",
]

# Normalised-key → standard Vietnamese prefix label
_PREFIX_LABEL: Dict[str, str] = {
    "thanh pho": "Thành phố", "tp": "Thành phố",
    "tinh": "Tỉnh",
    "quan": "Quận", "q": "Quận",
    "huyen": "Huyện", "h": "Huyện",
    "thi xa": "Thị xã", "tx": "Thị xã",
    "phuong": "Phường", "p": "Phường",
    "xa": "Xã",
    "thi tran": "Thị trấn", "tt": "Thị trấn",
    "ap": "Ấp", "thon": "Thôn", "to": "Tổ",
    "khom": "Khóm", "khu pho": "Khu phố", "ban": "Bản",
}

# OCR → Vietnamese diacritic corrections  (Latin umlauts mis-recognised)
_OCR_GLYPH_MAP: Dict[str, str] = {
    'ä': 'â', 'Ä': 'Â',
    'ë': 'ê', 'Ë': 'Ê',
    'ï': 'i', 'Ï': 'I',
    'ö': 'ô', 'Ö': 'Ô',
    'ü': 'ư', 'Ü': 'Ư',
    'ÿ': 'y', 'Ÿ': 'Y',
    'å': 'a', 'Å': 'A',
    'ø': 'o', 'Ø': 'O',
    'ç': 'c', 'Ç': 'C',
    'ñ': 'n', 'Ñ': 'N',
    'æ': 'ae', 'Æ': 'AE',
}


# ─────────────────────────────────────────────────────────────────────────────
# Output Schema
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_PYDANTIC:
    class ParsedAddress(BaseModel):
        """Pydantic V2 output schema for a parsed Vietnamese address."""
        model_config = ConfigDict(populate_by_name=True)

        original_address: str = PydField(
            default="",
            description="The raw address string as supplied by the caller.",
        )
        province: Optional[str] = PydField(
            default=None,
            description="Full standardised province/city name (e.g. 'Thành phố Hà Nội').",
        )
        district: Optional[str] = PydField(
            default=None,
            description="Full standardised district name (e.g. 'Quận Ba Đình').",
        )
        ward: Optional[str] = PydField(
            default=None,
            description="Full standardised ward name (e.g. 'Phường Phúc Xá').",
        )
        street: Optional[str] = PydField(
            default=None,
            description="Remaining street/hamlet/detail text.",
        )
        province_id: Optional[str] = PydField(
            default=None,
            description="Province GSO code, e.g. '01'.",
        )
        district_id: Optional[str] = PydField(
            default=None,
            description="District GSO code, e.g. '001'.",
        )
        ward_id: Optional[str] = PydField(
            default=None,
            description="Ward GSO code, e.g. '00001'.",
        )
        full_normalized_address: str = PydField(
            default="",
            description="Canonical formatted address, comma-separated from street to province.",
        )
        confidence: float = PydField(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Average fuzzy-match confidence of matched components.",
        )

else:  # fallback plain dataclass when pydantic not available
    @dataclass
    class ParsedAddress:  # type: ignore[no-redef]
        original_address: str = ""
        province: Optional[str] = None
        district: Optional[str] = None
        ward: Optional[str] = None
        street: Optional[str] = None
        province_id: Optional[str] = None
        district_id: Optional[str] = None
        ward_id: Optional[str] = None
        full_normalized_address: str = ""
        confidence: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal Admin-Unit Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _AdminUnit:
    """Lightweight record for a single administrative division."""
    unit_id: str          # GSO code string ("01", "001", "00001")
    full_name: str        # "Thành phố Hà Nội" — as stored in dataset
    core_name: str        # "Hà Nội"           — prefix stripped
    norm_key: str         # "ha noi"            — unaccented lowercase for O(1) lookup
    level: int            # 1=province, 2=district, 3=ward
    parent_id: Optional[str] = None
    children: List["_AdminUnit"] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Text Utilities (standalone, no external utils/ dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _remove_accents(text: str) -> str:
    """Remove Vietnamese diacritics → pure ASCII lowercase."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    return ascii_bytes.decode("ascii")


def _norm_key(text: str) -> str:
    """Canonical lookup key: unaccented, lowercase, alphanum-only."""
    s = _remove_accents(_nfc(text)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fix_ocr_glyphs(text: str) -> str:
    """Map OCR-confused Latin diacritics to Vietnamese equivalents."""
    # Compound diphthong repairs first
    text = (
        text.replace("ưö", "ươ").replace("uö", "ươ")
            .replace("ƯÖ", "ƯƠ").replace("UÖ", "ƯƠ")
    )
    return "".join(_OCR_GLYPH_MAP.get(c, c) for c in text)


def _fix_mojibake(text: str) -> str:
    """Best-effort Mojibake repair using ftfy when available."""
    try:
        import ftfy
        fixed = ftfy.fix_text(text)
        # Keep if Vietnamese char ratio improves or stays equal
        return fixed if fixed else text
    except Exception:
        return text


def _preprocess(raw: str) -> str:
    """Full normalization pipeline for a raw address string."""
    t = _nfc(raw)
    t = _fix_mojibake(t)
    t = _fix_ocr_glyphs(t)
    # Unify separators → comma
    t = t.replace(";", ",").replace("|", ",").replace(" - ", ", ")
    # Collapse whitespace / newlines
    t = re.sub(r"[\t\r\n]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _strip_prefix_from_segment(segment: str, prefix_list: List[str]) -> Tuple[str, str]:
    """
    Strips the longest matching administrative prefix from *segment*.
    Returns (prefix_label, core_name).
    """
    s = segment.strip()
    s_lower = s.lower()
    best_prefix = ""
    best_core = s
    for pfx in sorted(prefix_list, key=len, reverse=True):
        if s_lower.startswith(pfx):
            rest = s[len(pfx):].lstrip(". ").strip()
            if rest:
                best_prefix = pfx
                best_core = rest
                break
    return best_prefix, best_core


# ─────────────────────────────────────────────────────────────────────────────
# Data Manager — Download / Cache / Build Index
# ─────────────────────────────────────────────────────────────────────────────

class _DataManager:
    """
    Downloads and caches the DiaGioiHanhChinhVN JSON dataset.
    Builds a 3-level hierarchy tree and a set of fast O(1) look-up indexes.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._cache_path: Path = cache_path or (Path(__file__).parent / CACHE_FILENAME)
        self.provinces: List[_AdminUnit] = []
        # {province_id -> {district_id -> district_unit}}
        self._prov_map: Dict[str, _AdminUnit] = {}
        # Flat key-sets for each level
        self._prov_key_index: Dict[str, _AdminUnit] = {}    # norm_key → unit
        self._dist_key_index: Dict[str, List[_AdminUnit]] = {}  # norm_key → [units]
        self._ward_key_index: Dict[str, List[_AdminUnit]] = {}  # norm_key → [units]
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        raw_data = self._fetch_data()
        self._build_index(raw_data)
        self._loaded = True

    # ------------------------------------------------------------------
    def _fetch_data(self) -> List[Dict[str, Any]]:
        """Return parsed JSON — from local cache if available, else download."""
        if self._cache_path.is_file():
            log.debug("Loading admin data from cache: %s", self._cache_path)
            with open(self._cache_path, encoding="utf-8") as fh:
                return json.load(fh)

        log.info("Downloading admin data from %s …", DATA_URL)
        try:
            req = urllib.request.Request(
                DATA_URL,
                headers={"User-Agent": "vietnam-address-parser/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
            data = json.loads(content.decode("utf-8"))
            # Persist cache
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            log.info("Cached admin data → %s", self._cache_path)
            return data
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download admin data from {DATA_URL}: {exc}\n"
                "Please download manually and save as vietnam_administrative_data.json "
                "next to vietnam_address_parser.py"
            ) from exc

    # ------------------------------------------------------------------
    def _strip_prov_prefix(self, name: str) -> str:
        _, core = _strip_prefix_from_segment(name, _PROVINCE_PREFIXES)
        return core

    def _strip_dist_prefix(self, name: str) -> str:
        _, core = _strip_prefix_from_segment(name, _DISTRICT_PREFIXES)
        return core

    def _strip_ward_prefix(self, name: str) -> str:
        _, core = _strip_prefix_from_segment(name, _WARD_PREFIXES)
        return core

    # ------------------------------------------------------------------
    def _build_index(self, data: List[Dict[str, Any]]) -> None:
        for p_raw in data:
            p_unit = _AdminUnit(
                unit_id=p_raw["Id"],
                full_name=p_raw["Name"],
                core_name=self._strip_prov_prefix(p_raw["Name"]),
                norm_key=_norm_key(self._strip_prov_prefix(p_raw["Name"])),
                level=1,
            )
            # Also index the full name without prefix (e.g. "Ho Chi Minh")
            self.provinces.append(p_unit)
            self._prov_map[p_unit.unit_id] = p_unit
            self._prov_key_index[p_unit.norm_key] = p_unit
            # Extra alias: full_name norm key
            full_key = _norm_key(p_raw["Name"])
            if full_key != p_unit.norm_key:
                self._prov_key_index.setdefault(full_key, p_unit)

            for d_raw in p_raw.get("Districts", []):
                d_unit = _AdminUnit(
                    unit_id=d_raw["Id"],
                    full_name=d_raw["Name"],
                    core_name=self._strip_dist_prefix(d_raw["Name"]),
                    norm_key=_norm_key(self._strip_dist_prefix(d_raw["Name"])),
                    level=2,
                    parent_id=p_unit.unit_id,
                )
                p_unit.children.append(d_unit)
                # Full name alias
                full_d_key = _norm_key(d_raw["Name"])
                for idx_key in {d_unit.norm_key, full_d_key}:
                    self._dist_key_index.setdefault(idx_key, []).append(d_unit)

                for w_raw in d_raw.get("Wards", []):
                    w_unit = _AdminUnit(
                        unit_id=w_raw["Id"],
                        full_name=w_raw["Name"],
                        core_name=self._strip_ward_prefix(w_raw["Name"]),
                        norm_key=_norm_key(self._strip_ward_prefix(w_raw["Name"])),
                        level=3,
                        parent_id=d_unit.unit_id,
                    )
                    d_unit.children.append(w_unit)
                    full_w_key = _norm_key(w_raw["Name"])
                    for idx_key in {w_unit.norm_key, full_w_key}:
                        self._ward_key_index.setdefault(idx_key, []).append(w_unit)

        log.debug(
            "Index built: %d provinces, %d unique district keys, %d unique ward keys",
            len(self.provinces),
            len(self._dist_key_index),
            len(self._ward_key_index),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy Matching Core
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_threshold(core_name: str, base: int = DEFAULT_THRESHOLD) -> int:
    """
    Short names (≤ 4 chars unaccented) need a higher threshold to avoid
    false positives. Long names can afford a slightly lower floor.
    """
    norm = _remove_accents(core_name).replace(" ", "")
    length = len(norm)
    if length <= 3:
        return max(base, 92)
    if length <= 5:
        return max(base, 86)
    if length >= 12:
        return max(60, base - 5)
    return base


def _numeric_unit_regex(text: str) -> Optional[Tuple[str, int]]:
    """
    Detect numeric administrative units:
      'Quận 1', 'Q.3', 'Phường 15', 'P15', 'Huyện 9'
    Returns (canonical label, number) or None.
    """
    pattern = re.compile(
        r"^\s*(?P<pfx>quận|q\.?|phường|p\.?|huyện|h\.?|thị\s*xã|tx\.?|xã|x\.?)\s*(?P<num>\d{1,2})\s*$",
        re.IGNORECASE,
    )
    m = pattern.match(text)
    if m:
        pfx_raw = m.group("pfx").lower().rstrip(".")
        num = int(m.group("num"))
        pfx_map = {
            "quận": "Quận", "q": "Quận",
            "phường": "Phường", "p": "Phường",
            "huyện": "Huyện", "h": "Huyện",
            "thị xã": "Thị xã", "tx": "Thị xã",
            "xã": "Xã", "x": "Xã",
        }
        label = pfx_map.get(pfx_raw, pfx_raw.capitalize())
        return label, num
    return None


def _fuzzy_score(query_key: str, candidate_key: str) -> float:
    """
    Composite RapidFuzz score (0–100). Falls back to simple character overlap.
    """
    if not _HAS_RAPIDFUZZ:
        # primitive overlap ratio
        q, c = set(query_key.split()), set(candidate_key.split())
        if not q or not c:
            return 0.0
        return 100.0 * len(q & c) / max(len(q), len(c))

    r_sort = fuzz.token_sort_ratio(query_key, candidate_key)
    r_ratio = fuzz.ratio(query_key, candidate_key)
    r_partial = fuzz.partial_ratio(query_key, candidate_key)
    return r_sort * 0.50 + r_ratio * 0.30 + r_partial * 0.20


def _best_match(
    query: str,
    candidates: List[_AdminUnit],
    threshold: int = DEFAULT_THRESHOLD,
) -> Optional[Tuple[_AdminUnit, float]]:
    """
    Find the best-matching _AdminUnit for *query* from *candidates*.
    Returns (unit, score_0_to_1) or None if below threshold.
    """
    if not candidates:
        return None

    q_key = _norm_key(query)
    if not q_key:
        return None

    # 1. Exact key hit
    for cand in candidates:
        if cand.norm_key == q_key:
            return cand, 1.0

    # 2. Numeric unit shortcut — match by prefix + number
    num_info = _numeric_unit_regex(query)
    if num_info:
        label, num = num_info
        target_key = _norm_key(f"{label} {num}")
        for cand in candidates:
            if cand.norm_key == target_key:
                return cand, 1.0
        # Fuzzy among numeric candidates only
        numeric_cands = [c for c in candidates if re.search(r'\d', c.core_name)]
        if numeric_cands:
            candidates = numeric_cands

    # 3. Fuzzy
    best_unit: Optional[_AdminUnit] = None
    best_score = -1.0
    for cand in candidates:
        score = _fuzzy_score(q_key, cand.norm_key)
        # Also score against full_name key
        full_score = _fuzzy_score(_norm_key(query), _norm_key(cand.full_name))
        score = max(score, full_score)
        if score > best_score:
            best_score = score
            best_unit = cand

    thr = _adaptive_threshold(query, threshold)
    if best_score >= thr and best_unit is not None:
        return best_unit, best_score / 100.0
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Parser
# ─────────────────────────────────────────────────────────────────────────────

class VietnamAddressParser:
    """
    Parses a Vietnamese address string into 4 hierarchical levels.

    Parameters
    ----------
    cache_path : Path, optional
        Where to store the downloaded JSON cache.
        Defaults to `vietnam_administrative_data.json` alongside this module.
    threshold : int
        Base fuzzy-match threshold (0–100). Default 78.
    auto_load : bool
        Automatically download & index data on first parse.  Default True.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        threshold: int = DEFAULT_THRESHOLD,
        auto_load: bool = True,
    ) -> None:
        self._db = _DataManager(cache_path)
        self._threshold = threshold
        if auto_load:
            self._db.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Explicitly (re-)load the administrative database."""
        self._db.load()

    def parse(self, raw_address: str) -> "ParsedAddress":
        """
        Parse a raw Vietnamese address into a structured ParsedAddress.

        The algorithm works *right-to-left*:
          1. Identify province from rightmost segment.
          2. Identify district from next segment (restricted to province).
          3. Identify ward from next segment (restricted to district).
          4. Everything remaining is the street/detail.
        """
        if not self._db._loaded:
            self._db.load()

        if not raw_address or not raw_address.strip():
            return ParsedAddress(original_address=raw_address or "")

        preprocessed = _preprocess(raw_address)

        # ── Segment splitting ─────────────────────────────────────────────
        segments = self._split_segments(preprocessed)

        # ── Right-to-left matching ────────────────────────────────────────
        prov_unit: Optional[_AdminUnit] = None
        dist_unit: Optional[_AdminUnit] = None
        ward_unit: Optional[_AdminUnit] = None
        scores: List[float] = []

        # STEP 1 – Province
        prov_unit, scores, segments = self._match_level_from_right(
            segments, scores, level="province"
        )

        # STEP 2 – District  (scoped to matched province)
        dist_unit, scores, segments = self._match_level_from_right(
            segments, scores, level="district", parent=prov_unit
        )

        # STEP 3 – Ward  (scoped to matched district, else province)
        ward_unit, scores, segments = self._match_level_from_right(
            segments, scores, level="ward",
            parent=dist_unit or prov_unit
        )

        # STEP 4 – Street/Detail
        street_raw = ", ".join(segments).strip()
        street = self._clean_street(street_raw) or None

        # ── Assemble output ───────────────────────────────────────────────
        prov_name = prov_unit.full_name if prov_unit else None
        dist_name = dist_unit.full_name if dist_unit else None
        ward_name = ward_unit.full_name if ward_unit else None

        parts = [p for p in [street, ward_name, dist_name, prov_name] if p]
        full_addr = ", ".join(parts)

        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        return ParsedAddress(
            original_address=raw_address,
            province=prov_name,
            district=dist_name,
            ward=ward_name,
            street=street,
            province_id=prov_unit.unit_id if prov_unit else None,
            district_id=dist_unit.unit_id if dist_unit else None,
            ward_id=ward_unit.unit_id if ward_unit else None,
            full_normalized_address=full_addr,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _split_segments(self, text: str) -> List[str]:
        """
        Split preprocessed address into segments.
        Comma-delimited → each token is a candidate admin segment.
        No commas → treat as single sliding-window string.
        """
        if "," in text:
            parts = [s.strip() for s in text.split(",") if s.strip()]
            return parts
        return [text.strip()]

    # ------------------------------------------------------------------
    # Right-to-Left Level Matcher
    # ------------------------------------------------------------------

    def _match_level_from_right(
        self,
        segments: List[str],
        scores: List[float],
        level: str,
        parent: Optional[_AdminUnit] = None,
    ) -> Tuple[Optional[_AdminUnit], List[float], List[str]]:
        """
        Try to match *level* (province/district/ward) from the rightmost
        segment(s) of *segments*.  Returns (matched_unit, updated_scores, remaining_segments).
        """
        if not segments:
            return None, scores, segments

        # Build candidate pool
        candidates = self._get_candidates(level, parent)
        if not candidates:
            return None, scores, segments

        # Try rightmost segment first
        for idx in range(len(segments) - 1, -1, -1):
            seg = segments[idx]
            result = self._try_match_segment(seg, candidates, level)
            if result:
                unit, score = result
                scores = scores + [score]
                remaining = segments[:idx] + segments[idx + 1:]
                return unit, scores, remaining

            # If single non-comma segment: try sliding window over words
            if len(segments) == 1:
                words = seg.split()
                for wlen in range(min(6, len(words)), 1, -1):
                    cand_str = " ".join(words[-wlen:])
                    result = self._try_match_segment(cand_str, candidates, level)
                    if result:
                        unit, score = result
                        scores = scores + [score]
                        remaining_text = " ".join(words[: len(words) - wlen]).strip()
                        remaining = [remaining_text] if remaining_text else []
                        return unit, scores, remaining
                break  # single segment exhausted

        return None, scores, segments

    # ------------------------------------------------------------------

    def _try_match_segment(
        self,
        segment: str,
        candidates: List[_AdminUnit],
        level: str,
    ) -> Optional[Tuple[_AdminUnit, float]]:
        """Strip prefix variants of *segment* and attempt fuzzy match."""
        prefixes = {
            "province": _PROVINCE_PREFIXES,
            "district": _DISTRICT_PREFIXES,
            "ward": _WARD_PREFIXES,
        }[level]

        # Always try with and without prefix
        variants: List[str] = [segment]
        _, core = _strip_prefix_from_segment(segment, prefixes)
        if core != segment:
            variants.append(core)
        # Also try normalised short forms (e.g. "HCM" → try as-is)
        variants.append(_norm_key(segment))

        for v in variants:
            result = _best_match(v, candidates, self._threshold)
            if result:
                return result
        return None

    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        level: str,
        parent: Optional[_AdminUnit],
    ) -> List[_AdminUnit]:
        """
        Return the candidate pool for *level*, restricted to *parent*'s children
        when a parent is already matched.
        """
        if level == "province":
            return self._db.provinces

        if level == "district":
            if parent and parent.level == 1:
                return parent.children
            # No province match → search all districts
            seen: Set[str] = set()
            all_d: List[_AdminUnit] = []
            for units in self._db._dist_key_index.values():
                for u in units:
                    if u.unit_id not in seen:
                        all_d.append(u)
                        seen.add(u.unit_id)
            return all_d

        if level == "ward":
            if parent and parent.level == 2:
                return parent.children
            if parent and parent.level == 1:
                # Flatten all wards under province
                wards: List[_AdminUnit] = []
                for dist in parent.children:
                    wards.extend(dist.children)
                return wards
            # No match context → all wards (expensive but correct)
            seen2: Set[str] = set()
            all_w: List[_AdminUnit] = []
            for units in self._db._ward_key_index.values():
                for u in units:
                    if u.unit_id not in seen2:
                        all_w.append(u)
                        seen2.add(u.unit_id)
            return all_w

        return []

    # ------------------------------------------------------------------
    # Street Cleanup
    # ------------------------------------------------------------------

    _STREET_STRIP_RE = re.compile(r"^[\s,.\-_/]+|[\s,.\-_/]+$")

    def _clean_street(self, text: str) -> str:
        """Remove leading/trailing noise from the street segment."""
        cleaned = self._STREET_STRIP_RE.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + convenience function
# ─────────────────────────────────────────────────────────────────────────────

_default_parser: Optional[VietnamAddressParser] = None


def _get_default_parser() -> VietnamAddressParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = VietnamAddressParser()
    return _default_parser


def parse_address(
    raw_address: str,
    threshold: int = DEFAULT_THRESHOLD,
) -> "ParsedAddress":
    """
    Module-level convenience wrapper.  Uses a cached singleton parser.

    Parameters
    ----------
    raw_address : str
        Raw Vietnamese address (from OCR, form input, etc.)
    threshold : int
        Fuzzy match threshold 0–100. Default 78.

    Returns
    -------
    ParsedAddress
        Structured result with province, district, ward, street, and confidence.

    Examples
    --------
    >>> result = parse_address("Số 5 Nguyễn Trãi, Phường Khương Trung, Quận Thanh Xuân, Hà Nội")
    >>> result.province
    'Thành phố Hà Nội'
    """
    parser = _get_default_parser()
    parser._threshold = threshold
    return parser.parse(raw_address)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    test_cases = [
        # Standard comma-delimited with prefix labels
        "Số 5 Nguyễn Trãi, Phường Khương Trung, Quận Thanh Xuân, Thành phố Hà Nội",
        # Without accent (OCR output)
        "123 nguyen hue, p. ben nghe, q1, hcm",
        # Unaccented, no commas
        "To 5 Phuong Hoang Van Thu Quan Hoang Mai Ha Noi",
        # Short province name – Huế
        "Phú Hội, Thành phố Huế, Tỉnh Thừa Thiên Huế",
        # Highland – Ea H'leo
        "Thị trấn Ea Drăng, Huyện Ea H'leo, Tỉnh Đắk Lắk",
        # Southern hamlet + numeric ward
        "Ấp Tây, Xã Tân Bình, Huyện Châu Thành, Tỉnh Đồng Tháp",
        # Numeric district
        "Số 45/2 Tân Sơn, Phường 15, Quận Tân Bình, TP. Hồ Chí Minh",
        # Mojibake / OCR glyph confusion
        "Äp Tay, Xa Tan Binh, Huyen Chau Thanh, Tinh Dong Thap",
        # Province only
        "Cần Thơ",
        # Full Hanoi address
        "Số 1 Hoàng Diệu, Phường Quán Thánh, Quận Ba Đình, Hà Nội",
    ]

    parser = VietnamAddressParser()

    LINE = "=" * 78
    print(LINE)
    print("VIETNAM ADDRESS PARSER  –  Test Suite")
    print(f"  Database : {DATA_URL}")
    print(f"  Threshold: {DEFAULT_THRESHOLD}")
    print(LINE)

    for i, raw in enumerate(test_cases, 1):
        res = parser.parse(raw)
        print(f"\n[{i:02d}] Input    : {raw}")
        print(f"     Province : {res.province or '—':<35} id={res.province_id}")
        print(f"     District : {res.district or '—':<35} id={res.district_id}")
        print(f"     Ward     : {res.ward or '—':<35} id={res.ward_id}")
        print(f"     Street   : {res.street or '—'}")
        print(f"     Canonical: {res.full_normalized_address}")
        print(f"     Confidence: {res.confidence:.2%}")

    print(f"\n{LINE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _demo()
