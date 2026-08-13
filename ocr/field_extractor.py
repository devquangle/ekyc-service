import re
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from ocr.detector import OCRText
from ocr.layout_parser import LayoutLine, LayoutParser
from ocr.label_matcher import LabelMatcher, FIELD_LABELS
from ocr.normalizer import (
    normalize_unicode,
    normalize_gender,
    parse_date,
    normalize_identity_number,
    normalize_full_name,
    normalize_address,
)
from utils.text_utils import remove_vietnamese_accents
from utils.logger import logger


class ExtractedField(BaseModel):
    fieldName: str
    value: Optional[str] = None
    rawText: Optional[str] = None
    keyword: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[List[List[float]]] = None


class FieldExtractor:
    """
    Data-Driven Generic Field Extractor.
    Uses spatial bounding box layout parsing, LabelMatcher fuzzy label recognition,
    and line boundary isolation without hardcoding personal identity data or OCR typos.
    """

    def __init__(self):
        self.layout_parser = LayoutParser()
        self.label_matcher = LabelMatcher(min_similarity_threshold=0.70)

    def extract_all_fields(
        self, tokens: List[OCRText]
    ) -> Dict[str, ExtractedField]:
        if not tokens:
            return {}

        layout_lines = self.layout_parser.group_tokens_into_lines(tokens)
        keyword_matches = self._find_keyword_lines(layout_lines)

        extracted: Dict[str, ExtractedField] = {}

        # 1. Identity Number
        id_field = self._extract_identity_number(layout_lines, keyword_matches)
        if id_field:
            extracted["identityNumber"] = id_field

        # 2. Full Name
        name_field = self._extract_full_name(layout_lines, keyword_matches)
        if name_field:
            extracted["fullName"] = name_field

        # 3. Date of Birth
        dob_field = self._extract_date_field(layout_lines, keyword_matches, "dateOfBirth")
        if dob_field:
            extracted["dateOfBirth"] = dob_field

        # 4. Gender
        gender_field = self._extract_gender(layout_lines, keyword_matches)
        if gender_field:
            extracted["gender"] = gender_field

        # 5. Nationality
        nat_field = self._extract_nationality(layout_lines, keyword_matches)
        if nat_field:
            extracted["nationality"] = nat_field

        # 6. Place of Origin
        origin_field = self._extract_address_field(layout_lines, keyword_matches, "placeOfOrigin")
        if origin_field:
            extracted["placeOfOrigin"] = origin_field

        # 7. Place of Residence
        res_field = self._extract_address_field(layout_lines, keyword_matches, "placeOfResidence")
        if res_field:
            extracted["placeOfResidence"] = res_field

        # 8. Date of Issue
        doi_field = self._extract_date_field(layout_lines, keyword_matches, "dateOfIssue")
        if doi_field:
            extracted["dateOfIssue"] = doi_field

        # 9. Date of Expiry
        doe_field = self._extract_date_field(layout_lines, keyword_matches, "dateOfExpiry")
        if doe_field:
            extracted["dateOfExpiry"] = doe_field

        return extracted

    def _find_keyword_lines(
        self, layout_lines: List[LayoutLine]
    ) -> Dict[str, Tuple[int, LayoutLine, str]]:
        matches: Dict[str, Tuple[int, LayoutLine, str]] = {}

        for line_idx, line in enumerate(layout_lines):
            match_res = self.label_matcher.match_line_label(line.text)
            if match_res:
                field_name, matched_label, conf = match_res
                if field_name not in matches:
                    matches[field_name] = (line_idx, line, matched_label)

        return matches

    def _extract_identity_number(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("identityNumber")

        candidates = []
        for line_idx, line in enumerate(layout_lines):
            # Check full line
            norm_id = normalize_identity_number(line.text)
            if norm_id:
                score = 0.5 + line.confidence * 0.3
                if kw_info and abs(line_idx - kw_info[0]) <= 1:
                    score += 0.2
                if line.center_y < 400:
                    score += 0.1
                candidates.append((score, norm_id, line.text, line))
            else:
                # Sub-search in line text for 9 or 12 digit candidate
                match = re.search(r'\b[0O][0-9OIL]{11}\b|\b[0-9OIL]{12}\b|\b[0-9OIL]{9}\b', line.text, re.IGNORECASE)
                if match:
                    raw_id = match.group(0)
                    norm_id = normalize_identity_number(raw_id)
                    if norm_id:
                        score = 0.5 + line.confidence * 0.3
                        if kw_info and abs(line_idx - kw_info[0]) <= 1:
                            score += 0.2
                        if line.center_y < 400:
                            score += 0.1
                        candidates.append((score, norm_id, raw_id, line))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score, best_id, best_raw, best_line = candidates[0]

        kw_text = kw_info[1].text if kw_info else "Số / No."
        return ExtractedField(
            fieldName="identityNumber",
            value=best_id,
            rawText=best_raw,
            keyword=kw_text,
            language="VI/EN",
            confidence=round(min(0.99, best_score), 2),
            bbox=[[float(pt[0]), float(pt[1])] for pt in best_line.tokens[0].bbox] if best_line.tokens else None
        )

    def _extract_full_name(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("fullName")
        if not kw_info:
            return None

        kw_idx, kw_line, kw_str = kw_info

        raw_name = ""
        target_line_idx = kw_idx

        inline_val = self._strip_header_label(kw_line.text, "fullName")
        if inline_val and len(inline_val) > 2:
            raw_name = inline_val
        elif kw_idx + 1 < len(layout_lines):
            next_line = layout_lines[kw_idx + 1]
            if not self._is_keyword_line(next_line):
                raw_name = next_line.text
                target_line_idx = kw_idx + 1

        if not raw_name:
            return None

        canonical_val, raw_clean = normalize_full_name(raw_name)

        logger.info(f"[FIELD_EXTRACTOR] fullName keyword='{kw_str}' raw='{raw_clean}' canonical='{canonical_val}'")

        return ExtractedField(
            fieldName="fullName",
            value=canonical_val,
            rawText=raw_clean,
            keyword=kw_line.text,
            language="VI/EN",
            confidence=round(kw_line.confidence, 2),
            bbox=[[float(pt[0]), float(pt[1])] for pt in layout_lines[target_line_idx].tokens[0].bbox] if layout_lines[target_line_idx].tokens else None
        )

    def _extract_gender(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("gender")

        raw_gender_str = None
        target_line = None

        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            inline_val = self._strip_header_label(kw_line.text, "gender")
            if inline_val:
                raw_gender_str = inline_val
                target_line = kw_line
            elif kw_idx + 1 < len(layout_lines):
                raw_gender_str = layout_lines[kw_idx + 1].text
                target_line = layout_lines[kw_idx + 1]

        if not raw_gender_str:
            for line in layout_lines:
                clean_text = remove_vietnamese_accents(line.text).lower()
                if "nam" in clean_text or "male" in clean_text or "nu" in clean_text or "female" in clean_text:
                    raw_gender_str = line.text
                    target_line = line
                    break

        if not raw_gender_str:
            return None

        norm_gender = normalize_gender(raw_gender_str)
        if not norm_gender:
            return None

        logger.info(f"[FIELD_EXTRACTOR] gender raw='{raw_gender_str}' normalized='{norm_gender}'")

        return ExtractedField(
            fieldName="gender",
            value=norm_gender,
            rawText=raw_gender_str,
            keyword=kw_info[1].text if kw_info else "Giới tính / Sex",
            language="VI/EN",
            confidence=round(target_line.confidence, 2) if target_line else 0.95,
            bbox=[[float(pt[0]), float(pt[1])] for pt in target_line.tokens[0].bbox] if target_line and target_line.tokens else None
        )

    def _extract_nationality(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("nationality")

        raw_nat = "Việt Nam"
        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            inline_val = self._strip_header_label(kw_line.text, "nationality")
            if inline_val and len(inline_val) >= 3:
                raw_nat = inline_val

        return ExtractedField(
            fieldName="nationality",
            value="Việt Nam",
            rawText=raw_nat,
            keyword=kw_info[1].text if kw_info else "Quốc tịch / Nationality",
            language="VI/EN",
            confidence=0.98
        )

    def _extract_address_field(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]],
        field_name: str
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get(field_name)
        if not kw_info:
            return None

        kw_idx, kw_line, kw_str = kw_info

        gathered_tokens = []

        inline_val = self._strip_header_label(kw_line.text, field_name)
        if inline_val and len(inline_val) > 1:
            gathered_tokens.append(inline_val)

        for j in range(kw_idx + 1, len(layout_lines)):
            line = layout_lines[j]
            if self._is_keyword_line(line, exclude_field=field_name):
                break

            clean_j = remove_vietnamese_accents(line.text).lower()
            if re.search(r'co gia tri den|date of expiry|date expiry|bo cong an|ministry', clean_j):
                break

            gathered_tokens.append(line.text)

        if not gathered_tokens:
            return None

        raw_joined = " ".join(gathered_tokens)
        norm_val, clean_raw = normalize_address(raw_joined)

        if not norm_val:
            return None

        logger.info(f"[FIELD_EXTRACTOR] {field_name} raw='{clean_raw}' normalized='{norm_val}'")

        return ExtractedField(
            fieldName=field_name,
            value=norm_val,
            rawText=clean_raw,
            keyword=kw_line.text,
            language="VI/EN",
            confidence=round(kw_line.confidence, 2),
            bbox=[[float(pt[0]), float(pt[1])] for pt in kw_line.tokens[0].bbox] if kw_line.tokens else None
        )

    def _extract_date_field(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]],
        field_name: str
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get(field_name)
        if not kw_info:
            return None

        kw_idx, kw_line, kw_str = kw_info

        search_text = kw_line.text
        if kw_idx + 1 < len(layout_lines):
            search_text += " " + layout_lines[kw_idx + 1].text

        parsed = parse_date(search_text)
        if not parsed:
            return None

        match = re.search(r'\b(0[1-9]|[12]\d|3[01])[\/.\-](0[1-9]|1[0-2])[\/.\-]((?:19|20)\d{2})\b', search_text)
        raw_date_str = match.group(0) if match else parsed

        logger.info(f"[FIELD_EXTRACTOR] {field_name} raw='{raw_date_str}' iso='{parsed}'")

        return ExtractedField(
            fieldName=field_name,
            value=parsed,
            rawText=raw_date_str,
            keyword=kw_line.text,
            language="VI/EN",
            confidence=round(kw_line.confidence, 2),
            bbox=[[float(pt[0]), float(pt[1])] for pt in kw_line.tokens[0].bbox] if kw_line.tokens else None
        )

    def _strip_header_label(self, line_text: str, field_name: str) -> Optional[str]:
        kw_list = FIELD_LABELS.get(field_name, [])
        if not kw_list:
            return None

        # Build regex matching exact card labels for this field
        kw_patterns = [re.escape(k) for k in sorted(kw_list, key=len, reverse=True)]
        pattern_str = r'^.*?(?:' + '|'.join(kw_patterns) + r')[:\s\/._]*'

        cleaned = re.sub(pattern_str, '', line_text, flags=re.IGNORECASE).strip()

        # Also strip unaccented versions if OCR missed accents
        if cleaned == line_text:
            unaccented_patterns = [re.escape(remove_vietnamese_accents(k)) for k in sorted(kw_list, key=len, reverse=True)]
            unaccented_regex = r'^.*?(?:' + '|'.join(unaccented_patterns) + r')[:\s\/._]*'
            cleaned = re.sub(unaccented_regex, '', line_text, flags=re.IGNORECASE).strip()

        if cleaned and len(cleaned) > 0 and cleaned != line_text:
            return cleaned

        return None

    def _is_keyword_line(self, line: LayoutLine, exclude_field: Optional[str] = None) -> bool:
        match_res = self.label_matcher.match_line_label(line.text)
        if match_res:
            field_name, _, _ = match_res
            if exclude_field and field_name == exclude_field:
                return False
            return True
        return False
