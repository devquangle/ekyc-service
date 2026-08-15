import re
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from ocr.detector import OCRText
from ocr.layout_parser import LayoutLine, LayoutParser
from ocr.label_matcher import LabelMatcher, FIELD_LABELS, ADDRESS_STOP_KEYWORDS
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
    bbox: Optional[List[List[float]]] = None  # Polygon representation
    label_box: Optional[List[float]] = None    # [x_min, y_min, x_max, y_max]
    value_box: Optional[List[float]] = None    # [x_min, y_min, x_max, y_max]


class FieldExtractor:
    """
    Data-Driven Generic Field Extractor using Relative Spatial Graph Layout & Bounding Box Topology.
    Operates pixel-agnostically across all card scales, resolutions, and orientations.
    Calculates distinct label_box and value_box for all extracted fields.
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
        if not res_field:
            # Spatial Topology Fallback: On cards where residence label was faded/missing,
            # extract address tokens situated in the bottom residence quadrant relative to origin and expiry.
            res_field = self._extract_residence_fallback(layout_lines, origin_field)

        if res_field:
            extracted["placeOfResidence"] = res_field

        # 8. Date of Issue
        doi_field = self._extract_date_of_issue(layout_lines, keyword_matches)
        if doi_field:
            extracted["dateOfIssue"] = doi_field

        # 9. Date of Expiry
        doe_field = self._extract_date_field(layout_lines, keyword_matches, "dateOfExpiry")
        if doe_field:
            extracted["dateOfExpiry"] = doe_field

        return extracted

    def _compute_bbox_4(self, tokens: List[OCRText]) -> Optional[List[float]]:
        if not tokens:
            return None
        xs = [pt[0] for t in tokens if t and t.bbox for pt in t.bbox]
        ys = [pt[1] for t in tokens if t and t.bbox for pt in t.bbox]
        if not xs or not ys:
            return None
        return [
            round(float(min(xs)), 1),
            round(float(min(ys)), 1),
            round(float(max(xs)), 1),
            round(float(max(ys)), 1)
        ]

    def _compute_merged_bbox(self, tokens: List[OCRText]) -> Optional[List[List[float]]]:
        if not tokens:
            return None
        box4 = self._compute_bbox_4(tokens)
        if not box4:
            return None
        x1, y1, x2, y2 = box4
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _get_label_tokens(self, line: LayoutLine, value_tokens: List[OCRText]) -> List[OCRText]:
        val_set = set(id(t) for t in value_tokens)
        return [t for t in line.tokens if id(t) not in val_set]

    def _get_value_tokens(self, line: LayoutLine, field_name: str) -> List[OCRText]:
        patterns = FIELD_LABELS.get(field_name, [])
        val_tokens: List[OCRText] = []
        for t in line.tokens:
            t_clean = remove_vietnamese_accents(t.text).lower().strip()
            # If token matches label keywords, skip
            is_lbl = False
            for p in patterns:
                p_clean = remove_vietnamese_accents(p).lower().strip()
                if t_clean in p_clean or p_clean in t_clean:
                    is_lbl = True
                    break
            if not is_lbl:
                val_tokens.append(t)
        return val_tokens if val_tokens else line.tokens

    def _strip_header_label(self, text: str, field_name: str) -> str:
        patterns = FIELD_LABELS.get(field_name, [])
        clean = text
        for p in sorted(patterns, key=len, reverse=True):
            escaped = re.escape(p)
            clean = re.sub(r'(?i)\b' + escaped + r'\b[:\.\s/]*', '', clean)
            clean = re.sub(r'(?i)^' + escaped + r'[:\.\s/]*', '', clean)

        clean = re.sub(r'^(?:s[o\u1ed1]\s*[\/:]\s*no\.?|s[o\u1ed1]\s*[:\.\s]|no\.?[:\s]|h[o\u1ecd]\s*v[a\u00e0]\s*t[e\u00ea]n[:\s\/]*|full\s*name[:\s\/]*|ng[a\u00e0]y\s*sinh[:\s\/]*|date\s*of\s*birth[:\s\/]*|gi[o\u1edbi]\s*t[i\u00ed]nh[:\s\/]*|sex[:\s\/]*|qu[o\u00f4\u1ed1]c\s*t[i\u1ecb]ch[:\s\/]*|nationality[:\s\/]*|qu[e\u00ea]\s*qu[a\u00e1]n[:\s\/]*|place\s*of\s*origin[:\s\/]*|n[o\u01a1]i\s*th[u\u01b0\u1edd]ng\s*tr[u\u00fa][:\s\/]*|place\s*of\s*residence[:\s\/]*|c[o\u00f3]\s*gi[a\u00e1]\s*tr\u1ecb\s*\u0111[e\u1ebf]n[:\s\/]*|date\s*of\s*expiry[:\s\/]*)\s*', '', clean, flags=re.IGNORECASE)
        
        # Robust fallback for sticky labels without spaces/accents (e.g., Quequan/ Place oforigin)
        if field_name == "placeOfOrigin":
            clean_unaccented = remove_vietnamese_accents(clean).lower()
            m = re.match(r'^(?:q\s*u\s*e\s*q\s*u\s*a\s*n\s*[\/\-:]*\s*p\s*l\s*a\s*c\s*e\s*o\s*f\s*o\s*r\s*i\s*g\s*i\s*n|q\s*u\s*e\s*q\s*u\s*a\s*n|p\s*l\s*a\s*c\s*e\s*o\s*f\s*o\s*r\s*i\s*g\s*i\s*n)[\s:\/\-]*', clean_unaccented)
            if m:
                clean = clean[m.end():]
        elif field_name == "fullName":
            clean_unaccented = remove_vietnamese_accents(clean).lower()
            m = re.match(r'^(?:h\s*o\s*v\s*a\s*t\s*e\s*n\s*[\/\-:]*\s*f\s*u\s*l\s*l\s*n\s*a\s*m\s*e|h\s*o\s*v\s*a\s*t\s*e\s*n|f\s*u\s*l\s*l\s*n\s*a\s*m\s*e)[\s:\/\-]*', clean_unaccented)
            if m:
                clean = clean[m.end():]

        return clean.strip()

    def _find_keyword_lines(
        self, layout_lines: List[LayoutLine]
    ) -> Dict[str, Tuple[int, LayoutLine, str]]:
        matches: Dict[str, Tuple[int, LayoutLine, str]] = {}
        for idx, line in enumerate(layout_lines):
            line_matches = self.label_matcher.match_all_line_labels(line.text)
            for field_name, (matched_pattern, _) in line_matches.items():
                if field_name not in matches:
                    matches[field_name] = (idx, line, matched_pattern)
        return matches

    def _extract_identity_number(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("identityNumber")
        candidates = []

        for line_idx, line in enumerate(layout_lines):
            norm_id = normalize_identity_number(line.text)
            if not norm_id:
                match = re.search(r'\b[0O][0-9OIL]{11}\b|\b[0-9OIL]{12}\b|\b[0-9OIL]{9}\b', line.text, re.IGNORECASE)
                if match:
                    norm_id = normalize_identity_number(match.group(0))

            if norm_id:
                score = 0.5 + line.confidence * 0.3
                if len(norm_id) == 12:
                    score += 0.5  # Strong preference for 12-digit CCCD numbers
                if kw_info and abs(line_idx - kw_info[0]) <= 1:
                    score += 0.3
                if line.norm_center_y <= 0.40:
                    score += 0.2
                candidates.append((score, norm_id, line.text, line))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score, best_id, best_raw, best_line = candidates[0]

        kw_text = kw_info[2] if kw_info else "Số / No.:"
        val_tokens = self._get_value_tokens(best_line, "identityNumber")
        if kw_info:
            kw_line = kw_info[1]
            if kw_info[0] == layout_lines.index(best_line):
                lbl_tokens = self._get_label_tokens(best_line, val_tokens)
            else:
                lbl_tokens = kw_line.tokens
        else:
            lbl_tokens = []

        label_box = self._compute_bbox_4(lbl_tokens) if lbl_tokens else None
        value_box = self._compute_bbox_4(val_tokens)

        return ExtractedField(
            fieldName="identityNumber",
            value=best_id,
            rawText=best_raw,
            keyword=kw_text,
            language="VI/EN",
            confidence=best_line.confidence,
            bbox=best_line.tokens[0].bbox if best_line.tokens else None,
            label_box=label_box,
            value_box=value_box
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
        inline_text = self._strip_header_label(kw_line.text, "fullName")
        canonical_name, clean_name = normalize_full_name(inline_text)

        clean_inline_for_noise = re.sub(r'[^a-z\s]', '', remove_vietnamese_accents(inline_text).lower())
        is_label_noise = any(
            noise in clean_inline_for_noise
            for noise in ["chu dem", "chidem", "khai sinh", "khal sinh", "full name", "fuilname", "ho va ten"]
        )

        if canonical_name and len(canonical_name.split()) >= 2 and not is_label_noise:
            val_tokens = self._get_value_tokens(kw_line, "fullName")
            lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
            return ExtractedField(
                fieldName="fullName",
                value=canonical_name,
                rawText=clean_name or inline_text,
                keyword=kw_str,
                language="VI/EN",
                confidence=kw_line.confidence,
                bbox=self._compute_merged_bbox(val_tokens),
                label_box=self._compute_bbox_4(lbl_tokens),
                value_box=self._compute_bbox_4(val_tokens)
            )


        # Check next line
        if kw_idx + 1 < len(layout_lines):
            next_line = layout_lines[kw_idx + 1]
            next_text = next_line.text
            # Strip DOB and dates to prevent merging
            next_text = re.sub(r'(?i)(ngay\s*sinh|date\s*of\s*birth|dob)[\/\s\.:]*', '', remove_vietnamese_accents(next_text))
            next_text = re.sub(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', '', next_text)
            
            canonical_name, clean_name = normalize_full_name(next_text)
            
            if canonical_name and len(canonical_name.split()) >= 2:
                # verify it's not noise
                clean_inline_for_noise = re.sub(r'[^a-z\s]', '', remove_vietnamese_accents(next_text).lower())
                is_label_noise = any(
                    noise in clean_inline_for_noise
                    for noise in ["chu dem", "chidem", "khai sinh", "khal sinh", "full name", "fuilname", "ho va ten"]
                )
                if not is_label_noise:
                    return ExtractedField(
                        fieldName="fullName",
                        value=canonical_name,
                        rawText=clean_name or next_text,
                        keyword=kw_str,
                        language="VI/EN",
                        confidence=next_line.confidence,
                        bbox=self._compute_merged_bbox(next_line.tokens),
                        label_box=self._compute_bbox_4(kw_line.tokens),
                        value_box=self._compute_bbox_4(next_line.tokens)
                    )

        return None

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
        is_expiry = (field_name == "dateOfExpiry")
        clean_kw_line = remove_vietnamese_accents(kw_line.text).lower()

        # Disambiguation: avoid matching expiry lines as birth lines
        if field_name == "dateOfBirth" and any(w in clean_kw_line for w in ["het han", "expiry", "gia tri den"]) and not any(w in clean_kw_line for w in ["sinh", "birth"]):
            return None
        if field_name == "dateOfExpiry" and any(w in clean_kw_line for w in ["ngay sinh", "date of birth", "nam sinh"]) and not any(w in clean_kw_line for w in ["het han", "expiry", "gia tri"]):
            return None

        # 1. Try inline match
        date_val = parse_date(kw_line.text, is_expiry=is_expiry)

        if date_val:
            val_tokens = [t for t in kw_line.tokens if parse_date(t.text, is_expiry=is_expiry) or re.search(r'\d', t.text)]
            lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
            return ExtractedField(
                fieldName=field_name,
                value=date_val,
                rawText=kw_line.text,
                keyword=kw_str,
                language="VI/EN",
                confidence=kw_line.confidence,
                bbox=self._compute_merged_bbox(val_tokens if val_tokens else kw_line.tokens),
                label_box=self._compute_bbox_4(lbl_tokens),
                value_box=self._compute_bbox_4(val_tokens if val_tokens else kw_line.tokens)
            )

        # 2. Try next line
        if kw_idx + 1 < len(layout_lines):
            next_line = layout_lines[kw_idx + 1]
            date_val = parse_date(next_line.text, is_expiry=is_expiry)
            if date_val:
                return ExtractedField(
                    fieldName=field_name,
                    value=date_val,
                    rawText=next_line.text,
                    keyword=kw_str,
                    language="VI/EN",
                    confidence=next_line.confidence,
                    bbox=self._compute_merged_bbox(next_line.tokens),
                    label_box=self._compute_bbox_4(kw_line.tokens),
                    value_box=self._compute_bbox_4(next_line.tokens)
                )

        return None

    def _extract_gender(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("gender")
        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            val_tokens = [t for t in kw_line.tokens if normalize_gender(t.text)]
            gender_val = normalize_gender(val_tokens[0].text) if val_tokens else normalize_gender(kw_line.text)
            if gender_val:
                lbl_tokens = [t for t in kw_line.tokens if re.search(r'\b(gioi|tinh|sex)\b', remove_vietnamese_accents(t.text).lower())]
                if not lbl_tokens:
                    lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
                return ExtractedField(
                    fieldName="gender",
                    value=gender_val,
                    rawText=kw_line.text,
                    keyword=kw_str,
                    language="VI/EN",
                    confidence=kw_line.confidence,
                    bbox=self._compute_merged_bbox(val_tokens if val_tokens else kw_line.tokens),
                    label_box=self._compute_bbox_4(lbl_tokens),
                    value_box=self._compute_bbox_4(val_tokens if val_tokens else kw_line.tokens)
                )

        # Fallback scan across lines
        for line in layout_lines:
            val_toks = [t for t in line.tokens if normalize_gender(t.text)]
            if val_toks:
                g = normalize_gender(val_toks[0].text)
                if g:
                    lbl_toks = [t for t in line.tokens if re.search(r'\b(gioi|tinh|sex)\b', remove_vietnamese_accents(t.text).lower())]
                    return ExtractedField(
                        fieldName="gender",
                        value=g,
                        rawText=line.text,
                        keyword="Giới tính / Sex:",
                        language="VI/EN",
                        confidence=line.confidence,
                        bbox=self._compute_merged_bbox(val_toks),
                        label_box=self._compute_bbox_4(lbl_toks) if lbl_toks else None,
                        value_box=self._compute_bbox_4(val_toks)
                    )

        return None

    def _extract_nationality(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("nationality")
        if not kw_info:
            # Fallback scan layout_lines for nationality label tokens or "Việt Nam"
            for line in layout_lines:
                clean_text = remove_vietnamese_accents(line.text).lower()
                if "quoc tich" in clean_text or "nationality" in clean_text or "viet nam" in clean_text or "vietnam" in clean_text:
                    lbl_tokens = [t for t in line.tokens if re.search(r'\b(quoc|tich|nationality)\b', remove_vietnamese_accents(t.text).lower())]
                    min_lbl_x = min(p[0] for l in lbl_tokens for p in l.bbox) if lbl_tokens else 0.0
                    val_tokens = [
                        t for t in line.tokens
                        if "viet" in remove_vietnamese_accents(t.text).lower()
                        or ("nam" in remove_vietnamese_accents(t.text).lower() and min(p[0] for p in t.bbox) >= min_lbl_x and t.text.strip() != "Nam")
                    ]
                    if not val_tokens:
                        val_tokens = [t for t in line.tokens if "viet" in remove_vietnamese_accents(t.text).lower()]
                    if val_tokens:
                        return ExtractedField(
                            fieldName="nationality",
                            value="Việt Nam",
                            rawText=line.text,
                            keyword="Quốc tịch / Nationality:",
                            language="VI/EN",
                            confidence=line.confidence,
                            bbox=self._compute_merged_bbox(val_tokens),
                            label_box=self._compute_bbox_4(lbl_tokens) if lbl_tokens else None,
                            value_box=self._compute_bbox_4(val_tokens)
                        )
            return None

        kw_idx, kw_line, kw_str = kw_info
        clean_text = remove_vietnamese_accents(kw_line.text).lower()
        if "viet nam" in clean_text or "vietnam" in clean_text or "vnm" in clean_text:
            lbl_tokens = [t for t in kw_line.tokens if re.search(r'\b(quoc|tich|nationality)\b', remove_vietnamese_accents(t.text).lower())]
            min_lbl_x = min(p[0] for l in lbl_tokens for p in l.bbox) if lbl_tokens else 0.0
            val_tokens = [
                t for t in kw_line.tokens
                if "viet" in remove_vietnamese_accents(t.text).lower()
                or ("nam" in remove_vietnamese_accents(t.text).lower() and min(p[0] for p in t.bbox) >= min_lbl_x and t.text.strip() != "Nam")
            ]
            if not val_tokens:
                val_tokens = [t for t in kw_line.tokens if "viet" in remove_vietnamese_accents(t.text).lower()]
            if not lbl_tokens:
                lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
            return ExtractedField(
                fieldName="nationality",
                value="Việt Nam",
                rawText=kw_line.text,
                keyword=kw_str,
                language="VI/EN",
                confidence=kw_line.confidence,
                bbox=self._compute_merged_bbox(val_tokens if val_tokens else kw_line.tokens),
                label_box=self._compute_bbox_4(lbl_tokens),
                value_box=self._compute_bbox_4(val_tokens if val_tokens else kw_line.tokens)
            )

        return None

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

        gathered_tokens: List[OCRText] = []
        gathered_text_parts: List[str] = []

        inline_val = self._strip_header_label(kw_line.text, field_name)
        if inline_val and len(inline_val) > 1:
            gathered_text_parts.append(inline_val)
            val1_toks = self._get_value_tokens(kw_line, field_name)
            gathered_tokens.extend(val1_toks)
            lbl_tokens = self._get_label_tokens(kw_line, val1_toks)
        else:
            lbl_tokens = kw_line.tokens

        stop_keywords = ADDRESS_STOP_KEYWORDS.get(
            field_name,
            ADDRESS_STOP_KEYWORDS["placeOfOrigin"] + ADDRESS_STOP_KEYWORDS["placeOfResidence"]
        )

        stop_patterns = r'(co gia tri den|date of expiry|date expiry|bo cong an|ministry|cuc truong|ngon tro)'

        for j in range(kw_idx + 1, len(layout_lines)):
            line = layout_lines[j]
            clean_j = remove_vietnamese_accents(line.text).lower()

            match_res = self.label_matcher.match_line_label(line.text)
            if match_res:
                matched_field, _, _ = match_res
                if matched_field != field_name and matched_field not in ("dateOfExpiry",):
                    break

            # Spatial topology check for placeOfOrigin: at most 1 value line below label
            if field_name == "placeOfOrigin" and (gathered_tokens or gathered_text_parts):
                if j > kw_idx + 1 or any(kw in clean_j for kw in ("co gia tri den", "date of expiry", "noi thuong tru", "place of residence")):
                    break

            # Spatial filter for CCCD_OLD front where 2nd residence line shares row with Expiry on the left
            valid_addr_tokens: List[OCRText] = []
            for token in line.tokens:
                tok_clean = remove_vietnamese_accents(token.text).lower()
                token_center_x = sum(pt[0] for pt in token.bbox) / max(len(token.bbox), 1)
                is_left_quadrant = line.norm_min_x < 0.38

                is_expiry_stop = (
                    bool(re.search(r'\b(co|gia|tri|den|date|of|expiry|exp|ngay|thang|nam|het|han|bo|cong|an|cuc|truong)\b', tok_clean))
                    or bool(re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', token.text))
                    or bool(parse_date(token.text))
                )

                if is_left_quadrant and is_expiry_stop:
                    continue
                if re.search(r'\b(bo cong an|ministry of public security|cuc truong cuc canh sat)\b', tok_clean):
                    continue

                valid_addr_tokens.append(token)

            if valid_addr_tokens:
                gathered_tokens.extend(valid_addr_tokens)
                gathered_text_parts.append(" ".join(t.text for t in valid_addr_tokens))
            else:
                if any(kw in clean_j for kw in stop_keywords) or re.search(stop_patterns, clean_j):
                    break

        if not gathered_tokens and not gathered_text_parts:
            return None

        raw_text = " ".join([t.text for t in gathered_tokens]) if gathered_tokens else " ".join(gathered_text_parts)
        restored_address, canonical_address = normalize_address(raw_text)

        if not restored_address:
            return None

        logger.info(f"[FIELD_EXTRACTOR] {field_name} raw='{raw_text}' restored='{restored_address}'")

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(gathered_tokens)
        merged_bbox = self._compute_merged_bbox(gathered_tokens)
        kw_text = kw_str if kw_str else ("Nơi thường trú / Place of residence:" if field_name == "placeOfResidence" else "Quê quán / Place of origin:")

        return ExtractedField(
            fieldName=field_name,
            value=restored_address,
            rawText=raw_text,
            keyword=kw_text,
            language="VI/EN",
            confidence=0.98,
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
        )

    def _extract_residence_fallback(
        self, layout_lines: List[LayoutLine], origin_field: Optional[ExtractedField]
    ) -> Optional[ExtractedField]:
        res_tokens: List[OCRText] = []
        for line in layout_lines:
            # Check if line is in the lower residence topology (norm_center_y >= 0.50)
            if line.norm_center_y >= 0.50:
                for token in line.tokens:
                    tok_clean = remove_vietnamese_accents(token.text).lower()
                    is_left_quadrant = line.norm_min_x < 0.38
                    is_expiry_stop = (
                        bool(re.search(r'\b(co|gia|tri|den|date|of|expiry|exp|ngay|thang|nam|het|han|bo|cong|an|cuc|truong)\b', tok_clean))
                        or bool(re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', token.text))
                        or bool(parse_date(token.text))
                    )
                    if is_left_quadrant and is_expiry_stop:
                        continue
                    if re.search(r'\b(bo cong an|ministry of public security|cuc truong cuc canh sat)\b', tok_clean):
                        continue
                    if origin_field and origin_field.value_box:
                        if token.bbox and sum(pt[1] for pt in token.bbox) / 4.0 <= origin_field.value_box[3]:
                            continue
                    res_tokens.append(token)

        if not res_tokens:
            return None

        raw_res = " ".join(t.text for t in res_tokens)
        restored_res, _ = normalize_address(raw_res)
        if not restored_res:
            return None

        return ExtractedField(
            fieldName="placeOfResidence",
            value=restored_res,
            rawText=raw_res,
            keyword="Nơi thường trú / Place of residence:",
            language="VI/EN",
            confidence=0.85,
            bbox=self._compute_merged_bbox(res_tokens),
            label_box=None,
            value_box=self._compute_bbox_4(res_tokens)
        )

    def _extract_date_of_issue(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("dateOfIssue")
        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            # Try inline
            m = re.search(r'(\d{1,2})\s*[\/\.\-]\s*(\d{1,2})\s*[\/\.\-]\s*(\d{2,4})', kw_line.text)
            if m:
                d, mth, y = m.group(1), m.group(2), m.group(3)
                date_val = parse_date(f"{d}/{mth}/{y}")
                if date_val:
                    val_toks = [t for t in kw_line.tokens if re.search(r'\d', t.text)]
                    lbl_toks = self._get_label_tokens(kw_line, val_toks)
                    return ExtractedField(
                        fieldName="dateOfIssue",
                        value=date_val,
                        rawText=kw_line.text,
                        keyword=kw_str,
                        language="VI/EN",
                        confidence=kw_line.confidence,
                        bbox=self._compute_merged_bbox(val_toks if val_toks else kw_line.tokens),
                        label_box=self._compute_bbox_4(lbl_toks),
                        value_box=self._compute_bbox_4(val_toks if val_toks else kw_line.tokens)
                    )

        # Fallback: scan for back-side issue date pattern "ngày ... tháng ... năm ..."
        for line in layout_lines:
            clean = remove_vietnamese_accents(line.text).lower()
            m = re.search(r'ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})', clean)
            if m:
                d, mth, y = m.group(1), m.group(2), m.group(3)
                date_val = parse_date(f"{d}/{mth}/{y}")
                if date_val:
                    return ExtractedField(
                        fieldName="dateOfIssue",
                        value=date_val,
                        rawText=line.text,
                        keyword="Ngày, tháng, năm / Date, month, year:",
                        language="VI/EN",
                        confidence=line.confidence,
                        bbox=self._compute_merged_bbox(line.tokens),
                        label_box=None,
                        value_box=self._compute_bbox_4(line.tokens)
                    )

        return None
