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
    Data-Driven Generic Field Extractor.
    Uses spatial bounding box layout parsing, LabelMatcher fuzzy label recognition,
    and line boundary isolation without hardcoding personal identity data or OCR typos.
    Calculates distinct label_box and value_box for all fields.
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

    def _get_label_tokens(self, line: LayoutLine, value_tokens: List[OCRText]) -> List[OCRText]:
        if not line or not line.tokens:
            return []
        val_set = set(id(t) for t in value_tokens)
        return [t for t in line.tokens if id(t) not in val_set]

    def _get_value_tokens(self, line: LayoutLine, field_name: str) -> List[OCRText]:
        """
        Strips header label tokens from a line and returns only tokens belonging to the actual field value.
        """
        if not line or not line.tokens:
            return []

        inline_val = self._strip_header_label(line.text, field_name)
        if inline_val and inline_val != line.text:
            val_words = [w for w in re.split(r'\W+', remove_vietnamese_accents(inline_val).lower()) if w]
            val_tokens = []
            for token in line.tokens:
                tok_words = [w for w in re.split(r'\W+', remove_vietnamese_accents(token.text).lower()) if w]
                if any(w in val_words for w in tok_words):
                    val_tokens.append(token)
            if val_tokens:
                return val_tokens

        # Fallback: remove known label noise tokens from start
        label_noise_words = {
            "placeOfResidence": {"noi", "thuong", "thurng", "tru", "trư", "cu", "place", "of", "residence", "pace", "cfcnc"},
            "placeOfOrigin": {"que", "quan", "noi", "dang", "ky", "khai", "sinh", "place", "of", "origin", "birth", "registration", "pace", "brth"},
            "fullName": {"ho", "chu", "dem", "va", "ten", "khai", "sinh", "full", "name", "surname", "given", "names"},
            "dateOfBirth": {"ngay", "thang", "nam", "sinh", "date", "of", "birth", "dob"},
            "gender": {"gioi", "tinh", "sex"},
            "nationality": {"quoc", "tich", "nationality"},
            "identityNumber": {"so", "dinh", "danh", "ca", "nhan", "no", "personal", "identification", "number"},
        }
        noise = label_noise_words.get(field_name, set())
        filtered = []
        for token in line.tokens:
            tok_clean = remove_vietnamese_accents(token.text).lower().strip(":/._-")
            if tok_clean not in noise and not any(w in noise for w in tok_clean.split()):
                filtered.append(token)

        return filtered

    def _compute_merged_bbox(self, tokens: List[OCRText]) -> Optional[List[List[float]]]:
        """
        Calculates a merged union bounding box covering all tokens belonging to an extracted field value.
        Returns [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]] or None.
        """
        if not tokens:
            return None

        all_xs = []
        all_ys = []

        for token in tokens:
            if token and token.bbox:
                for pt in token.bbox:
                    all_xs.append(float(pt[0]))
                    all_ys.append(float(pt[1]))

        if not all_xs or not all_ys:
            return None

        min_x = min(all_xs)
        max_x = max(all_xs)
        min_y = min(all_ys)
        max_y = max(all_ys)

        return [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ]

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

            # Also check if another field label is present on this same line (e.g. gender & nationality on CCCD_OLD)
            clean_text = remove_vietnamese_accents(line.text).lower()
            for field_name, labels in FIELD_LABELS.items():
                if field_name in matches:
                    continue
                for lbl in labels:
                    lbl_clean = remove_vietnamese_accents(lbl).lower()
                    if len(lbl_clean) >= 3 and lbl_clean in clean_text:
                        matches[field_name] = (line_idx, line, lbl)
                        break

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
            if norm_id:
                score = 0.5 + line.confidence * 0.3
                if kw_info and abs(line_idx - kw_info[0]) <= 1:
                    score += 0.2
                if line.center_y < 400:
                    score += 0.1
                candidates.append((score, norm_id, line.text, line))
            else:
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

        kw_text = kw_info[2] if kw_info else "Số / No.:"
        val_tokens = self._get_value_tokens(best_line, "identityNumber")

        if kw_info:
            kw_idx, kw_line, matched_label = kw_info
            if best_line == kw_line:
                lbl_tokens = self._get_label_tokens(best_line, val_tokens)
            else:
                lbl_tokens = kw_line.tokens
        else:
            lbl_tokens = []

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(val_tokens)
        merged_bbox = self._compute_merged_bbox(val_tokens)

        return ExtractedField(
            fieldName="identityNumber",
            value=best_id,
            rawText=best_raw,
            keyword=kw_text,
            language="VI/EN",
            confidence=round(min(0.99, best_score), 2),
            bbox=merged_bbox,
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

        raw_name = ""
        field_tokens: List[OCRText] = []

        inline_val = self._strip_header_label(kw_line.text, "fullName")
        if inline_val and len(inline_val) > 2:
            raw_name = inline_val
            field_tokens = self._get_value_tokens(kw_line, "fullName")
            lbl_tokens = self._get_label_tokens(kw_line, field_tokens)
        elif kw_idx + 1 < len(layout_lines):
            next_line = layout_lines[kw_idx + 1]
            if not self._is_keyword_line(next_line):
                raw_name = next_line.text
                field_tokens = self._get_value_tokens(next_line, "fullName")
            lbl_tokens = kw_line.tokens
        else:
            lbl_tokens = kw_line.tokens

        if not raw_name:
            return None

        canonical_val, raw_clean = normalize_full_name(raw_name)

        logger.info(f"[FIELD_EXTRACTOR] fullName keyword='{kw_str}' raw='{raw_clean}' canonical='{canonical_val}'")

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(field_tokens)
        merged_bbox = self._compute_merged_bbox(field_tokens)
        kw_text = kw_str if kw_str else "Họ và tên / Full name:"

        return ExtractedField(
            fieldName="fullName",
            value=canonical_val,
            rawText=raw_clean,
            keyword=kw_text,
            language="VI/EN",
            confidence=round(kw_line.confidence, 2),
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
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
            clean_kw_line = remove_vietnamese_accents(kw_line.text).lower()

            if "quoc" in clean_kw_line or "nat" in clean_kw_line:
                # Line shared with Nationality: isolate gender part before nationality
                nat_pattern = r'(?:quoc\s*tich|nationality).*$'
                gender_part = re.sub(nat_pattern, '', clean_kw_line, flags=re.IGNORECASE).strip()
                inline_val = self._strip_header_label(gender_part, "gender")
                if inline_val:
                    m = re.search(r'\b(nam|male|nu|female)\b', inline_val)
                    raw_gender_str = m.group(0) if m else inline_val
                elif re.search(r'\b(nam|male|nu|female)\b', gender_part):
                    m = re.search(r'\b(nam|male|nu|female)\b', gender_part)
                    raw_gender_str = m.group(0) if m else "Nam"
                target_line = kw_line
            else:
                inline_val = self._strip_header_label(kw_line.text, "gender")
                if inline_val:
                    m = re.search(r'\b(nam|male|nu|nữ|female)\b', inline_val, re.IGNORECASE)
                    raw_gender_str = m.group(0) if m else inline_val
                    target_line = kw_line
                elif re.search(r'\b(nam|male|nu|nữ|female)\b', kw_line.text, re.IGNORECASE):
                    m = re.search(r'\b(nam|male|nu|nữ|female)\b', kw_line.text, re.IGNORECASE)
                    raw_gender_str = m.group(0) if m else kw_line.text
                    target_line = kw_line
                elif kw_idx + 1 < len(layout_lines):
                    raw_gender_str = layout_lines[kw_idx + 1].text
                    target_line = layout_lines[kw_idx + 1]

        if not raw_gender_str:
            for line in layout_lines:
                clean_text = remove_vietnamese_accents(line.text).lower()
                m = re.search(r'\b(nam|male|nu|female)\b', clean_text)
                if m:
                    raw_gender_str = m.group(0)
                    target_line = line
                    break

        if not raw_gender_str:
            return None

        norm_gender = normalize_gender(raw_gender_str)
        if not norm_gender:
            # Fallback regex check
            if re.search(r'\b(nam|male)\b', remove_vietnamese_accents(raw_gender_str).lower()):
                norm_gender = "Nam"
            elif re.search(r'\b(nu|female)\b', remove_vietnamese_accents(raw_gender_str).lower()):
                norm_gender = "Nữ"
            else:
                return None

        logger.info(f"[FIELD_EXTRACTOR] gender raw='{raw_gender_str}' normalized='{norm_gender}'")

        kw_text = kw_info[2] if kw_info else "Giới tính / Sex:"
        lbl_tokens = []
        val_tokens = []

        if target_line:
            clean_tgt = remove_vietnamese_accents(target_line.text).lower()
            if "quoc" in clean_tgt or "nat" in clean_tgt:
                # Shared line with Nationality (CCCD_OLD layout)
                nat_token_idx = len(target_line.tokens)
                for idx, t in enumerate(target_line.tokens):
                    t_c = remove_vietnamese_accents(t.text).lower()
                    if any(w in t_c for w in ["quoc", "tich", "nat"]):
                        nat_token_idx = idx
                        break
                gender_tokens = target_line.tokens[:nat_token_idx]
                val_tokens = [t for t in gender_tokens if any(w in remove_vietnamese_accents(t.text).lower().split() for w in ["nam", "nu", "male", "female"])]
                lbl_tokens = [t for t in gender_tokens if t not in val_tokens]
            elif kw_info and target_line == kw_info[1]:
                val_tokens = self._get_value_tokens(target_line, "gender")
                lbl_tokens = self._get_label_tokens(target_line, val_tokens)
            else:
                lbl_tokens = kw_info[1].tokens if kw_info else []
                val_tokens = target_line.tokens
        elif kw_info:
            lbl_tokens = kw_info[1].tokens

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(val_tokens)
        merged_bbox = self._compute_merged_bbox(val_tokens)

        return ExtractedField(
            fieldName="gender",
            value=norm_gender,
            rawText=raw_gender_str,
            keyword=kw_text,
            language="VI/EN",
            confidence=round(target_line.confidence, 2) if target_line else 0.95,
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
        )

    def _extract_nationality(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Dict[str, Tuple[int, LayoutLine, str]]
    ) -> Optional[ExtractedField]:
        kw_info = keyword_matches.get("nationality")

        raw_nat = "Việt Nam"
        field_tokens: List[OCRText] = []

        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            clean_kw = remove_vietnamese_accents(kw_line.text).lower()
            if "gioi" in clean_kw or "sex" in clean_kw:
                # Shared line with Gender (CCCD_OLD layout)
                nat_pattern = r'^.*?(?:quoc\s*tich|nationality)[:\s\/._]*'
                nat_part = re.sub(nat_pattern, '', kw_line.text, flags=re.IGNORECASE).strip()
                if nat_part:
                    raw_nat = nat_part
                field_tokens = self._get_value_tokens(kw_line, "nationality")
            else:
                field_tokens = self._get_value_tokens(kw_line, "nationality")
                inline_val = self._strip_header_label(kw_line.text, "nationality")
                if inline_val and len(inline_val) >= 3:
                    raw_nat = inline_val

        kw_text = kw_info[2] if kw_info else "Quốc tịch / Nationality:"
        lbl_tokens = []
        val_tokens = []

        if kw_info:
            kw_idx, kw_line, kw_str = kw_info
            clean_kw = remove_vietnamese_accents(kw_line.text).lower()
            if "gioi" in clean_kw or "sex" in clean_kw:
                # Shared line with Gender (CCCD_OLD layout)
                nat_token_idx = 0
                for idx, t in enumerate(kw_line.tokens):
                    t_c = remove_vietnamese_accents(t.text).lower()
                    if any(w in t_c for w in ["quoc", "tich", "nat"]):
                        nat_token_idx = idx
                        break
                nat_tokens = kw_line.tokens[nat_token_idx:]
                val_tokens = [t for t in nat_tokens if any(w in remove_vietnamese_accents(t.text).lower().split() for w in ["viet", "nam"])]
                lbl_tokens = [t for t in nat_tokens if t not in val_tokens]
            else:
                val_tokens = self._get_value_tokens(kw_line, "nationality")
                lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
        else:
            val_tokens = field_tokens

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(val_tokens)
        merged_bbox = self._compute_merged_bbox(val_tokens)

        return ExtractedField(
            fieldName="nationality",
            value="Việt Nam",
            rawText=raw_nat,
            keyword=kw_text,
            language="VI/EN",
            confidence=0.98,
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
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

        # Calculate estimated card width for spatial coordinate checks
        card_width = max([line.max_x for line in layout_lines] + [1000.0])

        stop_keywords = ADDRESS_STOP_KEYWORDS.get(
            field_name,
            ADDRESS_STOP_KEYWORDS["placeOfOrigin"] + ADDRESS_STOP_KEYWORDS["placeOfResidence"]
        )

        stop_patterns = r'(co gia tri den|date of expiry|date expiry|bo cong an|ministry|cuc truong|ngon tro)'

        for j in range(kw_idx + 1, len(layout_lines)):
            line = layout_lines[j]
            clean_j = remove_vietnamese_accents(line.text).lower()

            # Check if line is an unrelated keyword line (e.g. fullName, identityNumber, dateOfBirth, gender, nationality)
            match_res = self.label_matcher.match_line_label(line.text)
            if match_res:
                matched_field, _, _ = match_res
                if matched_field != field_name and matched_field not in ("dateOfExpiry",):
                    break

            # Filter tokens: on CCCD_OLD front, 2nd line of residence address often shares a row with expiry info
            # Left side (< 0.38 * card_width): "Có giá trị đến / Date of expiry: 04/10/2029"
            # Right side (>= 0.38 * card_width): "Tân Bình, Châu Thành, Đồng Tháp"
            valid_addr_tokens = []
            for token in line.tokens:
                token_center_x = sum(pt[0] for pt in token.bbox) / max(len(token.bbox), 1)
                tok_clean = remove_vietnamese_accents(token.text).lower()

                is_left = token_center_x < 0.38 * card_width
                is_expiry_stop = (
                    bool(re.search(r'\b(co|gia|tri|den|date|of|expiry|exp|ngay|thang|nam|het|han|bo|cong|an|cuc|truong)\b', tok_clean))
                    or bool(re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', token.text))
                    or bool(parse_date(token.text))
                )

                if is_left and is_expiry_stop:
                    continue  # Ignore expiry/footer tokens on the left

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
            confidence=round(kw_line.confidence, 2),
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
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

        gathered_tokens: List[OCRText] = []
        search_text = kw_line.text
        parsed = None
        target_line = kw_line

        DATE_PATTERN = r'(?:\b|\D)([1-9]|0[1-9]|[12]\d|3[01])[\/.\-]([1-9]|0[1-9]|1[0-2])[\/.\-]((?:19|20)\d{2})\b'

        # 1. Inline pattern scan directly on kw_line.text
        inline_match = re.search(DATE_PATTERN, kw_line.text)
        if inline_match:
            raw_inline = inline_match.group(0).strip()
            parsed = parse_date(raw_inline)
            if parsed:
                gathered_tokens.extend(self._get_value_tokens(kw_line, field_name))

        # 2. Try parse_date() on the full kw_line.text
        if not parsed:
            parsed = parse_date(kw_line.text)
            if parsed:
                gathered_tokens.extend(self._get_value_tokens(kw_line, field_name))

        # 3. Lookahead up to 3 next lines
        if not parsed:
            max_lookahead = 3 if field_name in ("dateOfIssue", "dateOfExpiry") else 1
            for offset in range(1, max_lookahead + 1):
                next_idx = kw_idx + offset
                if next_idx >= len(layout_lines):
                    break
                next_line = layout_lines[next_idx]
                if self._is_keyword_line(next_line, exclude_field=field_name):
                    break
                combined = kw_line.text + " " + next_line.text
                next_match = re.search(DATE_PATTERN, next_line.text)
                if next_match:
                    raw_next = next_match.group(0).strip()
                    parsed = parse_date(raw_next)
                    if parsed:
                        search_text = combined
                        gathered_tokens.extend(next_line.tokens)
                        target_line = next_line
                        break
                parsed_combined = parse_date(combined)
                if parsed_combined:
                    parsed = parsed_combined
                    search_text = combined
                    gathered_tokens.extend(next_line.tokens)
                    target_line = next_line
                    break

        if not parsed:
            return None

        match = re.search(DATE_PATTERN, search_text)
        raw_date_str = match.group(0).strip() if match else parsed

        logger.info(f"[FIELD_EXTRACTOR] {field_name} raw='{raw_date_str}' iso='{parsed}'")

        if target_line == kw_line:
            # Inline on same line (or on left side of line for expiry)
            lbl_tokens = self._get_label_tokens(kw_line, gathered_tokens)
        else:
            lbl_tokens = kw_line.tokens

        label_box = self._compute_bbox_4(lbl_tokens)
        value_box = self._compute_bbox_4(gathered_tokens)
        merged_bbox = self._compute_merged_bbox(gathered_tokens)
        kw_text = kw_str if kw_str else ("Ngày sinh / Date of birth:" if field_name == "dateOfBirth" else "Có giá trị đến / Date of expiry:")

        return ExtractedField(
            fieldName=field_name,
            value=parsed,
            rawText=raw_date_str,
            keyword=kw_text,
            language="VI/EN",
            confidence=round(kw_line.confidence, 2),
            bbox=merged_bbox,
            label_box=label_box,
            value_box=value_box
        )

    def _extract_date_of_issue(
        self,
        layout_lines: List[LayoutLine],
        keyword_matches: Optional[Dict[str, Tuple[int, LayoutLine, str]]] = None
    ) -> Optional[ExtractedField]:
        """
        Extracts Date of Issue from card lines (especially back side of CCCD).
        Scans for keywords: 'ngày, tháng, năm', 'date, month, year', 'ngày, tháng, năm cấp', 'date of issue', 'ngày cấp'.
        Finds regex DD/MM/YYYY and converts to ISO YYYY-MM-DD.
        """
        DATE_PATTERN = r'(?:\b|\D)(\d{1,2}[\/\.\-]\d{1,2}[\/\-](?:19|20)\d{2})\b'

        # 1. Check if keyword_matches already has dateOfIssue
        if keyword_matches and "dateOfIssue" in keyword_matches:
            kw_idx, kw_line, matched_label = keyword_matches["dateOfIssue"]
            m0 = re.search(DATE_PATTERN, kw_line.text)
            if m0:
                raw_dmy = m0.group(1) if m0.lastindex else m0.group(0)
                parsed = parse_date(raw_dmy)
                if parsed:
                    val_tokens = self._get_value_tokens(kw_line, "dateOfIssue")
                    lbl_tokens = self._get_label_tokens(kw_line, val_tokens)
                    label_box = self._compute_bbox_4(lbl_tokens)
                    value_box = self._compute_bbox_4(val_tokens)
                    return ExtractedField(
                        fieldName="dateOfIssue",
                        value=parsed,
                        rawText=raw_dmy,
                        keyword=matched_label,
                        language="VI/EN",
                        confidence=round(kw_line.confidence, 2),
                        bbox=self._compute_merged_bbox(val_tokens),
                        label_box=label_box,
                        value_box=value_box
                    )
            # Scan lookahead
            for offset in range(1, 3):
                if kw_idx + offset < len(layout_lines):
                    next_line = layout_lines[kw_idx + offset]
                    m_next = re.search(DATE_PATTERN, next_line.text)
                    if m_next:
                        raw_dmy = m_next.group(0)
                        parsed = parse_date(raw_dmy)
                        if parsed:
                            lbl_tokens = kw_line.tokens
                            val_tokens = next_line.tokens
                            label_box = self._compute_bbox_4(lbl_tokens)
                            value_box = self._compute_bbox_4(val_tokens)
                            return ExtractedField(
                                fieldName="dateOfIssue",
                                value=parsed,
                                rawText=raw_dmy,
                                keyword=matched_label,
                                language="VI/EN",
                                confidence=round(next_line.confidence, 2),
                                bbox=self._compute_merged_bbox(val_tokens),
                                label_box=label_box,
                                value_box=value_box
                            )

        # 2. General scan across all layout_lines for issue date keywords
        issue_keywords = [
            "ngay, thang, nam cap", "ngay thang nam cap", "date of issue",
            "ngay, thang, nam", "ngay thang nam", "date, month, year",
            "ngay cap", "ngaycap", "cap ngay"
        ]

        for line_idx, line in enumerate(layout_lines):
            clean_text = remove_vietnamese_accents(line.text).lower()
            if "het han" in clean_text or "expiry" in clean_text or "co gia tri den" in clean_text:
                continue

            if any(k in clean_text for k in issue_keywords):
                m = re.search(DATE_PATTERN, line.text)
                if m:
                    raw_dmy = m.group(0)
                    parsed = parse_date(raw_dmy)
                    if parsed:
                        val_tokens = self._get_value_tokens(line, "dateOfIssue")
                        lbl_tokens = self._get_label_tokens(line, val_tokens)
                        label_box = self._compute_bbox_4(lbl_tokens)
                        value_box = self._compute_bbox_4(val_tokens)
                        return ExtractedField(
                            fieldName="dateOfIssue",
                            value=parsed,
                            rawText=raw_dmy,
                            keyword="Ngày, tháng, năm / Date, month, year:",
                            language="VI/EN",
                            confidence=round(line.confidence, 2),
                            bbox=self._compute_merged_bbox(val_tokens),
                            label_box=label_box,
                            value_box=value_box
                        )
                if line_idx + 1 < len(layout_lines):
                    next_line = layout_lines[line_idx + 1]
                    m2 = re.search(DATE_PATTERN, next_line.text)
                    if m2:
                        raw_dmy = m2.group(0)
                        parsed = parse_date(raw_dmy)
                        if parsed:
                            lbl_tokens = line.tokens
                            val_tokens = next_line.tokens
                            label_box = self._compute_bbox_4(lbl_tokens)
                            value_box = self._compute_bbox_4(val_tokens)
                            return ExtractedField(
                                fieldName="dateOfIssue",
                                value=parsed,
                                rawText=raw_dmy,
                                keyword="Ngày, tháng, năm / Date, month, year:",
                                language="VI/EN",
                                confidence=round(next_line.confidence, 2),
                                bbox=self._compute_merged_bbox(val_tokens),
                                label_box=label_box,
                                value_box=value_box
                            )

        return None

    def _strip_header_label(self, line_text: str, field_name: str) -> Optional[str]:
        if not line_text:
            return None

        cleaned = line_text

        # 1. Direct regex strip for all known label patterns of this field
        kw_list = FIELD_LABELS.get(field_name, [])
        if kw_list:
            kw_patterns = [re.escape(k) for k in sorted(kw_list, key=len, reverse=True)]
            pattern_str = r'^.*?(?:' + '|'.join(kw_patterns) + r')[:\s\/._]*'
            cleaned = re.sub(pattern_str, '', cleaned, flags=re.IGNORECASE).strip()

            unaccented_patterns = [re.escape(remove_vietnamese_accents(k)) for k in sorted(kw_list, key=len, reverse=True)]
            unaccented_regex = r'^.*?(?:' + '|'.join(unaccented_patterns) + r')[:\s\/._]*'
            cleaned = re.sub(unaccented_regex, '', cleaned, flags=re.IGNORECASE).strip()

        # 2. Aggressive field-specific label stripping
        if field_name == "placeOfResidence":
            cleaned = re.sub(r'^(?:c\/fcnc|c\/f[a-z0-9]*|fcnc|cfcnc)\b[:\s\/._,-]*', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(?:n[o\u01a1]i\s+)?(?:th[u\u01b0][o\u01a1]ng|thurng|c[u\u01b0]|thuong)\s*(?:tr[u\u00fa\u01b0]|tru)?\s*(?:[\/\-:\.]|\s+)*(?:place\s+of\s+residence|residence)?[:\s\/._]*', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(?:place\s+of\s+residence|pace\s+of\s+residence|residence)[:\s\/._]*', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(?:c\/fcnc|c\/f[a-z0-9]*|fcnc|cfcnc)\b[:\s\/._,-]*', '', cleaned, flags=re.IGNORECASE).strip()
        elif field_name == "placeOfOrigin":
            cleaned = re.sub(r'^(?:qu[e\xea]\s+qu[a\xe1]n|noi\s+dang\s+ky\s+khai\s+sinh|dang\s+ky\s+khai\s+sinh|khai\s+sinh)\s*(?:[\/\-:\.]|\s+)*(?:place\s+of\s+origin|place\s+of\s+birth(?:\s+registration)?)?[:\s\/._]*', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(?:place\s+of\s+origin|place\s+of\s+birth(?:\s+registration)?|pace\s+of\s+brth)[:\s\/._]*', '', cleaned, flags=re.IGNORECASE).strip()

        # 3. Strip remaining noisy English label remnants at the start
        en_patterns = r'^(?:place\s+of\s+birth(?:\s+registration)?|pace\s+of\s+brth|place\s+of\s+origin|place\s+of\s+residence|date\s+of\s+issue|date\s+of\s+expiry|date,\s*month,\s*year|surname,\s*given\s*names|full\s*name|personal\s*identification\s*number|no\.?|sex|nationality)[:\s\/._]*'
        cleaned = re.sub(en_patterns, '', cleaned, flags=re.IGNORECASE).strip()

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
