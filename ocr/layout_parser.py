from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from ocr.detector import OCRText
from utils.logger import logger


def compute_tokens_bbox(tokens: List[OCRText]) -> Optional[List[float]]:
    """
    Computes [x_min, y_min, x_max, y_max] bounding box covering a list of OCRText tokens.
    """
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


class LayoutLine(BaseModel):
    tokens: List[OCRText]
    text: str
    center_y: float
    min_y: float
    max_y: float
    min_x: float
    max_x: float
    height: float
    confidence: float
    norm_center_y: float = 0.0
    norm_center_x: float = 0.0
    norm_min_x: float = 0.0
    norm_max_x: float = 0.0


class LayoutParser:
    """
    Pixel-Agnostic Spatial Layout Parser for Vietnamese Identity Cards.
    Groups spatial OCRText tokens into ordered LayoutLines based on relative spatial geometry,
    dynamic vertical clustering (averaging bounding box bounds), and multi-column separation.
    """

    def group_tokens_into_lines(
        self,
        tokens: List[OCRText],
        vertical_tol_factor: float = 0.45,
        column_split_gap_ratio: float = 0.25
    ) -> List[LayoutLine]:
        """
        Groups OCR tokens into reading-order lines.

        Args:
            tokens: List of OCRText tokens.
            vertical_tol_factor: Vertical clustering tolerance relative to token height.
            column_split_gap_ratio: Horizontal gap relative to card width to split multi-column rows.

        Returns:
            List of LayoutLine objects sorted vertically from top to bottom.
        """
        if not tokens:
            return []

        # 1. Compute global card boundaries for relative normalization
        all_xs = [pt[0] for t in tokens for pt in t.bbox]
        all_ys = [pt[1] for t in tokens for pt in t.bbox]
        card_w = max(1.0, float(max(all_xs) - min(all_xs)))
        card_h = max(1.0, float(max(all_ys) - min(all_ys)))
        origin_x = float(min(all_xs))
        origin_y = float(min(all_ys))

        # 2. Sort tokens top-to-bottom by center_y
        sorted_tokens = sorted(tokens, key=lambda t: t.center_y)

        # 3. Dynamic Vertical Clustering
        raw_lines: List[List[OCRText]] = []
        line_bounds: List[Tuple[float, float, float]] = []  # (min_y, max_y, avg_h)

        for token in sorted_tokens:
            placed = False
            tok_min_y = min(pt[1] for pt in token.bbox)
            tok_max_y = max(pt[1] for pt in token.bbox)
            tok_h = max(1.0, tok_max_y - tok_min_y)

            for idx, (l_min_y, l_max_y, l_h) in enumerate(line_bounds):
                # Calculate vertical overlap
                overlap = max(0.0, min(tok_max_y, l_max_y) - max(tok_min_y, l_min_y))
                overlap_ratio = overlap / min(tok_h, l_h)

                # Height similarity
                h_diff_ratio = abs(tok_h - l_h) / max(tok_h, l_h)

                # Overlap threshold or close center_y
                if overlap_ratio >= vertical_tol_factor or (overlap > 0 and h_diff_ratio < 0.5):
                    raw_lines[idx].append(token)
                    # Update line bounds without center_y drift
                    new_min_y = min(l_min_y, tok_min_y)
                    new_max_y = max(l_max_y, tok_max_y)
                    new_h = sum(t.height for t in raw_lines[idx]) / len(raw_lines[idx])
                    line_bounds[idx] = (new_min_y, new_max_y, new_h)
                    placed = True
                    break

            if not placed:
                raw_lines.append([token])
                line_bounds.append((tok_min_y, tok_max_y, tok_h))

        # 4. Process Multi-Column rows and format LayoutLines
        layout_lines: List[LayoutLine] = []

        for line_tokens in raw_lines:
            # Sort tokens horizontally left-to-right
            line_tokens_sorted = sorted(line_tokens, key=lambda t: min(pt[0] for pt in t.bbox))

            # Detect large horizontal gap for multi-column split (e.g. back card characteristics vs date)
            split_sublines: List[List[OCRText]] = []
            curr_subline: List[OCRText] = []

            for idx, token in enumerate(line_tokens_sorted):
                if not curr_subline:
                    curr_subline.append(token)
                else:
                    prev_tok = curr_subline[-1]
                    prev_max_x = max(pt[0] for pt in prev_tok.bbox)
                    tok_min_x = min(pt[0] for pt in token.bbox)
                    gap = tok_min_x - prev_max_x

                    if gap > (card_w * column_split_gap_ratio):
                        split_sublines.append(curr_subline)
                        curr_subline = [token]
                    else:
                        curr_subline.append(token)

            if curr_subline:
                split_sublines.append(curr_subline)

            for sub_tokens in split_sublines:
                layout_line = self._construct_layout_line(sub_tokens, origin_x, origin_y, card_w, card_h)
                layout_lines.append(layout_line)

        # 5. Final vertical sort by center_y, then secondary by min_x
        layout_lines.sort(key=lambda l: (l.center_y, l.min_x))
        return layout_lines

    def _construct_layout_line(
        self,
        tokens: List[OCRText],
        origin_x: float,
        origin_y: float,
        card_w: float,
        card_h: float
    ) -> LayoutLine:
        tokens_sorted = sorted(tokens, key=lambda t: min(pt[0] for pt in t.bbox))
        avg_h = sum(t.height for t in tokens_sorted) / float(len(tokens_sorted))

        text_parts: List[str] = []
        for idx, token in enumerate(tokens_sorted):
            if idx == 0:
                text_parts.append(token.text)
            else:
                prev_tok = tokens_sorted[idx - 1]
                tok_min_x = min(pt[0] for pt in token.bbox)
                prev_max_x = max(pt[0] for pt in prev_tok.bbox)
                gap_x = tok_min_x - prev_max_x

                if gap_x > 1.5 or gap_x > (0.10 * avg_h) or not (text_parts[-1].endswith(" ") or token.text.startswith(" ")):
                    text_parts.append(" " + token.text)
                else:
                    text_parts.append(token.text)

        full_text = "".join(text_parts).strip()

        xs = [pt[0] for t in tokens_sorted for pt in t.bbox]
        ys = [pt[1] for t in tokens_sorted for pt in t.bbox]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        avg_y = sum(t.center_y for t in tokens_sorted) / float(len(tokens_sorted))
        avg_conf = sum(t.confidence for t in tokens_sorted) / float(len(tokens_sorted))

        norm_center_y = (avg_y - origin_y) / card_h
        norm_center_x = ((min_x + max_x) / 2.0 - origin_x) / card_w
        norm_min_x = (min_x - origin_x) / card_w
        norm_max_x = (max_x - origin_x) / card_w

        return LayoutLine(
            tokens=tokens_sorted,
            text=full_text,
            center_y=avg_y,
            min_y=min_y,
            max_y=max_y,
            min_x=min_x,
            max_x=max_x,
            height=max_y - min_y,
            confidence=avg_conf,
            norm_center_y=round(norm_center_y, 4),
            norm_center_x=round(norm_center_x, 4),
            norm_min_x=round(norm_min_x, 4),
            norm_max_x=round(norm_max_x, 4)
        )
