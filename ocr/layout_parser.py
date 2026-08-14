from typing import List, Optional
from pydantic import BaseModel
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


class LayoutParser:
    """
    Groups spatial OCRText tokens into horizontal LayoutLines based on center_y,
    height, and bounding box overlap with gap-aware word spacing.
    """

    def group_tokens_into_lines(
        self, tokens: List[OCRText], vertical_tol_factor: float = 0.5
    ) -> List[LayoutLine]:
        if not tokens:
            return []

        # Sort tokens primarily by center_y
        sorted_tokens = sorted(tokens, key=lambda t: t.center_y)

        lines: List[List[OCRText]] = []

        for token in sorted_tokens:
            placed = False
            for line in lines:
                line_avg_y = sum(t.center_y for t in line) / float(len(line))
                line_avg_h = sum(t.height for t in line) / float(len(line))
                tolerance = max(token.height, line_avg_h) * vertical_tol_factor

                if abs(token.center_y - line_avg_y) <= tolerance:
                    line.append(token)
                    placed = True
                    break

            if not placed:
                lines.append([token])

        layout_lines: List[LayoutLine] = []

        for line_tokens in lines:
            # Sort tokens in line left to right by center_x
            line_tokens_sorted = sorted(line_tokens, key=lambda t: t.center_x)

            avg_height = sum(t.height for t in line_tokens_sorted) / float(len(line_tokens_sorted))

            text_parts = []
            for idx, token in enumerate(line_tokens_sorted):
                if idx == 0:
                    text_parts.append(token.text)
                else:
                    prev_token = line_tokens_sorted[idx - 1]
                    token_min_x = min(pt[0] for pt in token.bbox)
                    prev_max_x = max(pt[0] for pt in prev_token.bbox)
                    gap_x = token_min_x - prev_max_x

                    # Mandatory space insertion if gap_x > 1.5 or gap_x > 0.1 * avg_height or tokens lack whitespace
                    if gap_x > 1.5 or gap_x > 0.1 * avg_height or not (text_parts[-1].endswith(" ") or token.text.startswith(" ")):
                        text_parts.append(" " + token.text)
                    else:
                        text_parts.append(token.text)

            full_text = "".join(text_parts).strip()

            xs = [t.bbox[i][0] for t in line_tokens_sorted for i in range(len(t.bbox))]
            ys = [t.bbox[i][1] for t in line_tokens_sorted for i in range(len(t.bbox))]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            avg_y = sum(t.center_y for t in line_tokens_sorted) / float(len(line_tokens_sorted))
            avg_conf = sum(t.confidence for t in line_tokens_sorted) / float(len(line_tokens_sorted))

            layout_lines.append(
                LayoutLine(
                    tokens=line_tokens_sorted,
                    text=full_text,
                    center_y=avg_y,
                    min_y=min_y,
                    max_y=max_y,
                    min_x=min_x,
                    max_x=max_x,
                    height=max_y - min_y,
                    confidence=avg_conf
                )
            )

        # Sort final layout lines top to bottom
        layout_lines = sorted(layout_lines, key=lambda l: l.center_y)
        return layout_lines
