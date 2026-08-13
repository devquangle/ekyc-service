from typing import List, Optional
from pydantic import BaseModel
from ocr.detector import OCRText
from utils.logger import logger


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
    height, and bounding box overlap.
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
            full_text = " ".join([t.text for t in line_tokens_sorted])

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
