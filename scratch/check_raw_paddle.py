import os
import sys
import cv2
from paddleocr import PaddleOCR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test():
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "image")
    ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)

    for img_name in ["cccd_c_mt.jpg", "cccd_c_ms.jpg", "cccd_m_mt.jpg", "cccd_m_ms.jpg"]:
        p = os.path.join(img_dir, img_name)
        img = cv2.imread(p)
        print(f"\n==================== {img_name} ====================")
        res = ocr.ocr(img, cls=True)
        if res and res[0]:
            for item in res[0]:
                bbox, (text, score) = item
                print(f"[{score:.2f}] {text!r} | len={len(text)} | chars={[c for c in text]}")

if __name__ == "__main__":
    test()
