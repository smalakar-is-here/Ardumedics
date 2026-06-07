#!/usr/bin/env python3
"""
ArduMedics OCR — Prescription Analysis on Raspberry Pi 5

Runs Multi-Engine OCR Fusion on prescription images.

Usage:
    python3 run_ocr.py --image prescription.jpg
    python3 run_ocr.py --camera 0 --realtime
"""

import argparse
import sys
import time

def run_ocr(image_path, engines=['tesseract', 'easyocr', 'paddleocr'], use_fusion=True):
    """
    Run OCR on a prescription image using multiple engines with fusion.

    Args:
        image_path: Path to the prescription image
        engines: List of OCR engines to use
        use_fusion: Whether to apply confidence-weighted fusion

    Returns:
        dict with extracted text from each engine + fused result
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image {image_path}")
        return None

    results = {}

    # Tesseract
    if 'tesseract' in engines:
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            results['tesseract'] = {'text': text, 'confidence': 0.7}
            print(f"[Tesseract] {text[:100]}...")
        except ImportError:
            print("[WARNING] Tesseract not installed. Run: sudo apt install tesseract-ocr && pip install pytesseract")

    # EasyOCR
    if 'easyocr' in engines:
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            detections = reader.readtext(img)
            text = ' '.join([det[1] for det in detections])
            avg_conf = sum([det[2] for det in detections]) / max(len(detections), 1)
            results['easyocr'] = {'text': text, 'confidence': avg_conf}
            print(f"[EasyOCR] {text[:100]}... (conf: {avg_conf:.2f})")
        except ImportError:
            print("[WARNING] EasyOCR not installed. Run: pip install easyocr")

    # PaddleOCR
    if 'paddleocr' in engines:
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            detections = ocr.ocr(img, cls=True)
            text = ' '.join([line[1][0] for line in detections[0]]) if detections[0] else ''
            avg_conf = sum([line[1][1] for line in detections[0]]) / max(len(detections[0]), 1) if detections[0] else 0
            results['paddleocr'] = {'text': text, 'confidence': avg_conf}
            print(f"[PaddleOCR] {text[:100]}... (conf: {avg_conf:.2f})")
        except ImportError:
            print("[WARNING] PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")

    # Fusion
    if use_fusion and len(results) > 1:
        fused = confidence_weighted_fusion(results)
        results['fusion'] = fused
        print(f"\n[FUSION] {fused['text'][:100]}... (conf: {fused['confidence']:.2f})")

    return results


def confidence_weighted_fusion(results):
    """
    Apply confidence-weighted fusion across OCR engine results.
    Engines with higher confidence get more weight in the final text.
    """
    total_conf = sum(r['confidence'] for r in results.values())
    if total_conf == 0:
        # Equal weight if no confidence
        weights = {k: 1.0/len(results) for k in results}
    else:
        weights = {k: r['confidence']/total_conf for k, r in results.items()}

    # Simple fusion: return highest-confidence result
    # (More sophisticated fusion would align text segments)
    best_engine = max(results, key=lambda k: results[k]['confidence'])
    fused_text = results[best_engine]['text']
    fused_conf = results[best_engine]['confidence']

    return {
        'text': fused_text,
        'confidence': fused_conf,
        'weights': weights,
        'best_engine': best_engine
    }


def main():
    parser = argparse.ArgumentParser(description='ArduMedics OCR Prescription Analysis')
    parser.add_argument('--image', type=str, help='Path to prescription image')
    parser.add_argument('--camera', type=int, help='Camera index for realtime OCR')
    parser.add_argument('--engines', nargs='+', default=['tesseract', 'easyocr', 'paddleocr'],
                       help='OCR engines to use')
    parser.add_argument('--no-fusion', action='store_true', help='Disable fusion')

    args = parser.parse_args()

    if args.image:
        results = run_ocr(args.image, engines=args.engines, use_fusion=not args.no_fusion)
    elif args.camera is not None:
        import cv2
        cap = cv2.VideoCapture(args.camera)
        print("Press SPACE to capture, 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('ArduMedics OCR', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                temp_path = '/tmp/ocr_capture.jpg'
                cv2.imwrite(temp_path, frame)
                print("\n--- Capturing and running OCR ---")
                run_ocr(temp_path, engines=args.engines, use_fusion=not args.no_fusion)
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("ERROR: Provide --image or --camera argument")
        parser.print_help()


if __name__ == '__main__':
    main()
