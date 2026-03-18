# =============================================================================
# metadata.py — Tier 1 signals: image_urls, image_count, has_dead_images
# =============================================================================
import re
from config import S3_URL_PATTERN


def extract_image_urls(ocr_fields: dict) -> int:
    """Count ALL S3 URLs — includes plain OCR scan URLs."""
    raw = str(ocr_fields)
    urls = re.findall(S3_URL_PATTERN, raw, re.IGNORECASE)
    return len(urls)


def extract_diagram_url_count(ocr_fields: dict) -> int:
    """Count only URLs that look like actual diagram/figure images.
    Diagram URLs end in .png/.jpg/.jpeg or contain 'image' in filename.
    Plain OCR scan URLs are random strings with no extension.
    """
    raw = str(ocr_fields)
    urls = re.findall(S3_URL_PATTERN, raw, re.IGNORECASE)
    diagram_urls = [
        u for u in urls
        if re.search(r'\.(png|jpg|jpeg|gif|svg|webp)', u, re.IGNORECASE)
        or re.search(r'image', u.split('/')[-1], re.IGNORECASE)
    ]
    return len(diagram_urls)


def extract_image_count(ocr_fields: dict) -> int:
    """Sum image_count across all fields (question + solution)."""
    count = 0
    if isinstance(ocr_fields, dict):
        for val in ocr_fields.values():
            if isinstance(val, dict):
                count += int(val.get("image_count", 0) or 0)
    return count


def extract_has_dead_images(has_dead_images) -> bool:
    """Return True if has_dead_images flag is set."""
    return bool(has_dead_images)
