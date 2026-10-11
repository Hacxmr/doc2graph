"""High-precision text matches between OCR nodes on different pages."""

import re
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=None)
def _tokens(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(re.findall(r"[^\W_]+(?:[-/][^\W_]+)*", normalized, flags=re.UNICODE))


def _template(text):
    """Text with digit runs masked, so 'Page 3' and 'Page 4' coincide."""
    return " ".join(re.sub(r"\d+", "#", " ".join(_tokens(text))).split())


def _same_slot(src_box, dst_box, dx=0.05, dy=0.02):
    """Whether two page-normalised [x0, y0, x1, y1] boxes sit in the same place."""
    dist_x = abs((src_box[0] + src_box[2]) / 2 - (dst_box[0] + dst_box[2]) / 2)
    dist_y = abs((src_box[1] + src_box[3]) / 2 - (dst_box[1] + dst_box[3]) / 2)
    return dist_x <= dx and dist_y <= dy


def _is_identifier(token):
    """Whether a token looks like a useful document reference (e.g. AB-123)."""
    return any(char.isalpha() for char in token) and any(char.isdigit() for char in token)


def _match(
    src_text, dst_text, src_box=None, dst_box=None, link_page_numbers=False, slot_tolerance=(0.05, 0.02)
):
    src_tokens = _tokens(src_text)
    dst_tokens = _tokens(dst_text)
    src_norm = " ".join(src_tokens)
    dst_norm = " ".join(dst_tokens)
    src_useful = {t for t in src_tokens if len(t) > 2}
    dst_useful = {t for t in dst_tokens if len(t) > 2}

    if src_norm and src_norm == dst_norm and src_useful:
        return 1.0, "same_text"

    shared_ids = {
        token for token in src_useful & dst_useful if _is_identifier(token)
    }
    if shared_ids:
        return 0.9, "shared_identifier"

    # Running headers/footers repeat in the same place with a changing number.
    if src_box is not None and dst_box is not None and _same_slot(src_box, dst_box, *slot_tolerance):
        template = _template(src_text)
        y = (src_box[1] + src_box[3]) / 2
        in_margin = y < 0.12 or y > 0.88  # bare page numbers live in the margins
        if template and template == _template(dst_text):
            # A bare number ("#") says nothing about which document it is from,
            # so it is linked only on request.
            if any(ch.isalpha() for ch in template) or (link_page_numbers and in_margin):
                return 0.8, "repeated_slot"

    shared = src_useful & dst_useful
    if len(shared) >= 2:
        score = len(shared) / len(src_useful | dst_useful)
        if score >= 0.5:
            return float(score), "shared_terms"
    return None


def build_cross_page_links(
    texts_by_page, top_k=3, boxes_by_page=None, link_page_numbers=False, slot_tolerance=(0.05, 0.02)
):
    """Find conservative cross-page identity/continuation candidates.

    The output describes repeated text or shared identifiers. It is not a
    predicted key-value relation. No word list is used: a repeated table heading
    simply links the pages it appears on. When ``boxes_by_page`` (page-normalised [x0, y0, x1, y1]) is given, running
    headers/footers whose number changes between pages are also linked when
    their boxes are within ``slot_tolerance`` (x, y) of each other.
    """
    if len(texts_by_page) < 2 or top_k <= 0:
        return []

    links = []
    for src_page, src_texts in enumerate(texts_by_page):
        for src_node, src_text in enumerate(src_texts):
            matches = []
            for dst_page, dst_texts in enumerate(texts_by_page):
                if dst_page == src_page:
                    continue
                for dst_node, dst_text in enumerate(dst_texts):
                    match = _match(
                        src_text,
                        dst_text,
                        boxes_by_page[src_page][src_node] if boxes_by_page else None,
                        boxes_by_page[dst_page][dst_node] if boxes_by_page else None,
                        link_page_numbers,
                        slot_tolerance,
                    )
                    if match is not None:
                        score, match_type = match
                        matches.append((score, dst_page, dst_node, match_type))

            matches.sort(key=lambda match: (-match[0], match[1], match[2]))
            for score, dst_page, dst_node, match_type in matches[:top_k]:
                # Emit each page-ordered node pair once, avoiding reverse-edge
                # duplicates when both pages contain the same anchor.
                if (src_page, src_node) < (dst_page, dst_node):
                    links.append(
                        {
                            "source": {"page": src_page, "node": src_node},
                            "target": {"page": dst_page, "node": dst_node},
                            "score": score,
                            "match_type": match_type,
                            "page_delta": abs(dst_page - src_page),
                        }
                    )

    # A pair may be selected from both endpoint searches. Keep its strongest
    # match and return deterministic output.
    unique = {}
    for link in links:
        key = (
            link["source"]["page"],
            link["source"]["node"],
            link["target"]["page"],
            link["target"]["node"],
        )
        if key not in unique or link["score"] > unique[key]["score"]:
            unique[key] = link
    return list(unique.values())
