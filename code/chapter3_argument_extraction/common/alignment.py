def find_occurrences(text, span_text):
    positions = []
    start = text.find(span_text)
    while start != -1:
        positions.append((start, start + len(span_text)))
        start = text.find(span_text, start + 1)
    return positions


def align_nearest(text, span_text, anchor):
    occ = find_occurrences(text, span_text)
    if not occ:
        return None
    return min(occ, key=lambda p: min(abs(p[0] - anchor), abs(p[1] - anchor)))


def align_with_used(text, span_text, anchor, used):
    occ = [p for p in find_occurrences(text, span_text) if p not in used]
    if not occ:
        return None
    best = min(occ, key=lambda p: min(abs(p[0] - anchor), abs(p[1] - anchor)))
    used.add(best)
    return best
