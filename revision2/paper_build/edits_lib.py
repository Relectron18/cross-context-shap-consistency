"""Shared paragraph-editing helpers for fix_v9.py / fix_v10.py (operate on the module-level `body`)."""
import copy
from lxml import etree
import mathfix
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MN = "http://schemas.openxmlformats.org/officeDocument/2006/math"
q = lambda ns, t: f"{{{ns}}}{t}"
body = None
def ptext(p):
    return "".join(e.text or "" for e in p.iter(q(W, "t"), q(MN, "t")))


def find_p(prefix):
    hits = [p for p in body.iter(q(W, "p")) if ptext(p).lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def replace(p, old, new):
    """Replace text inside one paragraph; the match may span several plain runs but not an equation."""
    segs = []  # (t element or None for an equation barrier)
    for ch in p.iter():
        if ch.tag == q(W, "t"): segs.append(ch)
        elif ch.tag == q(MN, "oMath"): segs.append(None)
    pieces, pos = [], 0
    for s in segs:
        txt = "￼" if s is None else (s.text or "")
        pieces.append((s, pos, txt)); pos += len(txt)
    joined = "".join(t for _, _, t in pieces)
    i = joined.find(old)
    assert i >= 0 and joined.find(old, i + 1) < 0, ("not found once", old[:60], ptext(p)[:60])
    j = i + len(old)
    first = True
    for s, start, txt in pieces:
        end = start + len(txt)
        if end <= i or start >= j: continue
        assert s is not None, ("match crosses an equation", old[:60])
        a, b = max(i, start) - start, min(j, end) - start
        if first:
            s.text = txt[:a] + new + txt[b:]; first = False
        else:
            s.text = txt[b:] if a == 0 else txt[:a] + txt[b:]
        s.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def rewrite(p, segs):
    """Replace a paragraph's whole content by plain runs; segs = [text | (text, 'subscript'|'superscript')]."""
    proto = next(p.iter(q(W, "r")), None)
    rpr = proto.find(q(W, "rPr")) if proto is not None else None
    for ch in list(p):
        if ch.tag != q(W, "pPr"): p.remove(ch)
    for s in segs:
        text, va = (s, None) if isinstance(s, str) else s
        r = etree.SubElement(p, q(W, "r"))
        if rpr is not None: r.append(copy.deepcopy(rpr))
        if va:
            pr = r.find(q(W, "rPr"))
            if pr is None: pr = etree.SubElement(r, q(W, "rPr")); r.insert(0, pr)
            v = etree.SubElement(pr, q(W, "vertAlign")); v.set(q(W, "val"), va)
        t = etree.SubElement(r, q(W, "t")); t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def split_token(p, token, segs):
    """Turn a token inside one run into several runs (to create subscripts / superscripts)."""
    for r in p.iter(q(W, "r")):
        t = r.find(q(W, "t"))
        if t is None or token not in (t.text or ""): continue
        before, after = t.text.split(token, 1)
        parent = r.getparent(); idx = parent.index(r)
        new = []
        for s in [before] + segs + [after]:
            text, va = (s, None) if isinstance(s, str) else s
            if not text: continue
            n = copy.deepcopy(r)
            for ch in list(n):
                if ch.tag != q(W, "rPr"): n.remove(ch)
            if va:
                pr = n.find(q(W, "rPr"))
                if pr is None: pr = etree.Element(q(W, "rPr")); n.insert(0, pr)
                v = etree.SubElement(pr, q(W, "vertAlign")); v.set(q(W, "val"), va)
            tt = etree.SubElement(n, q(W, "t")); tt.text = text
            tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            new.append(n)
        parent.remove(r)
        for k, n in enumerate(new): parent.insert(idx + k, n)
        return
    raise AssertionError(("token not found", token))


def merge_next(p):
    """Append the next paragraph's runs to p and delete the next paragraph (repairs a broken line)."""
    nxt = p.getnext()
    for ch in list(nxt):
        if ch.tag != q(W, "pPr"): p.append(ch)
    nxt.getparent().remove(nxt)


def to_math(p, token, sym, sub=None):
    """In the run that contains token, turn the occurrence of sym inside token into an inline equation
    (with an optional subscript that directly follows it in token)."""
    piece = sym + (sub or "")
    sub = sub.lstrip("_") if sub else sub
    for r in p.iter(q(W, "r")):
        t = r.find(q(W, "t"))
        if t is None or token not in (t.text or ""): continue
        k = t.text.index(token) + token.index(piece)
        before, after = t.text[:k], t.text[k + len(piece):]
        parent = r.getparent(); idx = parent.index(r)
        new = []
        if before: new.append(mathfix._clone_run(r, before))
        new.append(mathfix._omath(sym, r, sub=sub))
        if after: new.append(mathfix._clone_run(r, after))
        parent.remove(r)
        for kk, n in enumerate(new): parent.insert(idx + kk, n)
        return
    raise AssertionError(("token not found", token))

