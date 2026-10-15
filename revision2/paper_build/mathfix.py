"""Typeset mathematical symbols as Word equation objects and normalise characters.

Plain-text Greek letters (rho looks like a Latin 'p', kappa like 'k') and Unicode minus signs are
treated by similarity checkers as characters substituted from another alphabet. Here every such
symbol in running text becomes an inline OMML equation (how Word represents mathematics), minus signs
become ASCII hyphen-minus, arrows become words and no-break spaces become ordinary spaces."""
import copy, re
from lxml import etree
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
q = lambda ns, t: f"{{{ns}}}{t}"
MATH = set("ρτκΔχϕφ≈≥≤′½×√")
PLAIN = {"−": "-", " ": " ", "→": " to ", "‑": "-"}

def _rpr_size(r):
    rpr = r.find(q(W, "rPr"))
    out = etree.Element(q(W, "rPr"))
    f = etree.SubElement(out, q(W, "rFonts")); f.set(q(W, "ascii"), "Cambria Math"); f.set(q(W, "hAnsi"), "Cambria Math")
    if rpr is not None:
        for tag in ("sz", "szCs"):
            e = rpr.find(q(W, tag))
            if e is not None: out.append(copy.deepcopy(e))
    return out

def _mrun(text, r):
    mr = etree.Element(q(MNS, "r")); mr.append(_rpr_size(r))
    t = etree.SubElement(mr, q(MNS, "t")); t.text = text
    return mr

def _omath(sym, r, sub=None, sup=None):
    om = etree.Element(q(MNS, "oMath"))
    if sub is None and sup is None:
        om.append(_mrun(sym, r)); return om
    kind = "sSub" if sub is not None else "sSup"
    s = etree.SubElement(om, q(MNS, kind)); e = etree.SubElement(s, q(MNS, "e")); e.append(_mrun(sym, r))
    x = etree.SubElement(s, q(MNS, "sub" if sub is not None else "sup")); x.append(_mrun(sub if sub is not None else sup, r))
    return om

def _valign(r):
    rpr = r.find(q(W, "rPr"))
    if rpr is None: return None
    v = rpr.find(q(W, "vertAlign"))
    return None if v is None else v.get(q(W, "val"))

def _clone_run(r, text):
    n = copy.deepcopy(r)
    for ch in list(n):
        if ch.tag != q(W, "rPr"): n.remove(ch)
    t = etree.SubElement(n, q(W, "t")); t.text = text; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return n

def fix(body_elements):
    nmath = 0
    for el in body_elements:
        for t in el.iter(q(W, "t")):
            if t.text:
                for a, b in PLAIN.items(): t.text = t.text.replace(a, b)
        for p in list(el.iter(q(W, "p"))):
            kids = list(p)
            i = 0
            while i < len(kids):
                r = kids[i]
                if r.tag != q(W, "r") or r.find(q(W, "t")) is None or not any(c in MATH for c in (r.find(q(W, "t")).text or "")):
                    i += 1; continue
                if len(r.findall(q(W, "t"))) != 1 or len([c for c in r if c.tag not in (q(W, "rPr"), q(W, "t"))]):
                    i += 1; continue
                text = r.find(q(W, "t")).text
                parts = re.split("([" + re.escape("".join(sorted(MATH))) + "])", text)
                new = []
                consumed_next = False
                for j, seg in enumerate(parts):
                    if not seg: continue
                    if seg in MATH:
                        is_last = all(not s for s in parts[j + 1:])
                        nxt = kids[i + 1] if i + 1 < len(kids) else None
                        va = _valign(nxt) if (is_last and nxt is not None and nxt.tag == q(W, "r")) else None
                        if va in ("subscript", "superscript") and nxt.find(q(W, "t")) is not None:
                            s_text = nxt.find(q(W, "t")).text
                            new.append(_omath(seg, r, sub=s_text if va == "subscript" else None, sup=s_text if va == "superscript" else None))
                            consumed_next = True
                        else:
                            new.append(_omath(seg, r))
                        nmath += 1
                    else:
                        new.append(_clone_run(r, seg))
                pos = p.index(r)
                p.remove(r)
                if consumed_next: p.remove(kids[i + 1])
                for k, n in enumerate(new): p.insert(pos + k, n)
                kids = list(p); i = pos + len(new)
    return nmath
