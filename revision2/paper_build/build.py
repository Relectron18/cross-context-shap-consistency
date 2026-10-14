import re, zipfile, shutil, sys
from lxml import etree
sys.path.insert(0, ".")
from lib import Doc, NS, W, X
import content_a, content_m, content_r, content_c
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SRC = str(REPO / "paper" / "SHAP_VPN_final_v6_merged.docx")  # v6 with merged runs (template for styles, equations, Figs 1 and 6)
OUT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "paper" / "SHAP_VPN_revised_v7.docx")
z = zipfile.ZipFile(SRC)
doc = etree.fromstring(z.read("word/document.xml"))
body = doc.find(f"{{{W}}}body")
O = list(body)
sect = O[-1]
_ts = list(O[16].iter(f"{{{W}}}t")); _ts[0].text = _ts[0].text + _ts[1].text; _ts[1].text = ""
D = Doc()
content_a.front(D, O); content_a.intro(D, O); content_a.related(D, O); content_a.data(D, O)
content_m.methods(D, O)
content_r.baseline(D); content_r.consistency(D, O); content_r.families(D); content_r.consequence(D); content_r.sensitivity(D)
content_c.summary_discussion(D); content_c.threats(D); content_c.conclusion(D)
# ---- citation renumbering by first appearance
T = f"{{{W}}}t"
CIT = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
def expand(s):
    out = []
    for part in re.split(r"\s*,\s*", s):
        if re.search(r"[–-]", part):
            a, b = [int(x) for x in re.split(r"\s*[–-]\s*", part)]; out += list(range(a, b + 1))
        else: out.append(int(part))
    return out
order = []
for el in D.els:
    for t in el.iter(T):
        for m in CIT.finditer(t.text or ""):
            for n in expand(m.group(1)):
                if n not in order: order.append(n)
mapping = {old: i + 1 for i, old in enumerate(order)}
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]}, {nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ", ".join(out)
for el in D.els:
    for t in el.iter(T):
        if t.text and CIT.search(t.text):
            t.text = CIT.sub(lambda m: "[" + compress([mapping[n] for n in expand(m.group(1))]) + "]", t.text)
# ---- references (original paragraphs 270..326 are [1]..[57])
D.h1("REFERENCES")
refs = {i - 269: O[i] for i in range(270, 327)}
unused = [n for n in refs if n not in mapping]
print("cited", len(mapping), "unused", unused)
for old in order:
    el = refs[old]
    first = next(el.iter(T)); assert first.text.strip() == f"[{old}]", first.text
    first.text = f"[{mapping[old]}]"
    D.add(el)
# ---- assemble body
for ch in list(body): body.remove(ch)
for el in D.els: body.append(el)
body.append(sect)
# ---- package: images and relationships
rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
RNS = "http://schemas.openxmlformats.org/package/2006/relationships"
xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
used = set(re.findall(rb'r:(?:embed|id)="(\w+)"', xml))
drop_media = []
for r in list(rels):
    if r.get("Type").endswith("/image") and r.get("Id").encode() not in used:
        drop_media.append("word/" + r.get("Target")); rels.remove(r)
for rid, path, name in D.images:
    e = etree.SubElement(rels, f"{{{RNS}}}Relationship"); e.set("Id", rid)
    e.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"); e.set("Target", f"media/{name}")
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in z.infolist():
    if item.filename in drop_media: continue
    if item.filename == "word/document.xml": zo.writestr(item, xml)
    elif item.filename == "word/_rels/document.xml.rels": zo.writestr(item, etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True))
    else: zo.writestr(item, z.read(item.filename))
for rid, path, name in D.images: zo.write(path, f"word/media/{name}")
zo.close()
print("tables", D.tab_n, "figures", D.fig_n, "->", OUT)
