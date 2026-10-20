"""v16 = v15 with one sentence reworded: "An earlier version without the per-file split was discarded ..." read as
revision history, which Concerns 4 (minor point 1) asked to keep out of the manuscript. It becomes a methods statement.
Usage: python fix_v16.py <v15 docx> <output docx>
"""
import sys, zipfile
from lxml import etree
import edits_lib as E
from edits_lib import W, q, ptext, replace
SRC, OUT = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC); doc = etree.fromstring(zin.read("word/document.xml")); body = E.body = doc.find(q(W, "body"))
old = "An earlier version without the per-file split was discarded because its two halves could hold very different shares of Non-VPN and VPN rows."
new = "Without the per-file split, the two halves of a pseudo-context could hold very different shares of Non-VPN and VPN rows."
hits = [p for p in body.iter(q(W, "p")) if old in ptext(p)]; assert len(hits) == 1
replace(hits[0], old, new)
xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist(): zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close(); print("->", OUT)
