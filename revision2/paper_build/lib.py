"""Minimal OOXML builder that reuses the formatting of the original manuscript."""
import re, copy, zipfile, shutil
from xml.sax.saxutils import escape
from lxml import etree
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W, "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "w14": "http://schemas.microsoft.com/office/word/2010/wordml"}
DECL = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
def X(s):
    return etree.fromstring(f'<root {DECL}>{s}</root>')[0]

TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\^\{.+?\}|_\{.+?\})")
def runs(text, size=None, bold=False, italic=False):
    out = []
    for part in TOKEN.split(text):
        if not part: continue
        b, i, va = bold, italic, None
        if part.startswith("**"): part, b = part[2:-2], True
        elif part.startswith("*"): part, i = part[1:-1], True
        elif part.startswith("^{"): part, va = part[2:-1], "superscript"
        elif part.startswith("_{"): part, va = part[2:-1], "subscript"
        rpr = ""
        if b: rpr += "<w:b/><w:bCs/>"
        if i: rpr += "<w:i/><w:iCs/>"
        if va: rpr += f'<w:vertAlign w:val="{va}"/>'
        if size: rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)

class Doc:
    def __init__(self):
        self.els = []; self.tab_n = 0; self.fig_n = 0; self.images = []
    def add(self, el): self.els.append(el)
    def raw(self, xml): self.add(X(xml))
    def h1(self, t):
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:keepNext/><w:spacing w:before="280" w:after="120"/><w:jc w:val="center"/></w:pPr>{runs(t)}</w:p>')
    def h2(self, t):
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:keepNext/><w:spacing w:before="180" w:after="80"/></w:pPr>{runs(t)}</w:p>')
    def p(self, t, indent=True):
        ind = '<w:ind w:firstLine="284"/>' if indent else ""
        self.raw(f'<w:p><w:pPr><w:spacing w:after="100" w:line="264" w:lineRule="auto"/>{ind}<w:jc w:val="both"/></w:pPr>{runs(t)}</w:p>')
    def lim(self, head, t):
        self.raw(f'<w:p><w:pPr><w:spacing w:after="100" w:line="264" w:lineRule="auto"/><w:ind w:firstLine="284"/><w:jc w:val="both"/></w:pPr>{runs("**" + head + "** " + t)}</w:p>')
    def eq_text(self, t, num):
        self.raw(f'<w:p><w:pPr><w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs><w:spacing w:before="100" w:after="100"/></w:pPr><w:r><w:tab/></w:r>{runs(t)}<w:r><w:tab/><w:t>({num})</w:t></w:r></w:p>')
    def table(self, title, widths, rows, note=None, header_rows=1, label=None, align=None):
        self.tab_n += 1
        roman = to_roman(self.tab_n)
        self.raw(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t>TABLE {roman}</w:t></w:r></w:p>')
        self.raw(f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="80" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{runs(title, size=20)}</w:p>')
        tw = sum(widths)
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        trs = []
        for ri, row in enumerate(rows):
            hdr = ri < header_rows
            sep = isinstance(row, dict)  # section row spanning all columns
            tcs = []
            if sep:
                tcs.append(self._tc(row["section"], tw, True, span=len(widths), italic=True, left=True))
            else:
                ci = 0
                for cell in row:
                    span = 1
                    if isinstance(cell, tuple): cell, span = cell
                    w = sum(widths[ci:ci + span])
                    tcs.append(self._tc(str(cell), w, hdr, span=span, left=(ci == 0 if align is None else align[ci] == "l")))
                    ci += span
            trpr = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if hdr else '') + '<w:jc w:val="center"/></w:trPr>'
            trs.append(f'<w:tr>{trpr}{"".join(tcs)}</w:tr>')
        self.raw(f'<w:tbl><w:tblPr><w:tblW w:w="{tw}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="22" w:type="dxa"/><w:left w:w="55" w:type="dxa"/><w:bottom w:w="22" w:type="dxa"/><w:right w:w="55" w:type="dxa"/></w:tblCellMar><w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
        if note:
            self.raw(f'<w:p><w:pPr><w:keepLines/><w:spacing w:before="40" w:after="160" w:line="230" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:t xml:space="preserve">Note. </w:t></w:r>{runs(note, size=17)}</w:p>')
        else:
            self.raw('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>')
        return roman
    def _tc(self, text, w, hdr, span=1, italic=False, left=True):
        gs = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
        jc = "" if left else '<w:jc w:val="center"/>'
        paras = "".join(f'<w:p><w:pPr><w:spacing w:after="0" w:line="228" w:lineRule="auto"/>{jc}</w:pPr>{runs(line, size=17, bold=hdr, italic=italic)}</w:p>' for line in text.split("\n"))
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{gs}<w:tcBorders><w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders><w:vAlign w:val="center"/></w:tcPr>{paras}</w:tc>')
    def figure(self, path, caption, width_in=6.25):
        from PIL import Image
        self.fig_n += 1
        im = Image.open(path); wpx, hpx = im.size
        cx = int(width_in * 914400); cy = int(cx * hpx / wpx)
        rid = f"rIdRev2Fig{self.fig_n}"; n = 900 + self.fig_n
        self.images.append((rid, path, f"rev2_fig{self.fig_n}.png"))
        self.raw(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{n}" name="Figure {self.fig_n}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="Figure {self.fig_n}"/><pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        self.raw(f'<w:p><w:pPr><w:keepLines/><w:spacing w:after="200" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">Fig. {self.fig_n}. </w:t></w:r>{runs(caption, size=20)}</w:p>')
        return self.fig_n

def to_roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in vals:
        while n >= v: s += r; n -= v
    return s

def _eq_omml(self, inner, num):
    self.raw('<w:p><w:pPr><w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs><w:spacing w:before="100" w:after="100"/></w:pPr>'
             f'<w:r><w:tab/></w:r><m:oMath>{inner}</m:oMath><w:r><w:tab/><w:t>({num})</w:t></w:r></w:p>')
Doc.eq_omml = _eq_omml
