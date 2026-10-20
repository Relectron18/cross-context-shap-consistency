"""v15 = the team-edited v14 (paper/SHAP_VPN_v14_team_edited.docx, 20 Oct 2026) with its three edits repaired.
The team changed three passages of v14: a sentence in II-C (subject lost, ungrammatical), the X-E paragraph (rewritten in a
conversational register with a garbled phrase) and the spacing in reference [55]. Everything else in their file is kept.
Usage: python fix_v15.py <team-edited v14 docx> <output docx>
"""
import sys, zipfile
from lxml import etree
import mathfix, edits_lib as E
from edits_lib import W, q, ptext, replace, rewrite
SRC, OUT = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC); doc = etree.fromstring(zin.read("word/document.xml"))
body = E.body = doc.find(q(W, "body"))
def find_c(s):
    hits = [p for p in body.iter(q(W, "p")) if s in ptext(p)]; assert len(hits) == 1, (s[:60], len(hits)); return hits[0]
rep = lambda old, new: replace(find_c(old), old, new)
rep("Although, the datasets belong to different networks (and therefore different attack mixes and labels) therefore cannot be attributed to a single change in the observation conditions.",
    "Those datasets, however, come from different networks with different attack mixes and labels, so a change in explanation cannot be traced to a single change in observation conditions.")
p = find_c("In terms of comparing two sets of rankings:")
old = ptext(p); i = old.index("In terms of comparing two sets of rankings:"); j = old.index("Similar explanations alone do not guarantee")
rewrite(p, [old[:i] + "When two sets of rankings are compared, it helps to know how much of the difference comes only from explaining the models on different records: under random halves this accounts for about 21–28% of the Random Forest gap here and for almost none of the XGBoost gap. Explaining a model on the other context's records evaluates it outside its own data, so this is a diagnostic, not a better ranking. When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context. " + old[j:]])
rep("in ranking problems,”Biometrika,", "in ranking problems,” Biometrika,")
print("equations added", mathfix.fix([body]))
xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist(): zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close(); print("->", OUT)
