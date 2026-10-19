"""v12: address the in-project review of v11 (paper/reviews/2026-10-20-review.md), text only.

Every number added here is read from an existing result file (see revision2/results/p7_audit.json,
p7_distinct_lists.json, p3_results.json, p4_results.json, p5_results.json); no experiment is re-run.
New references: Kendall (1945) for tau-b, and the public Deep Packet implementation that supplied the
capture-file prefix list. References are renumbered by first appearance and checked.
Usage: python fix_v12.py <v11 docx> <output docx>
"""
import sys, copy, re, zipfile
from lxml import etree
import mathfix, edits_lib as E
from edits_lib import W, MN, q, ptext, find_p, replace, rewrite

SRC, OUT = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = E.body = doc.find(q(W, "body"))
for e in list(body.iter(q(W, "lastRenderedPageBreak"))): e.getparent().remove(e)
K45, DPC = 1001, 1002   # temporary citation numbers for the two new references

R = [
 # ---- abstract (M2, M5, M6, M7)
 ("Abstract",
  "Keeping related flows in the same half reduces the gap by an amount depending on the proxy for capture sessions: to about zero with clusters of similar flows roughly the size of the capture files, where the half-models are weak, but only to 0.07 and 0.15 with blocks of consecutive flows in the original files, and the gap is non-negative in every repeat, even with first- and second-half splits.",
  "Keeping related flows in the same half reduces the gap by an amount that depends on the proxy for capture sessions: to about zero on average with clusters of similar flows roughly the size of the capture files, where the half-models are weak and opposite gaps in the two contexts cancel, but only to 0.07 and 0.15 with blocks of consecutive flows in the original files, and to 0.03 and 0.09 with first- and second-half splits."),
 ("Abstract", "but a SHAP-chosen Top-10 feature list transfers with little loss.",
  "while a SHAP-chosen Top-10 feature list loses little macro F1 when transferred, in a test of limited power."),
 ("Abstract", "Part of the random-split gap is explained by related flows, and part is likely to remain, but since the two contexts are different sets of captures, none of the differences can be attributed to tunnelling.",
  "Part of the random-split gap is explained by related flows and part probably remains, although no design tests that remainder directly, and since the two contexts are different sets of captures, none of the differences can be attributed to tunnelling."),
 # ---- IV-C: Kendall tau-b (m2)
 ("where dj is the rank difference", " [44],", f" [{K45}],"),
 # ---- IV-D calibration source (m3)
 ("Calibrating the grouping.", "The public list of capture-file prefixes in the Deep Packet code [7] names",
  f"The list of capture-file prefixes in a public, independent implementation of Deep Packet [7, {DPC}] names"),
 # ---- IV-D row-order proxy (M1)
 ("Row-order blocks.",
  "Neighbouring rows are much more alike than random rows of the same class: in every class and context, the nearest of the 40 rows within 20 positions lies at 0.30–0.65 times the distance of the nearest of 40 random rows, and the similarity fades over tens to a few hundred rows. This suggests that the files keep flows from the same capture close together, although their documentation does not say so.",
  "Neighbouring rows are more alike than random rows of the same class, though not equally in every class. Over the deduplicated flows, the nearest of the 40 rows within 20 positions lies at 0.37–0.76 times the distance of the nearest of 40 random rows, so some local structure exists in every class. Adjacent rows, however, are almost as different as random pairs for VPN FT, MAIL, P2P and VOIP (median distance ratio 0.81–0.99, against 0.11–0.77 in the other classes), and within a class the row position is itself correlated with some features (|ρ| up to 0.68). The files therefore seem to keep many flows of one capture close together, but the evidence is weak for those four VPN classes, and the dataset documentation does not describe the row order."),
 ("Row-order blocks.", "and the first-versus-second-half split may also separate different applications within a class.",
  "and the first-versus-second-half split may also separate different applications within a class. BROWSING has no capture files of its own and P2P has a single capture per context, so for these two classes, which hold 34–40% of the training flows, no within-class grouping can separate captures. The row-order blocks are a proxy for related flows, not a capture split, and they do not establish session independence."),
 # ---- IV-D explained records (M4)
 ("XGBoost half-models are explained on the full test partition", "Importance is also computed with true classes weighted equally.",
  "Within-context pairs are thus explained on the same test records, whereas cross-context pairs are explained on the two contexts' different test partitions. Part of the cross-context disagreement may therefore come from the difference in explained records rather than from the models, and none of the designs controls for this. Importance is also computed with true classes weighted equally."),
 # ---- VI-C calibrated clusters (M5)
 ("The calibrated grouping, using approximately", "and half-model macro F1 is 0.53–0.68.",
  "and half-model macro F1 is 0.53–0.68. The near-zero mean also hides opposite signs. For Random Forest, within-Non-VPN agreement exceeds cross-context agreement by 0.228, whereas within-VPN agreement lies 0.193 below it (XGBoost: +0.067 and -0.035). The calibrated mean therefore does not show that the gap disappears in either context; it averages a positive and a negative gap."),
 # ---- VI-C row blocks (M2)
 ("Blocks of consecutive rows in the original files yield", "Under every row-order setting the gap is at or above zero in all 11 repeats for both models.",
  "Under every row-order setting the Spearman gap is at or above zero in all 11 repeats for both models, but this is a weak observation. The repeats are not independent replications: they share one dataset, and in the first-versus-second-half split every repeat uses nearly the same two halves of each class, so only the outer partition and the seeds change. The smallest XGBoost value there is 0.0002, and the RBO and Top-5 gaps are negative in some of these repeats (RBO in 1 of 11 for XGBoost and 2 of 11 for Random Forest, Top-5 in 3 and 2)."),
 # ---- VI-C synthesis (M2, M6, M4)
 ("Taken together, cross-context rankings disagree more",
  "Blocks of consecutive rows divide flows by their position in the files, not by their features, and leave a gap at or above zero in each repeat, even where only one boundary per class cuts through a capture.",
  "Blocks of consecutive rows divide flows by their position in the files rather than directly by their features, and leave a positive mean Spearman gap at every setting."),
 ("Taken together, cross-context rankings disagree more",
  "with 0.03 and 0.09 under the strictest row-order split. Part of the random-split gap therefore reflects variation between groups of related flows, and part of it is retained under grouping by capture order.",
  "with 0.03 and 0.09 under the strictest row-order split. These are descriptive comparisons, not a test of that excess: the permutation null was run only for random halves, where it already gives +0.028 for Random Forest, no grouped design keeps the test records apart from the training sessions, and within-context pairs are explained on shared test records while cross-context pairs are not. Part of the random-split gap therefore reflects variation between groups of related flows, and part of it is retained under grouping by row order."),
 # ---- VI-E permutation importance (M3)
 ("Permutation importance. Permutation importance on the test partitions", "0.326 and 0.267 for XGBoost.",
  "0.326 and 0.267 for XGBoost. These statements concern the minimum IATs. Over the whole ranking, permutation importance agrees less across contexts than SHAP does: cross-context ρ is 0.611 (Random Forest) and 0.719 (XGBoost) for marginal and 0.657 and 0.575 for conditional permutation importance, against 0.830 and 0.786 for SHAP."),
 ("In other words, the minimum IATs are important as a group", "Permutation-based measures show smaller shifts than SHAP for both models,",
  "Permutation-based measures move the minimum IATs less than SHAP does for both models, although their whole rankings differ more between contexts,"),
 # ---- VII families (m1, m4)
 ("When ranked by mean importance per member",
  "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost, larger than the semantic-family gaps, which fits with the three minimum IATs staying in separate clusters.",
  "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost. For XGBoost this is larger than every semantic-family gap (0.064–0.088), which fits with the three minimum IATs staying in separate clusters; for Random Forest it is not (0.096–0.149 under the semantic groupings)."),
 ("These results indicate that part of the XGBoost difference",
  "These results indicate that part of the XGBoost difference between contexts persists at the family level, even though it shrinks at the level of individual features.",
  "These results indicate that, under fine grouping, part of the XGBoost difference between contexts persists at the family level even though it shrinks at the level of individual features; with clusters of about 35 flows or the calibrated grouping, the XGBoost family gap also falls (0.014 and +0.016)."),
 # ---- VIII-B list transfer (M7)
 ("Reusing the other context's feature list costs much less", "with an interval that includes zero.",
  "with an interval that includes zero. The five seeds are not five different lists: they yield one to three distinct Top-k lists per model, context and k, and each distinct list is refitted once with a fixed seed."),
 # ---- IX ranges (m5)
 ("Across the six model and TreeSHAP configurations",
  "0.969–0.984. Extra Trees agrees less well with the path-dependent Random Forest (0.709–0.801).",
  "0.969–0.984 for Random Forest and XGBoost and 0.960–0.970 for Extra Trees. Extra Trees agrees less well with Random Forest (ρ = 0.730–0.801 for the path-dependent and 0.709–0.776 for the interventional formulation)."),
 # ---- X-A (M2, M5, M6)
 ("The answer to the question posed in the title", "but the half-models are weak and the gap varies widely,",
  "but the half-models are weak, the gap varies widely, and for Random Forest the mean averages a gap of +0.23 on the Non-VPN side and -0.19 on the VPN side,"),
 ("The answer to the question posed in the title", "and even the first-versus-second-half split leaves 0.03 and 0.09, at or above zero in every repeat.",
  "and even the first-versus-second-half split leaves 0.03 and 0.09 on average. The Spearman gap stays at or above zero in every repeat, but the repeats reuse nearly the same halves, and the RBO and Top-5 gaps turn negative in some of them."),
 ("The answer to the question posed in the title", "and that some excess probably remains. Without capture identifiers",
  "and that some excess probably remains. That excess is described, not tested: the permutation null covers only random halves, the outer split is random in every design, and within-context and cross-context pairs are explained on different kinds of test records. Without capture identifiers"),
 # ---- X-C (m4)
 ("Aggregation into semantic families reduces the gaps",
  "Summing importance favours larger families, while per-member means and correlation-based clusters give larger gaps.",
  "Summing importance favours larger families. Per-member means give a larger gap than summing for Random Forest but not for XGBoost, and correlation-based clusters give a larger gap for XGBoost but not for Random Forest."),
 # ---- X-D (M7)
 ("The two consequence tests show different tendencies", "For some target contexts, the transferred list performs better than the context-specific list.",
  "For some target contexts, the transferred list performs better than the context-specific list. The test has limited power, however: the selection seeds yield only one to three distinct lists per model, context and k, each list is refitted once, and there is no equivalence margin, so the test rules out a large cost rather than showing that there is none."),
 # ---- XI limitations (M1, M2, M4, M6, M7)
 ("Session dependence and the missing group split",
  "The two proxies give different answers (a gap near zero against a positive gap in every repeat), and the calibration from capture counts is rough. Only the division of training data into halves is grouped; the outer 80/20 split is random.",
  "For VPN FT, MAIL, P2P and VOIP, adjacent rows are nearly as different as random rows, and BROWSING and P2P cannot be split by capture at all. The two proxies give different answers (a mean gap near zero against a positive mean gap), and the calibration from capture counts is rough. Only the division of training data into halves is grouped; the outer 80/20 split is random, and the permutation null was run only for random halves. Within-context pairs are explained on shared test records and cross-context pairs are not; a comparison on a common set of explained records is left for future work."),
 ("Consequence tests. The feature-list test", "The feature-list test uses five inner splits per context and one refit per feature set;",
  "The feature-list test uses five inner splits per context, which yield only one to three distinct lists, and one refit per feature set;"),
 # ---- XII conclusion (M2, M5, M6, M7)
 ("The experiments using consecutive blocks of flows", "The gaps remain non-negative across the tested repetitions.",
  "The Spearman gaps remain non-negative across the tested repetitions, but these repetitions share one dataset and, for the first-versus-second-half split, nearly the same halves, and the RBO and Top-5 gaps are negative in some of them."),
 ("The feature-level analysis also reveals differences", "Matching the selected feature set to its original context provides only a small additional benefit.",
  "Matching the selected feature set to its original context provides only a small additional benefit. Because the selection seeds produced few distinct feature sets, this result has limited power."),
 ("Taken together, these findings suggest",
  "Feature-based clustering reduces the excess gap to approximately zero at a coarse resolution, whereas grouping flows by their position in the original files leaves a positive difference.",
  "Feature-based clustering reduces the average excess gap to approximately zero at a coarse resolution, partly because opposite gaps in the two contexts cancel, whereas grouping flows by their position in the original files leaves a positive average difference. Neither result is a direct test of the remaining excess."),
 ("Taken together, these findings suggest",
  "Further investigation will require datasets that preserve session identifiers or independent captures of both traffic contexts.",
  "Further investigation will require datasets that preserve session identifiers or independent captures of both traffic contexts, and a comparison in which both contexts are explained on a common set of records."),
]
cache = {}
for pre, old, new in R:
    if pre not in cache: cache[pre] = find_p(pre)
    replace(cache[pre], old, new)

# ---- data and code availability section, before the reference list (M8)
refs_head = find_p("REFERENCES")
head = copy.deepcopy(refs_head); rewrite(head, ["DATA AND CODE AVAILABILITY"])
para = copy.deepcopy(find_p("This study examined whether the features"))
rewrite(para, ["The ISCX VPN-nonVPN data, including the Scenario A2 time-based ARFF files used here, are publicly available from the Canadian Institute for Cybersecurity [50]. The analysis code, the deduplicated partitions and the result files behind the tables and figures are kept in a private repository and are available from the corresponding author on reasonable request. The mapping of each flow to its row in the original ARFF files can be regenerated from the public files with that code."])
refs_head.addprevious(head); refs_head.addprevious(para)

# ---- new reference entries (temporary numbers), cloned from existing entries
def new_ref(template_num, text_runs, num):
    tpl = find_p(f"[{template_num}]")
    p = copy.deepcopy(tpl)
    runs = [r for r in p if r.tag == q(W, "r")]
    for r in runs[len(text_runs):]: p.remove(r)
    for r, txt in zip(runs, text_runs):
        for t in r.findall(q(W, "t"))[1:]: r.remove(t)
        r.find(q(W, "t")).text = txt
    p.find(q(W, "r")).find(q(W, "t")).text = f"[{num}]" + text_runs[0]
    tpl.addnext(p)
    return p
new_ref(44, ["M. G. Kendall, “The treatment of ties in ranking problems,” ", "Biometrika",
             ", vol. 33, no. 3, pp. 239-251, 1945, doi: 10.1093/biomet/33.3.239."], K45)
new_ref(50, ["munhouiani, “Deep-Packet: PyTorch implementation of Deep Packet,” GitHub repository. Accessed: Oct. 15, 2026. [Online]. Available: https://github.com/munhouiani/Deep-Packet"], DPC)

# ---- renumber citations by first appearance
paras = list(body.iter(q(W, "p")))
iref = [k for k, p in enumerate(paras) if ptext(p).strip() == "REFERENCES"][0]
text_paras, ref_paras = paras[:iref], paras[iref + 1:]
CIT = re.compile(r"\[(\d+(?:\s*[–-]\s*\d+)?(?:,\s*\d+(?:\s*[–-]\s*\d+)?)*)\]")
def expand(s):
    out = []
    for part in re.split(r",\s*", s):
        a = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
        out += list(range(a[0], a[-1] + 1))
    return out
order = []
for p in text_paras:
    for t in p.iter(q(W, "t")):
        for m in CIT.finditer(t.text or ""):
            for n in expand(m.group(1)):
                if n not in order: order.append(n)
refmap = {}
for p in ref_paras:
    m = re.match(r"\[(\d+)\]", ptext(p))
    if m: refmap[int(m.group(1))] = p
assert set(order) == set(refmap), (sorted(set(refmap) - set(order)), sorted(set(order) - set(refmap)))
new = {old: i + 1 for i, old in enumerate(order)}
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(x) for x in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)
for p in text_paras:
    for t in p.iter(q(W, "t")):
        if t.text and CIT.search(t.text):
            t.text = CIT.sub(lambda m: "[" + compress([new[n] for n in expand(m.group(1))]) + "]", t.text)
anchor = ref_paras[0].getprevious()
for p in refmap.values(): p.getparent().remove(p)
for old in order:
    p = refmap[old]
    t = p.find(q(W, "r")).find(q(W, "t"))
    t.text = re.sub(r"^\[\d+\]", f"[{new[old]}]", t.text)
    anchor.addnext(p); anchor = p
print("references", len(order), "| Kendall 1945 ->", new[K45], "| Deep-Packet code ->", new[DPC])

print("equations added", mathfix.fix([body]))
xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close()
print("->", OUT)
