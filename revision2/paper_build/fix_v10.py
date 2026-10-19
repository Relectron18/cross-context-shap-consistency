"""Correct the second team-edited copy of v9 (eb2da454-SHAP_VPN_revised_v9.docx) -> v10.

The team reworded twelve more passages of v9, deleted the body of Section III-D and turned every en dash
into a hyphen. This script keeps their new wording and repairs what broke: the deleted section, a garbled
conclusion, four lost equation symbols, factual slips, a misplaced sentence, dashes and spacing. Numbers are
unchanged. Usage: python fix_v10.py <team docx> <output docx>
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


def new_para_after(anchor, template, segs):
    p = copy.deepcopy(template)
    anchor.addnext(p)
    rewrite(p, segs)
    return p


# ---------------- 1. Section III-D: both paragraphs had been deleted (Table III was left unreferenced)
head = find_p("D. Models and Implementation")
tmpl = find_p("BROWSING is the lone exception")
p1 = new_para_after(head, tmpl, [
    "Two tree ensembles are used because they handle correlated inputs differently. Random Forest [51] averages many deep trees, and each split picks from a random subset of features. XGBoost [52] grows shallower trees one after another, with regularisation. Both are explained with TreeSHAP, and Extra Trees [53] appears only in the robustness checks of Section IX. We did not keep a linear baseline: multinomial Logistic Regression, even after a log(x + 2) transform, reached only 0.754 (Non-VPN) and 0.724 (VPN) macro F1, against 0.85-0.93 for the ensembles."])
new_para_after(p1, tmpl, [
    "Table III lists the configurations, which are the same in both contexts. Hyperparameters were set once and not tuned; Section IX shows how far the rankings move under other XGBoost settings and seeds. The primary models use random_state 42, and every model in the repeated designs of Section IV-D has its own seed. The code uses scikit-learn [54] and XGBoost [52], with TreeSHAP from the SHAP library [14], and all results come from one software environment (Table III)."])

# ---------------- 2. text repairs inside the team's wording
R = [
 ("Abstract", "Abstract- ", "Abstract—"),
 ("Explanation capability is being required", "[11, 25-27].Warnecke", "[11, 25-27]. Warnecke"),
 ("The selection process in the feature list transfer", "is separated into an 80% inner-training", "is separated into a stratified 80% inner-training"),
 ("The selection process in the feature list transfer", "to compute importance), the k highest important features are kept.", "to compute importance), and the k most important features are kept."),
 ("There are two sets of control selections",
  "context and the 20 random selections of 10 features. For the first selection seed with k = 10, 3,000 paired bootstrap resamples of test records are performed. These percentile intervals fix the fitted models and ignore the uncertainty associated with training and selection.",
  "context, and 20 random selections of 10 features, drawn once and used for every model and target. For the first selection seed with k = 10, 3,000 class-stratified paired bootstrap resamples of the test records give percentile intervals for the own-minus-transferred difference. These intervals fix the fitted models and ignore the uncertainty associated with training and selection, which the five seeds show separately."),
 ("For Random Forest, the Spearman", "which means that one sixth of the total significance gets transferred,", "which means that about one sixth of the total importance changes hands,"),
 ("For Random Forest, the Spearman",
  " = 0.65).  Equal weighting of the true classes does not affect the ranking within contexts significantly (between the weighted and unweighted ranking 0.976-0.998), but the cross-context values remain at the same level.",
  " = 0.65). Equal weighting of the true classes changes the rankings within contexts very little (ρ between the weighted and unweighted rankings 0.976-0.998), and the cross-context values stay at about the same level."),
 ("Some of the movers are correlated",
  "credit.  A model can move credit within such a group",
  "credit. Nearly redundant pairs are duration versus total_fiat and total_biat (Spearman’s correlation 0.89-0.95), total_fiat versus total_biat (0.84-0.90), and the two throughput features (0.94). A model can move credit within such a group"),
 ("The permutation null shows that the gap",
  "or class-mix differential alone. The allocation of flows randomly between two fake contexts with the same Non-VPN and VPN class mixes results in a   of",
  "or the class-mix difference alone. Allocating flows at random between two pseudo-contexts with the same Non-VPN and VPN class mixes gives a Δ of"),
 ("Entries below -1",
  "attribute of the Random Forest models (1.0% and 0.6% of XGBoost), and when omitted from the training, leave the rankings virtually unaltered ( with the main ranking 0.986-0.997), and are not responsible for minimum-IAT changes.",
  "attribution of the Random Forest models (1.0% and 0.6% for XGBoost). When they are omitted from training, the rankings stay virtually unaltered (ρ with the main ranking 0.986-0.997), so they are not responsible for the minimum-IAT changes."),
 ("In other words, the minimum IATs", "the minimum IATs are important as a class.", "the minimum IATs are important as a group."),
 ("An explanation report for a traffic classifier", "drawn from the target context.",
  "drawn from the target context. Similar explanations alone do not guarantee that a deployed model will perform reliably in a different observation context."),
]
cache = {}
for pre, old, new in R:
    if pre not in cache: cache[pre] = find_p(pre)
    replace(cache[pre], old, new)

# the list of near-redundant pairs had been moved below the sentence that refers to it
dup = find_p("Redundant pairs of features are:")
dup.getparent().remove(dup)

# ---------------- 3. conclusion: three garbled paragraphs rebuilt in the team's wording (plus the lost last one)
c2 = find_p("Both models degrade under VPN")
c3 = find_p("The question is whether the difference")
c4 = find_p("Using the original files")
rewrite(c2, ["Both models degrade under VPN (macro F1 of XGBoost from 0.9340 to 0.8836, Random Forest from 0.9171 to 0.8533), and a model trained under one context gets only 0.43-0.61 macro F1 under the other. SHAP importance values in different contexts are correlated but not the same: Spearman’s rank correlation is 0.83 and 0.79, RBO is 0.58 and 0.75, and both models retain three of their five most important features."])
rewrite(c3, ["Whether the difference comes from the context depends on the reference it is compared with. Compared to randomly selected halves of the same context, the feature-level gap is 0.13 for XGBoost and 0.19 for Random Forest on average across 11 repeats with different halves, and a permutation null that preserves both class distributions produces little or no gap. Keeping related traffic together reduces the gap. With clusters of similar flows it is 0.08-0.10 with about 10 flows per cluster and 0.01-0.04 with about 35, and its average is close to zero at the resolution suggested by the capture statistics, where the half-models are weak."])
rewrite(c4, ["With blocks of sequential flows from the original files, which do not use any features, the gap is 0.07 and 0.15 at that resolution and 0.03 and 0.09 when each class is split into its first and second halves, and it is at least zero every time. The top-weighted RBO gap and the XGBoost family-level gap shrink less than the Spearman gap under clusters. The mix of classes matters little. The most evident change, the rise of the minimum IATs, is mostly a re-allocation of Shapley values for XGBoost, whereas for Random Forest the block of minimum IATs also gains importance. A SHAP-selected feature set moves between contexts at a loss of at most about 0.013 macro F1 on average across selection seeds, for sets of ten or more features, and the effect of matching a set to its context is small."])
new_para_after(c4, c4, ["On this benchmark, then, part of the random-split gap between SHAP rankings reflects variation between groups of related flows within a context, and how much remains depends on how related flows are grouped: about none under coarse feature clusters, and a positive excess under grouping by position in the original files. None of the differences can be attributed to tunnelling, and the proxy experiments here cannot settle how large a true between-session effect is. Settling the question requires captures that keep session identifiers or a second independent capture of both contexts, more recent traffic, and class-specific explanations."])

# ---------------- 4. en dashes for numeric and citation ranges in the main text (references keep hyphens)
RANGE = re.compile(r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?!\d)")
nd = 0
for p in body.iter(q(W, "p")):
    if ptext(p).strip() == "REFERENCES": break
    for t in p.iter(q(W, "t")):
        if t.text:
            t.text, k = RANGE.subn(r"\1–\2", t.text); nd += k
print("ranges with en dash", nd)

# ---------------- 5. symbols inside equations, then every plain-text symbol becomes an equation
for t in find_p("Spearman and Kendall weight a swap").iter(q(MN, "t")):
    if t.text == "'": t.text = "′"
n = mathfix.fix([body])
for om in body.iter(q(MN, "oMath")):
    if "within,NV" in "".join(t.text or "" for t in om.iter(q(MN, "t"))):
        for t in om.iter(q(MN, "t")):
            if t.text == "-": t.text = "−"
print("equations added", n)

xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close()
print("->", OUT)
