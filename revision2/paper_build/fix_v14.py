"""v14 = v13 + (A) an editorial pass over the whole manuscript + (B) two new analyses.

(A) Editorial pass: sentence- or paragraph-level rewording where the prose was awkward, generic or repetitive
    (mostly passages from the team edits in II, III, VI-E, VII, X and XII). Every edit goes through `guard`,
    which compares the numbers and citation brackets of the old and new text and stops the build if they
    differ, unless the edit is declared in ALLOW with a reason. Equations are never inside an edited span,
    except in paragraphs that are rewritten whole, where mathfix turns symbols back into equations.
(B) New results (common explained records; permutation null with grouped halves), read from
    results/p11_v14.json, which src/p11_v14_summary.py writes from the per-repeat result files.

Usage: python fix_v14.py <v13 docx> <output docx>
"""
import sys, copy, json, re, zipfile
from collections import Counter
from pathlib import Path
from lxml import etree
import mathfix, edits_lib as E
from edits_lib import W, MN, q, ptext, replace, rewrite

SRC, OUT = sys.argv[1], sys.argv[2]
RES = Path(__file__).resolve().parents[1] / "results"
zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = E.body = doc.find(q(W, "body"))
for e in list(body.iter(q(W, "lastRenderedPageBreak"))): e.getparent().remove(e)

TOK = re.compile(r"\[[^\]]*\d[^\]]*\]|\d+(?:[.,]\d+)*%?")
def tokens(s): return Counter(TOK.findall(s.replace("−", "-")))
LOG = []
def guard(old, new, allow=None):
    a, b = tokens(old), tokens(new)
    if a != b:
        assert allow, ("numbers or citations changed in a stylistic edit", old[:80], dict(a - b), dict(b - a))
        LOG.append((allow, dict(a - b), dict(b - a)))

def find_c(s):
    hits = [p for p in body.iter(q(W, "p")) if s in ptext(p)]
    assert len(hits) == 1, ("paragraph not unique", s[:70], len(hits))
    return hits[0]

def rep(old, new, allow=None, where=None):
    guard(old, new, allow); replace(find_c(where or old), old, new)

def rew(prefix, new, allow=None):
    p = find_c(prefix); old = ptext(p)
    guard(old, new, allow); rewrite(p, [new])

# =====================================================================================  (A) editorial pass
# ---- I. Introduction
rep("Shapley values deal badly with correlated features: credit is apportioned according to the fitted model, not the data distribution [18–20], and flow-based traffic has heavily-correlated features. The minimum, mean, maximum and total inter-arrival times of a flow change together. When two similar models rank them differently, this could be due to an actual difference in modeling, or simply due to training randomness.",
    "Correlated features are a known weak point of Shapley values: credit is divided according to the fitted model, not the data distribution [18–20], and flow features are strongly correlated, since the minimum, mean, maximum and total inter-arrival times of a flow tend to move together. When two similar models rank such features differently, the cause may be a real difference in what the models learned or simply training randomness.")

# ---- Abstract
rep("Compared to models trained on random halves of one context,", "Compared with models trained on random halves of one context,")
rep("A model trained in one context achieves at best 0.43–0.61 macro F1 in the other,", "A model trained in one context reaches only 0.43–0.61 macro F1 in the other,")
rep("Each model keeps only three of its five leading features when the observation context changes.", "Each model keeps only three of its five leading features across contexts.")

# ---- II-A to II-E
rew("As encryption became widespread, traffic classification gradually moved",
    "As encryption spread, traffic classification moved from port and payload analysis to statistical and learning-based methods, as several surveys report [1–3]. Encryption does not remove all information about a flow: inter-arrival times, packet sizes, byte and packet rates and flow duration remain observable and can serve as features. Classical learners on such features perform well on many tasks [3, 4], and deep models that take packet sequences or raw bytes as input [5–8, 21] have done better in some settings. Comparing these methods is difficult because studies prepare, split and label their data differently [22, 23]. VPN traffic adds another layer. Draper-Gil et al. [4] captured traffic from several application categories inside and outside a VPN and showed that time-related features can both detect VPN traffic and identify the application inside the tunnel; the companion paper [24] used similar features to recognise Tor traffic. The same dataset was later used to evaluate deep learning models that take packet- or byte-level input [5, 7]. In these studies a method is usually judged by its accuracy or F1 on a single train/test partition.")
rew("ISCX VPN-nonVPN is a popular benchmark for encrypted and VPN traffic classification.",
    "ISCX VPN-nonVPN is a popular benchmark for encrypted and VPN traffic classification. Some studies use its time-related features [4]; others rebuild packet- or byte-level input from the raw captures [5, 7, 8]. Two properties of the benchmark matter here: it is old, and it has been used heavily. Jacobs et al. [23] distilled network-security classifiers into interpretable surrogate models and found that several relied on spurious, context-dependent patterns that would probably not hold outside the benchmark. Their warning applies directly to reading benchmark explanations as general facts about network traffic. We do not try to improve a classifier on this benchmark. We keep the classifier family fixed and ask whether its explanation is consistent across the two observation contexts that the dataset was built to contrast.")
rep("Security machine learning guidelines on evaluation methods require protection against sampling bias, data snooping and spurious correlations [26, 30]. These criteria should also apply to model interpretations.",
    "Guidelines for evaluating machine learning in security call for protection against sampling bias, data snooping and spurious correlations [26, 30], and the same care should apply when models are interpreted.")
rew("In SHAP the model output is treated as a payoff of a cooperative game",
    "In SHAP the model output is treated as the payoff of a cooperative game, and each feature receives its Shapley value [31] in that game. For a single instance the values sum to the difference between the model's prediction and a baseline value [13]. TreeSHAP [14] computes these values efficiently for tree ensembles, in two common formulations. The tree-path-dependent version takes its reference behaviour from the cover statistics stored in the trees, so it needs no background dataset. The interventional version averages over a background sample, which breaks the dependence between features and yields a marginal (interventional) value function [19, 32]. The two versions answer different questions, and neither makes a claim about causal relationships in the data-generating process [19, 32, 33].")
rew("Most prior work on explanation stability considers",
    "Most prior work on explanation stability considers input perturbations [15, 16], sampling in local surrogate methods [40], adversarial manipulation of explanations [41] or disagreement between explanation methods [17]. Fisher et al. [42] observe that models with roughly equal performance can have substantially different importance profiles, so the profile of one model is only one member of a set of plausible profiles. Comparing feature rankings calls for rank statistics, such as the number of features shared by the top of two rankings. Spearman's ρ [43] and Kendall's τ [44] are the standard rank correlations and treat all positions alike, whereas rank-biased overlap [45] gives more weight to the top of a ranking. Bootstrap resampling can put confidence bounds around a ranking [46, 47].")

# ---- III
rep("Duplicate rows that were exactly equal in all 23 features and the label were removed (271 and 413 rows for Non-VPN and VPN contexts, respectively), resulting in 8,694 Non-VPN and 9,380 VPN flows. These were then split once in stratified 80% and 20% training and test partitions, respectively. This partitioning is our own and not provided with the dataset, hereafter referred to as the primary split and its parts as the primary partitions.",
    "Rows identical in all 23 features and the label were removed (271 Non-VPN and 413 VPN rows), leaving 8,694 Non-VPN and 9,380 VPN flows. We then split each context once into stratified 80% training and 20% test partitions. This split is our own, not one supplied with the dataset; we call it the primary split and its parts the primary partitions.")
rep("All of them have more than one label, hence they survived exact-row deduplication.",
    "Each of them occurs with more than one label, which is why exact-row deduplication kept them.")
rep("Table II and Fig. 1 show the distribution of all flows of each class in the primary partitions. Within each context, training and test have the same profile, with total variation distance between them equal to 0.001 for both Non-VPN and for VPN, as expected for a stratified split. By contrast, the picture is different across contexts.",
    "Table II and Fig. 1 give the class distribution of the primary partitions. Within each context, training and test have the same profile (total variation distance 0.001 in both contexts), as a stratified split should. Across contexts they differ.")
rep("MAIL is consistently the smallest class in either context, but increases between contexts from 2.7% to 4.6%.",
    "MAIL is the smallest class in both contexts but grows from 2.7% to 4.6%.")
rep("In Non-VPN the biggest class is 11.9 times bigger than the smallest, and 5.8 times bigger in the VPN.",
    "The largest class is 11.9 times the size of the smallest in Non-VPN and 5.8 times in VPN.")
rep("We audited the partitions prior to training anything.", "We audited the partitions before training any model.")
rep("Since the labels are inconsistent, their memorization by the model cannot be used to improve the accuracy on the test set.",
    "Because the labels disagree, memorising these vectors cannot raise test accuracy.")

# ---- IV
rep("There are two sets of control selections that serve as a check on the importance of any SHAP-based selection: the bottom-10, which corresponds to the ten least important features of the selector in the target context, and 20 random selections",
    "Two control selections show whether SHAP-based selection matters at all: the bottom-10 list, made of the ten least important features of the selector in the target context, and 20 random selections")
rep("and the ten indicators are appended; while in S2,", "and the ten indicators are appended. In S2,")

# ---- V
rep("We do not read this as indication that a VPN makes streaming easier to identify;", "We do not read this as evidence that a VPN makes streaming easier to identify;")

# ---- VI
rep("Some of the movers are correlated with other features (Fig. 6), which is important to how SHAP shares out credit.",
    "Several of the features that move are correlated with others (Fig. 6), and that matters for how SHAP divides credit.")
rep("Spearman correlation is 0.60 (Non-VPN) and 0.60 (VPN) between min_fiat and min_biat,",
    "Their Spearman correlation is 0.60 in both contexts between min_fiat and min_biat,",
    allow="the same value 0.60 for both contexts is stated once")
rep("As a group, the minimum IATs contain information that the other features do not have (individually, they are partially substitutable); dropping one increases the credit of the others: For example, in the VPN Random Forest, dropping min_fiat raises",
    "As a group, the minimum IATs carry information that the other features lack, but individually they partly substitute for one another, so dropping one shifts credit to the others. In the VPN Random Forest, for example, dropping min_fiat raises")
rep("points before that correction); whereas in the Non-VPN XGBoost dropping min_flowiat moves min_biat up the ranking from 10th to 5th.",
    "points before that correction). In the Non-VPN XGBoost, dropping min_flowiat moves min_biat from 10th to 5th place.")
rep("Such a negative gap between consecutive packets is simply not possible, probably due to an error in the timestamps. There are 17 and 12 flows with negative gaps, respectively, in these testing samples; these flows account for",
    "A negative gap between consecutive packets is impossible, so these values are probably timestamp errors. The test partitions hold 17 and 12 such flows, which account for")
rep("Class balance may also play a role as the rarest VPN class has 4.6% of training flows against 2.7% for Non-VPN MAIL, and rare classes give noisier attributions.",
    "Class balance may contribute as well: the rarest VPN class holds 4.6% of the training flows, against 2.7% for Non-VPN MAIL, and rare classes give noisier attributions.")

# ---- VII
rew("The stability of the six-family summary depends on the grouping method.",
    "How stable the family summary looks depends on how features are grouped. With importance summed over the six semantic families, the half-split gap is 0.112 for Random Forest and 0.075 for XGBoost, below the feature-level gaps. Six-family rankings are coarse, though: swapping one adjacent pair changes ρ by about 0.06. The eight-member directional family ranks first in every comparison, partly because it is the largest.")
rep("with throughput or duration emerging as the leading family.", "with throughput or duration as the leading family.")
rep("The eight data-driven clusters produce gaps of", "The eight data-driven clusters give gaps of")
rew("These results indicate that, under fine grouping, part of the XGBoost difference",
    "Under fine grouping, then, part of the XGBoost difference between contexts persists at the family level although it shrinks at the level of single features. With clusters of about 35 flows or the calibrated grouping, the XGBoost family gap falls as well (0.014 and +0.016). Redundancy among related flows therefore does not explain the XGBoost family-level difference at the finest grouping, though it may at coarser ones. When we say that grouping reduces the gap, we mean the feature-level gap, and a family summary should be read together with its grouping rule and family sizes.")

# ---- VIII
rep("The controls make it clear that SHAP-based selection is doing real work.", "The controls show that SHAP-based selection does real work.")

# ---- X
rew("Predictive accuracy, whole-ranking correlation and agreement among the leading features are distinct properties.",
    "Predictive accuracy, whole-ranking correlation and agreement among the leading features measure different things. Here a cross-context Spearman of about 0.8 comes with only three shared Top-5 features and an RBO of 0.58–0.75. The leading features also depend on the explanation method: the minimum IATs that rise most under SHAP move much less under marginal or conditional permutation importance. A single summary statistic would hide both points. Reporting accuracy, whole-ranking and top-weighted agreement, a within-context reference built from data that are as independent as possible, and a second importance measure gives a fuller picture.")
rep("At the family level, the XGBoost gap remains similar when using clusters of approximately ten flows, so related-flow redundancy explains it less clearly than it explains the feature-level gap. Therefore, family-level conclusions should always be interpreted alongside the grouping method and family sizes.",
    "At the family level, the XGBoost gap stays about the same under clusters of about ten flows, so related-flow redundancy explains it less clearly than it explains the feature-level gap, and family-level conclusions depend on the grouping rule and family sizes (Section VII).")
rep("The two consequence tests show different tendencies. A trained model does not transfer well across contexts, with macro F1 dropping to 0.43–0.61. SHAP-selected feature lists, however, transfer more reliably.",
    "The two consequence tests point in different directions. A trained model transfers poorly, with macro F1 dropping to 0.43–0.61 in the other context, whereas a SHAP-selected feature list transfers more reliably.")
rep("Consistency statements should be checked relative to a proper reference that does not involve overlapping sessions because the random partitioning of dependent streams would result in an overestimation of within-context consistency.",
    "Claims of consistency should be checked against a reference in which the compared models do not share sessions, because randomly splitting dependent flows overstates within-context consistency.")

# ---- XI
rep("This is the most serious limitation, which determines the main conclusion.",
    "This is the most serious limitation, and it bears directly on the main conclusion.")

# ---- availability statement: British spelling as elsewhere
rep("The processed artifacts (the deduplicated partitions,", "The processed artefacts (the deduplicated partitions,")

PART_B = Path(__file__).with_name("fix_v14_results.py")
if PART_B.exists(): exec(PART_B.read_text())

print("guarded edits with declared number changes:"); [print("  ", x) for x in LOG]
print("equations added", mathfix.fix([body]))
xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = xml if item.filename == "word/document.xml" else globals().get("REPLACE_MEDIA", {}).get(item.filename) or zin.read(item.filename)
    zo.writestr(item, data)
zo.close()
print("->", OUT)
