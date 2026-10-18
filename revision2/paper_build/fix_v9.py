"""Correct the team-edited manuscript (js_SHAP_VPN_revised_v8_nk_CHANGED_1_revised.docx) -> v9.

The team rephrased 38 passages of v8. This script keeps their wording and fixes only what the rephrasing
broke: factual errors, LaTeX markup left as text, plain-text Greek letters (integrity flag), citation style,
split paragraphs, spacing and typos. Every number in tables and text is unchanged.
Usage: python fix_v9.py <team docx> <output docx>
"""
import sys, copy, zipfile, re
from lxml import etree
import mathfix

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MN = "http://schemas.openxmlformats.org/officeDocument/2006/math"
q = lambda ns, t: f"{{{ns}}}{t}"
SRC, OUT = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = doc.find(q(W, "body"))


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


R = []  # (paragraph prefix, old, new)
add = lambda pre, *pairs: R.extend((pre, pairs[k], pairs[k + 1]) for k in range(0, len(pairs), 2))

# ---------------- Abstract
add("Abstract", "We evaluate it on the seven-class application task in the ISCX VPN-nonVPN benchmark, showing how Random Forest and XGBoost models trained separately on Non-VPN and VPN flows can be explained by TreeSHAP.",
    "We evaluate this on the seven-class application task of the ISCX VPN-nonVPN benchmark, using TreeSHAP to explain Random Forest and XGBoost models trained separately on Non-VPN and VPN flows.",
    "Each model has only three of its five leading features carried over from the same observation context into another.",
    "Each model keeps only three of its five leading features when the observation context changes.",
    "Compared to models trained on one half of one context, cross-context Spearman agreements are lower by 0.13 (XGBoost) and 0.19 (Random Forest) per model,",
    "Compared to models trained on random halves of one context, cross-context Spearman agreement is lower by 0.13 (XGBoost) and 0.19 (Random Forest),",
    "the proxy capturing sessions: down to zero", "the proxy for capture sessions: to about zero",
    "but the SHAP-chosen Top-10 feature list maintains itself with little loss.", "but a SHAP-chosen Top-10 feature list transfers with little loss.")
# ---------------- Introduction
add("A VPN does not change which application", "look similar [4].Several", "look similar [4]. Several")
add("Post-hoc explanation methods", "Importance values depend on the training data (that the model was trained on), the model itself, and the explanation method [15-17].",
    "Importance values depend on the training data, the model itself, and the explanation method [15–17].",
    "correlated features : credit", "correlated features: credit", "[18-20]", "[18–20]",
    "Min, mean, max and total inter-arrival times", "The minimum, mean, maximum and total inter-arrival times")
add("The VPN and Non-VPN components of the dataset",
    "provide for comparing model reliance across two observation contexts using same labels and same feature encoding. Class composition and capture sessions differ also between the components, though. Disagreement across contexts thus is not informative on its own and should be considered in light of disagreement between models trained on different data within a context. Only as good as the independence of its data is the reference provided by such disagreement, however, which is compromised when flows from a single capture session are often near repetitions of each other. Thus, a random division of one context will put related flows on both sides of the split, inflating apparent within-session agreement over between-session agreement.",
    "make it possible to compare model reliance across two observation contexts that use the same labels and the same feature encoding. The comparison is observational, though: class composition and capture sessions also differ between the components. Disagreement across contexts is therefore not informative on its own and has to be set against the disagreement between models trained on different data within one context. That reference is only as good as the independence of its data, and flows from a single capture session are often near repetitions of each other. A random division of one context therefore puts related flows on both sides of the split and makes within-context agreement look higher than it would be across sessions.")
add("The work makes four contributions",
    "the cost of using an already trained model for the other setting, and the benefit of using the other setting’s SHAP-selected feature list.",
    "the cost of using an already trained model in the other context, and the cost of reusing the other context’s SHAP-selected feature list.",
    "Section II. Related Work. Section III. Data and Audit Description. Section IV. Methodology and Section V–IX. Results. Section X. Summary and Discussion. Section XI. Threats to Validity, and Section XII. Conclusion.",
    "Section II reviews related work and Section III describes the data and its audit. Section IV gives the methodology and Sections V to IX the results. Section X summarises and discusses them, Section XI sets out threats to validity, and Section XII concludes.")
# ---------------- Related work
add("As encryption became widespread",
    "traffic classification based on port and payload analysis was gradually shifted to the use of statistical and learning-based methods, which was reported in a number of works [1–3].",
    "traffic classification gradually moved from port and payload analysis to statistical and learning-based methods, as several surveys report [1–3].",
    "VPN traffic has additional challenges since Draper-Gil et al. [4] showed that for many application categories, their traffic can be distinguished from standard ones, both inside and outside the VPN. Moreover, the same work showed that temporal features could be used for classifying both the application inside the VPN and the VPN itself.",
    "VPN traffic adds another layer. Draper-Gil et al. [4] captured traffic from several application categories both inside and outside a VPN and showed that time-related features can be used both to detect VPN traffic and to identify the application inside the tunnel.",
    "The data set was used to evaluate", "The same data set was later used to evaluate",
    "In all papers, the performance of the methods was evaluated using accuracy or F1 on the test set.",
    "In these studies a method is usually judged by its accuracy or F1 on a single train/test partition.")
add("ISCX VPN-nonVPN is a popular benchmark",
    "that would not generalize to other data than the one the model was trained on. Their warning has direct implications for interpreting the meaning of the benchmarks.",
    "that would probably not hold outside the benchmark. Their warning applies directly to reading benchmark explanations as general facts about network traffic.",
    "but taking a classifier family as a whole and asking if", "but keeping the classifier family fixed and asking whether")
add("Security classifiers are increasingly expected", "(11, 25-27)", "[11, 25–27]", "(9)", "[9]", "(10)", "[10]", "(28)", "[28]",
    "(12)", "[12]", "(29)", "[29]", "(26, 30)", "[26, 30]")
add("In SHAP the model output is treated", "there are two common implementations", "there are two common formulations",
    "The two versions answer slightly different questions and none of them make a claim", "The two versions answer different questions, and neither makes a claim")
add("Most prior work on explanation stability", "[15,16]", "[15, 16]", "[46,47]", "[46, 47]", "Kendall’s tau [44]", "Kendall’s τ [44]")
# ---------------- Dataset
add("Table II and Fig. 1 show the distribution",
    "A model fitted to one class distribution can give a different attribution to flows within classes than a model fitted to another class distribution, for equivalent contents of the classes within each flow. We treat class mix as a potential confounder therefore, evaluating it directly in Section VI-D.",
    "A model fitted to one class distribution can attribute importance differently from a model fitted to another, even when the flows within each class look the same, so we treat class mix as a potential confounder and test it directly in Section VI-D.")
add("BROWSING is the lone exception", "patterns.It", "patterns. It")
add("We audited the partitions prior to training",
    "Withing each partition, repeated instances of the same row were removed. After that, Non-VPN has 6951 unique feature vectors in the training set and 1737 in the test, with only one instance present in both. In the case of the VPN, there are 7489 and 1875 vectors in the train and test sets, respectively, with thirteen instances present in common. For each such instance we have one label in the training and one label in the test set, which only allows to refer to them as collisions of feature vectors.",
    "Within each partition, repeated instances of the same row were removed. After that, Non-VPN has 6,951 distinct feature vectors in the training set and 1,737 in the test set, and exactly one vector appears in both. VPN has 7,489 and 1,875, with thirteen shared. Each shared vector has one label in training and another in test, so we call them feature-vector collisions rather than copies.",
    "These collisions constitute 0.7% of the rows", "These collisions make up at most 0.7% of the rows")
add("Near duplicates needed an individual audit",
    "Near duplicates needed an individual audit. On standardised raw characteristics the nearest neighbour distances are minuscule (Non-VPN median test-to-train distance 0.0032 standard deviations), because a couple of extreme values dominate each heavy-tailed characteristic.",
    "Near duplicates needed a different check. On standardised raw features the nearest-neighbour distances are tiny (Non-VPN median test-to-train distance 0.0032 standard deviations), because a few extreme values dominate each heavy-tailed feature.")
add("The configurations are described in Table III",
    "Hyperparameters were tuned only once, and not re-tuned;", "Hyperparameters were set once and not tuned;",
    "The default random_state is 42 for all the main models,", "The primary models use random_state 42,")
# ---------------- Methods
add("Two things about the method limit",
    "the one selected by the model receives all the attribution, and two similar models can have different attributions for the same feature [18, 19]. Moreover, SHAP values are model-specific; thus, a feature with a high attribution is not necessarily an informative one: the model outcome depends on this feature, but it does not mean that it causes the application label [20, 32].We",
    "the one selected by the model receives the credit, and two otherwise similar models can divide attribution among correlated features quite differently [18, 19]. Moreover, SHAP values describe the model: a feature with a high attribution is one the model output depends on, which does not make it a cause of the application label [20, 32]. We")
add("In feature list transfer, selection is based", "[55-57]", "[55–57]")
add("Two controls determine whether",
    "Two controls determine whether a choice based on SHAPs is of any interest at all: the ten least relevant features of the target-context selector (bottom-10), and 20 randomly chosen 10-feature subsets, each drawn once and used consistently across all models and targets. For the first choice of seeds at k = 10, 3,000 class stratified paired bootstrap draws of the test records yield percentiles confidence intervals for own-minus-transferred difference, fixed in fitted models, omitting training and selection uncertainty, separately revealed by the five seeds.",
    "Two controls show whether SHAP-based selection matters at all: the ten least relevant features of the target-context selector (bottom-10), and 20 randomly chosen 10-feature subsets, each drawn once and used for all models and targets. For the first selection seed at k = 10, 3,000 class-stratified paired bootstrap draws of the test records give percentile intervals for the own-minus-transferred difference. These intervals keep the fitted models fixed and leave out training and selection uncertainty, which the five seeds show separately.")
add("In the processed features, the value -1", "replaced by the NaN", "replaced by NaN",
    "the share of these indicators are reported separately", "the share of these indicators is reported separately")
add("Per class, VOIP is relatively unaffected", "the classes streaming is confused with, might differ between contexts, and a flow level dataset can't distinguish these explanations.",
    "the classes streaming is confused with might differ between contexts, and a flow-level dataset cannot separate these explanations.")
# ---------------- Results
add("Some of the movers are correlated", "0.89-0.95", "0.89–0.95", "0.84-0.90", "0.84–0.90", "0.94.A model", "0.94. A model")
add("Taken together, cross-context rankings", "How much of it reflects flows related to one another depends on what approximates relatedness.",
    "How much of it reflects related flows depends on how relatedness is approximated.")
add("Note. Every repeat was redrawn",
    "Every repeat was redrawn stratified 80/20 partition of each context. Within values were calculated as the pair of halves of one context; cross values, as the pair of Non-VPN-half versus VPN-half from the same repeat. For Δ we took the average within value minus cross value per repeat; for SD and range – their spread across repeats (not standard errors). Half-model F1 is the macro F1 of the repeat’s test partition.",
    "Every repeat redraws the stratified 80/20 partition of each context. Within values compare the two halves of one context; cross values average the four Non-VPN-half versus VPN-half pairs of the same repeat. Δ is the within value minus the cross value, averaged over repeats; SD and range describe its spread across repeats and are not standard errors. Half-model F1 is macro F1 on the repeat’s test partition.",
    "The null design pooled the flows within each class across contexts, then resampled them with replacement into pseudo-contexts having the same class distribution as the originals. The half-models of Random Forest were evaluated on 800 test records per repeat.",
    "The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts. Random Forest half-models are explained on 800 test records per repeat (their macro F1 uses the full test partition).")
add("Note. Every repeat was redrawn",
    "Grouped designs kept whole k-means clusters (fit within class on log(x+2)-standardised features) in one half of the training partition; the calibrated design used numbers of clusters per class equal to that class’s number of capture files (Section IV-D). Row-block designs kept whole",
    "Grouped designs keep whole k-means clusters (fitted within class on log(x + 2)-standardised features) in one half of the training partition; the calibrated design sets the number of clusters per class to that class’s number of capture files (Section IV-D). Row-block designs keep whole")
add("Random Forest  (11 repeats per design)", "Random Forest  (11", "Random Forest (11")
add("Class-matched training yields the same answer",
    "two independent natural-mix draws", "two disjoint natural-mix draws", "0.77-0.82", "0.77–0.82",
    "class-matched agreement minus cross-context agreement would still be 0.09 with natural mixes. Two matched draws from Non-VPN share all 190 MAIL flows, and so the 0.971 agreement figure is not compared against this value.",
    "at this size the within-minus-cross gap is still 0.09 with natural mixes. Two matched draws from Non-VPN share all 190 MAIL flows, so their agreement (0.971) is not used as a reference.")
add("Class- mix swap", "Class- mix swap", "Class-mix swap")
add("Drop-one and grouped retraining",
    "decreases macro F1 by +0.001 to -0.016, whereas dropping all three costs 0.012-0.028.", "changes macro F1 by +0.001 to -0.016, whereas dropping all three costs 0.012–0.028.",
    "by 2.2 and 1.5, respectively", "by 2.2 and 1.5 percentage points, respectively",
    "so they are not entirely uninformative either.", "so the disagreement between contexts is not confined to the three minima.")
add("Permutation importance. Permutation importance on the test partitions",
    "but it is not an uninformative reference either: With dependent features, it evaluates the model’s performance on inputs that were not present in the training data [34].We",
    "but it is not a neutral reference either: with dependent features, it evaluates the model at feature combinations that do not occur in the data [34]. We")
add("Entries below -1. All entries below -1", "in the above data", "in the data")
add("In short, the minimum IATs are a group of importance", "the minimum IATs are a group of importance in both contexts", "the minimum IATs matter as a group in both contexts")
add("Under flow-level random halves, VPN rankings",
    "VPN rankings are more consistent with each other than with Non-VPN rankings (XGBoost 0.964 against 0.943; Random Forest 0.976 against 0.928), though lower accuracy for VPNs.",
    "VPN rankings agree with each other more closely than Non-VPN rankings do (XGBoost 0.964 against 0.943; Random Forest 0.976 against 0.928), even though VPN accuracy is lower.",
    "within-VPN agreement becomes below", "within-VPN agreement falls below",
    "random halves of a VPN's data", "random halves of VPN data",
    "from redundancy among related flows (VPN) rather than from any stability of the explanation itself for the VPN.",
    "from redundancy among related VPN flows rather than from any stability of the VPN explanation itself.")
add("The stability of the six-family summary", "changes the ranking by approximately 0.06", "changes ρ by approximately 0.06")
add("When ranked by mean importance per member", "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost.",
    "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost, larger than the semantic-family gaps, which fits with the three minimum IATs staying in separate clusters.")
add("These results indicate that some differences",
    "These results indicate that some differences between XGBoost contexts persist at the family level, even when they decrease at the individual-feature level.",
    "These results indicate that part of the XGBoost difference between contexts persists at the family level, even though it shrinks at the level of individual features.")
add("Reusing the other context's feature list", "\xa0The VPN random forest", "The VPN Random Forest")
# ---------------- Discussion
add("The answer to the question posed in the title", "one context,VPN", "one context, VPN",
    "nor the class-mix difference produce such", "nor the class-mix difference produces such")
add("Predictive accuracy, whole-ranking correlation and agreement among the leading features are distinct",
    "A cross-context Spearman of about 0.8, three shared Top-5 features and an RBO of between 0.58-0.75 each accompany this feature ranking dissimilarity; moreover top features change depending on explanation method, as the minimum IATs which increase the most when using SHAP, decrease the least when using marginal or conditional permutation importance.",
    "A cross-context Spearman of about 0.8 goes together with only three shared Top-5 features and an RBO of 0.58–0.75, and the leading features also change with the explanation method: the minimum IATs that rise most under SHAP move much less under marginal or conditional permutation importance.")
add("Aggregation into semantic families reduces the gaps",
    "while per-member means and correlation-based clusters provide alternative perspectives.", "while per-member means and correlation-based clusters give larger gaps.")
add("Correlated features complicate the interpretation",
    "whereas Random Forest assigns greater importance to the block feature.", "whereas for Random Forest the block as a whole gains importance under VPN.",
    "making the effect of related-flow redundancy less clear than at the feature level.", "so related-flow redundancy explains it less clearly than it explains the feature-level gap.")

cache = {}
for pre, old, new in R:
    if pre not in cache: cache[pre] = find_p(pre)   # resolve before an edit changes the paragraph's opening
    replace(cache[pre], old, new)
# prime of s' sits inside an equation as an ASCII apostrophe
for t in find_p("Spearman and Kendall weight a swap").iter(q(MN, "t")):
    if t.text == "'": t.text = "\u2032"

# ---------------- paragraphs rebuilt whole (LaTeX markup and broken lines)
rewrite(find_p("For Random Forest, the Spearman $"), [
    "For Random Forest, the Spearman ρ between the Non-VPN and VPN rankings is 0.8300 and Kendall τ", ("b", "subscript"),
    " is 0.6443; for XGBoost, ρ = 0.7856 and τ", ("b", "subscript"),
    " = 0.5810 (Table VI). The whole-ranking correlations are high, but each model keeps only three of its top five features (κ", ("5", "subscript"),
    " = 0.49), and RBO is 0.580 for Random Forest and 0.755 for XGBoost. The L1 distance between share vectors is 0.333 and 0.360, so about a sixth of the total importance changes hands, and the mean absolute rank displacement is 3.04 and 3.39 positions. With ten features the overlap is 9 of 10 for Random Forest (κ", ("10", "subscript"),
    " = 0.82) and 8 of 10 for XGBoost (κ", ("10", "subscript"),
    " = 0.65). Weighting true classes equally changes the within-context rankings little (ρ between weighted and unweighted rankings 0.976–0.998) and leaves the cross-context values at a similar level."])
rewrite(find_p("Fig. 5 plots single features"), [
    "Fig. 5 plots single features (Non-VPN rank to VPN rank). In Random Forest, min_fiat (11 to 2) and min_biat (10 to 3) move up, but total_biat (1 to 5), total_fiat (3 to 8) and mean_fiat (7 to 14) all move down. In XGBoost, mean_biat (12 to 5), min_biat (10 to 4) and max_flowiat (13 to 7) rise, whereas total_biat (3 to 12) and flowPktsPerSecond (5 to 14) fall. Some directional features rise while others in the same family fall, so a description in terms of forward versus backward misses most of what happens."])
rewrite(find_p("The two consequence tests show different tendencies"), [
    "The two consequence tests show different tendencies. A trained model does not transfer well across contexts, with macro F1 dropping to 0.43–0.61. SHAP-selected feature lists, however, transfer more reliably. At k = 10 and k = 15, transferring a feature list reduces macro F1 by at most about 0.013 on average across selection seeds, with a maximum of 0.017 for an individual seed. For some target contexts, the transferred list performs better than the context-specific list."])
rewrite(find_p("At \\(k=10\\), matching the feature list"), [
    "At k = 10, matching the feature list to its context has a small effect, with differences of +0.008 for Random Forest and +0.010 for XGBoost. The largest per-target cost occurs for the VPN Random Forest and mostly reflects a Non-VPN feature list that is weaker on both targets. At k = 5, where two of the five selected features differ, single comparisons reach 0.05. The conclusion therefore holds only for moderately long feature lists."])
# VIII-B: the word "zero." was split into its own paragraph
merge_next(find_p("Reusing the other context's feature list"))
replace(find_p("Reusing the other context's feature list"), "includes", "includes ")
# XII: the second and third paragraphs were broken into four lines
p1 = find_p("Both ensembles show drops in performance")
lines = [p1, p1.getnext(), p1.getnext().getnext(), p1.getnext().getnext().getnext()]
tail = ptext(lines[3]).split("difference.. ", 1)[1]
rewrite(p1, ["Both ensembles lose performance under VPN (XGBoost macro F1 0.9340 to 0.8836, Random Forest 0.9171 to 0.8533), and a model trained on one context achieves only 0.43–0.61 macro F1 on the other. The primary SHAP rankings are related between contexts but not interchangeable: Spearman ρ is 0.83 and 0.79, RBO is 0.58 and 0.75, and each model retains three of its five top features."])
rewrite(lines[1], ["Whether this disagreement is due to context depends on the reference. Against randomly partitioned halves of one context, the feature-level gap is 0.13 for XGBoost and 0.19 for Random Forest, averaged over 11 repeats with new partitions each time, and a permutation null that keeps both class mixes gives little or no gap. " + tail])
for p in lines[2:]: p.getparent().remove(p)

# ---------------- subscripts / superscripts left as text
split_token(find_p("Table II and Fig. 1 show the distribution"), "χ^2", ["χ", ("2", "superscript")])
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

pdef = find_p("where ϕ0,c is the expected output")
to_math(pdef, "s_j", "s", sub="_j")
to_math(pdef, "of feature j is", "j")
to_math(pdef, "the n explained", "n")
# render hints (lastRenderedPageBreak) and runs holding several text nodes make mathfix skip a run
for e in list(body.iter(q(W, "lastRenderedPageBreak"))): e.getparent().remove(e)
for r in body.iter(q(W, "r")):
    ts = r.findall(q(W, "t"))
    if len(ts) > 1 and all(c.tag in (q(W, "rPr"), q(W, "t")) for c in r):
        ts[0].text = "".join(t.text or "" for t in ts)
        for t in ts[1:]: r.remove(t)

# ---------------- every remaining Greek letter, prime, times sign, arrow and no-break space
n = mathfix.fix([body])
# the minus sign of equation (6) was retyped as a hyphen; inside an equation the proper minus sign is fine
for om in body.iter(q(MN, "oMath")):
    if "within,NV" in "".join(t.text or "" for t in om.iter(q(MN, "t"))):
        for t in om.iter(q(MN, "t")):
            if t.text == "-": t.text = "\u2212"
print("equations added", n)

xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close()
print("->", OUT)
