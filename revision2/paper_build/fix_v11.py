"""v11: replace the conclusion of v10 with the team's own text (attachment of 20 Oct 2026).

Three corrections to that text, everything else verbatim:
- Spearman 0.83 / 0.79 and RBO 0.58 / 0.75 belong to Random Forest / XGBoost (Table VI), not the reverse;
- "reduces the observed differences further" -> "also reduces the observed differences" (the reduction is
  relative to random halves, not beyond the permutation null);
- "At the selected resolution" -> "At the resolution suggested by the capture statistics" (the setting meant).
Usage: python fix_v11.py <v10 docx> <output docx>
"""
import sys, zipfile
from lxml import etree
import mathfix, edits_lib as E
from edits_lib import W, q, find_p, rewrite

SRC, OUT = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = E.body = doc.find(q(W, "body"))

NEW = [
 "This study examined whether the features used by a classifier to distinguish seven application categories remain consistent when network traffic is observed with and without a VPN. Experiments were conducted using the ISCX VPN-nonVPN benchmark and its 23 time-related flow features. The results show that classification performance declines in the VPN setting for both models. XGBoost's macro F1-score falls from 0.9340 to 0.8836, while Random Forest's decreases from 0.9171 to 0.8533. The cross-context results are less favourable, with macro F1-scores ranging from 0.43 to 0.61 when a model trained in one context is evaluated in the other. Although the SHAP feature rankings remain correlated across contexts, they are not identical. Spearman's rank correlations are 0.83 for Random Forest and 0.79 for XGBoost, while the corresponding RBO values are 0.58 and 0.75. Both models retain three of their five highest-ranked features, suggesting that some important features remain consistent even when the observation context changes.",
 "A key finding is that the measured difference in feature importance depends on the reference used for comparison. When each context is divided into randomly selected halves, the average feature-level gaps across 11 repetitions are 0.13 for XGBoost and 0.19 for Random Forest. However, a permutation-based null comparison that preserves both class distributions produces little or no gap. Grouping related traffic flows together also reduces the observed differences. For clusters containing approximately 10 flows, the gap ranges from 0.08 to 0.10, whereas clusters of around 35 flows reduce it to between 0.01 and 0.04. At the coarser resolution suggested by the capture statistics, the average gap approaches zero, although the resulting half-models are also weak. These results indicate that the apparent instability of SHAP rankings is influenced, at least in part, by how traffic flows are grouped within the same context.",
 "The experiments using consecutive blocks of flows from the original files provide a different perspective. Unlike feature-based clustering, this approach does not rely on the input features to construct groups. At the resolution suggested by the capture statistics, the observed gaps are 0.07 for XGBoost and 0.15 for Random Forest. Splitting each class into its first and second halves produces gaps of 0.03 and 0.09, respectively. The gaps remain non-negative across the tested repetitions. Not all measures of explanation stability respond equally to these changes: the top-weighted RBO gap and the XGBoost family-level gap decrease less under feature-based clustering than the Spearman gap. Changes in class composition have comparatively little effect on the results.",
 "The feature-level analysis also reveals differences between the two models. Minimum inter-arrival time (IAT) features show the most noticeable change in importance across contexts. For XGBoost, this change appears to arise mainly from a redistribution of Shapley values among features. Random Forest, in contrast, shows an increase in the importance of the minimum-IAT feature group itself. Despite these differences in feature attribution, the practical impact on feature selection is relatively small. For SHAP-selected feature sets containing at least 10 features, transferring the selected set between contexts results in an average macro-F1 loss of no more than approximately 0.013 across selection seeds. Matching the selected feature set to its original context provides only a small additional benefit.",
 "Taken together, these findings suggest that differences between SHAP rankings cannot be interpreted solely as evidence of a VPN-induced change in model behaviour. Some of the observed gap also reflects variation among related flows within a context, and the estimated contribution of this variation depends on the grouping strategy. Feature-based clustering reduces the excess gap to approximately zero at a coarse resolution, whereas grouping flows by their position in the original files leaves a positive difference. These experiments do not establish that tunnelling causes the observed changes, nor do they determine the magnitude of a genuine between-session effect. The grouping strategies used here are proxies and cannot fully separate within-session variation from differences between sessions. Further investigation will require datasets that preserve session identifiers or independent captures of both traffic contexts. Evaluating more recent traffic and examining explanations separately for each application category would also help establish whether the observed patterns generalise beyond this benchmark.",
]
OLD = ["We asked whether a classifier explains", "Both models degrade under VPN", "Whether the difference comes from the context",
       "With blocks of sequential flows from the original files", "On this benchmark, then, part of the random-split gap"]
paras = [find_p(o) for o in OLD]
for p, text in zip(paras, NEW):
    rewrite(p, [text])
print("equations added", mathfix.fix([body]))

xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close()
print("->", OUT)
