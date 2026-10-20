# Part B of fix_v14.py (executed inside it): the two new analyses. Every number is read from
# results/p11_v14.json (src/p11_v14_summary.py). Uses rep / rew / guard / find_c / body from fix_v14.py.
G = json.load(open(RES / "p11_v14.json"))
GN, CR = G["grouped_null"], G["common_records"]
f3 = lambda x: f"{x:.3f}"
def sg(x, d=3):
    s = f"{x:+.{d}f}"
    return s[1:] if float(s) == 0 else s
NAMES = {"grouped10": "clusters of about 10 flows", "grouped35": "clusters of about 35 flows", "calibrated": "calibrated clusters",
         "orderedcal": "calibrated row blocks", "orderedhalf": "the first-versus-second-half split"}
ND = ("grouped10", "grouped35", "calibrated", "orderedcal", "orderedhalf")
NEW = "results (v14)"

# ---- I. contributions: the two new references
rep("and a permutation null in which flows are reassigned at random between two pseudo-contexts.",
    "and a permutation null in which flows are reassigned at random between two pseudo-contexts, run with random and with grouped halves. It also checks whether the gap survives when the models of both contexts are explained on the same records.")

# ---- IV-D: grouped null and common explained records (method)
rep("Under this null, Δ shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts.",
    "Under this null, Δ shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts. We also apply five of the grouped designs to the same pseudo-contexts: clusters of about 10 and 35 flows, calibrated clusters, calibrated row blocks and the first-versus-second-half split. Because each pseudo-context mixes flows from the Non-VPN and the VPN file, the grouping is applied separately to the flows of each file within a class, with the cluster or capture counts of that file's context, and each half takes its share from both; otherwise a half could hold mostly flows of one file, and the within pair would partly be a cross-context pair. For row blocks, flows keep their position in their original ARFF file. This grouped null shows what each grouping procedure produces by itself when the two pseudo-contexts differ only in class mix.",
    allow=NEW)
rep("Part of the cross-context disagreement may therefore come from the difference in explained records rather than from the models. The permutation null shares this asymmetry, since each pseudo-context is explained on its own test partition; with records that are exchangeable between pseudo-contexts, the asymmetry and the class-mix difference together give a mean gap of +0.028 for Random Forest and -0.001 for XGBoost. No grouped design controls for it, and in the real contexts the explained records also differ in distribution.",
    "Part of the cross-context disagreement could therefore come from the difference in explained records rather than from the models. To measure this, we refit the random-halves design with the same seeds and explain every half-model on the explained records of both contexts, so that within- and cross-context pairs can be compared on one common record set: the Non-VPN records, the VPN records, or both together. Each refitted half-model reproduces the stored explanation on its own records exactly. The permutation null shares the asymmetry, since each pseudo-context is explained on its own test partition, but there the records of the two pseudo-contexts are exchangeable.",
    allow=NEW)

# ---- Table VII: grouped-null rows after each model's "Permutation null" row
def table_after(caption):
    cap = [p for p in body.iter(q(W, "p")) if ptext(p).strip() == caption]; assert len(cap) == 1, caption
    el = cap[0].getnext()
    while el.tag != q(W, "tbl"): el = el.getnext()
    return el
def set_par(p, text):
    rs = list(p.iter(q(W, "r"))); assert rs, "empty cell paragraph"
    for r in rs[1:]: r.getparent().remove(r)
    t = rs[0].find(q(W, "t")); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
T7 = table_after("TABLE VII")
LABEL = {"grouped10": "Null, grouped ~10 per cluster", "grouped35": "Null, grouped ~35 per cluster", "calibrated": "Null, grouped calibrated",
         "orderedcal": "Null, row blocks calibrated", "orderedhalf": "Null, row blocks first vs second half"}
model = None; anchors = {}
for tr in T7.iter(q(W, "tr")):
    first = ptext(tr).strip()
    if first.startswith("XGBoost ("): model = "XGB"
    elif first.startswith("Random Forest ("): model = "RF"
    elif first.startswith("Permutation null") and model: anchors[model] = tr
assert set(anchors) == {"XGB", "RF"}
for m, tr0 in anchors.items():
    prev = tr0
    for d in ND:
        g = GN[f"{m}_{d}"]; s = g["null"]
        vals = [[LABEL[d]], [f"{f3(s['f1_nv'])} / {f3(s['f1_vpn'])}"], [f3(s["rho_wnv"])], [f3(s["rho_wvpn"])], [f3(s["rho_cross"])],
                [f"{sg(s['rho_gap'])} ({f3(g['null_gap_sd'])})", f"{g['null_gap_min']:.3f} to {g['null_gap_max']:.3f}"],
                [sg(s["rbo_gap"])], [sg(s["top5_gap"], 2)], [sg(s["rho_w_gap"])]]
        new = copy.deepcopy(tr0)
        cells = list(new.iter(q(W, "tc"))); assert len(cells) == len(vals)
        for tc, v in zip(cells, vals):
            ps = [p for p in tc.iter(q(W, "p")) if ptext(p).strip()]
            assert len(ps) == len(v), (d, v, [ptext(p) for p in ps])
            for p, text in zip(ps, v): set_par(p, text)
        prev.addnext(new); prev = new
rep("The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts.",
    "The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts; the rows marked Null with a grouped design split those pseudo-contexts into halves with that design.")

# ---- Fig. 2: pipeline figure regenerated by src/figs.py (lists the grouped null and the same-record check)
FIG2_BYTES = (RES.parent / "figures" / "fig2_pipeline.png").read_bytes()
REPLACE_MEDIA = {"word/media/image2.png": FIG2_BYTES}
