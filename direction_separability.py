"""Direction separability test.

Question: does ANY available entry-time feature predict whether the signal's
direction is right? Baseline hit rate is ~50.6% (coin flip).

Walk-forward by month, train on strictly-earlier months, test on current month.
Read-only. No pipeline changes.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

PATH = "/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/2026-07-24/validation_details.csv"

NUM = ["dte", "underlying_price", "bid_ask_spread_pct", "entry_ask", "entry_bid",
       "contract_reference_strike"]
CAT = ["direction", "strategy_kind", "market_regime", "sector",
       "base_pattern_family", "contract_dte_bucket", "contract_moneyness_bucket"]


def main() -> int:
    d = pd.read_csv(PATH, low_memory=False)
    d = d[d.status == "SCORED"].copy()
    d = d[d.stock_proxy_move.notna()].copy()

    # directional hit: was the signal's direction correct on the underlying?
    sign = np.where(d.direction.str.lower().str.startswith("bull"), 1.0, -1.0)
    d["signed_move"] = sign * d.stock_proxy_move.astype(float)
    d["hit"] = (d.signed_move > 0).astype(int)
    d["month"] = d.signal_date.str.slice(0, 7)

    print(f"rows={len(d)}  base hit rate={d.hit.mean():.4f}")
    print("by month:")
    print(d.groupby("month").hit.agg(["size", "mean"]).round(4).to_string())
    print()

    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CAT:
        d[c] = d[c].fillna("NA").astype(str)
    d[NUM] = d[NUM].fillna(d[NUM].median())

    months = sorted(d.month.unique())
    pre = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    rows = []
    oof = []
    for m in months[1:]:
        tr = d[d.month < m]
        te = d[d.month == m]
        if len(tr) < 500 or len(te) < 50 or tr.hit.nunique() < 2:
            continue
        model = Pipeline([
            ("pre", pre),
            ("clf", GradientBoostingClassifier(random_state=0, n_estimators=200,
                                               max_depth=3, learning_rate=0.05,
                                               subsample=0.8)),
        ])
        model.fit(tr[NUM + CAT], tr.hit)
        p = model.predict_proba(te[NUM + CAT])[:, 1]
        auc = roc_auc_score(te.hit, p) if te.hit.nunique() > 1 else float("nan")
        k = max(1, int(len(te) * 0.10))
        top = np.argsort(-p)[:k]
        rows.append({
            "month": m, "n_train": len(tr), "n_test": len(te),
            "base_hit": te.hit.mean(), "auc": auc,
            "top10_hit": te.hit.values[top].mean(),
            "top10_avgR": te.net_r.values[top].mean(),
        })
        t = te.copy()
        t["p"] = p
        oof.append(t)
        print(f"  {m}: n={len(te):5d} base={te.hit.mean():.4f} auc={auc:.4f} "
              f"top10%hit={te.hit.values[top].mean():.4f} top10%avgR={te.net_r.values[top].mean():+.4f}",
              flush=True)

    r = pd.DataFrame(rows)
    print()
    print("=== POOLED ===")
    print(f"mean OOS AUC        : {r.auc.mean():.4f}   (0.50 = no signal)")
    print(f"folds with AUC>0.52 : {(r.auc > 0.52).sum()}/{len(r)}")
    o = pd.concat(oof)
    print(f"pooled base hit     : {o.hit.mean():.4f}  n={len(o)}")
    k = max(1, int(len(o) * 0.10))
    top = o.nlargest(k, "p")
    print(f"pooled top10% hit   : {top.hit.mean():.4f}  n={len(top)}")
    print(f"pooled top10% avg R : {top.net_r.mean():+.4f}")
    w = top.net_r[top.net_r > 0].sum()
    l = -top.net_r[top.net_r < 0].sum()
    print(f"pooled top10% PF    : {(w/l if l else float('nan')):.4f}")

    print()
    print("=== hit rate by single feature (pooled, sanity) ===")
    for c in ["direction", "market_regime", "strategy_kind", "contract_dte_bucket"]:
        g = o.groupby(c).hit.agg(["size", "mean"]).round(4)
        print(f"-- {c}\n{g[g['size'] >= 100].to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
