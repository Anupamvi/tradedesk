# Delta Gate + IVR Scoring + GEX Regime Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add delta gating, IVR-aware conviction scoring, and GEX regime detection across both the daily pipeline and trade-history pipeline to reject structurally unsound trades and improve trade quality.

**Architecture:** Three independent signals (delta, IVR, GEX) are added as data flow enrichments in the Schwab validation phase, then consumed as blockers in the approval phase and as scoring adjustments in Stage-1 conviction. GEX is computed per-ticker from the full option chain's gamma × OI. All three apply to both FIRE and SHIELD tracks with opposite polarities.

**Tech Stack:** Python 3.11, pandas, Schwab API (via schwab_auth.py), dataclasses

---

## Task 1: Add Delta + GEX Fields to SwingScore Dataclass

**Files:**
- Modify: `uwos/swing_trend_pipeline.py:247-290`

**Step 1: Add new fields to SwingScore after line 282 (after `live_validation_note`)**

```python
    # Live greeks (populated during Schwab validation)
    short_delta_live: float = math.nan       # Short leg delta (SHIELD verticals)
    long_delta_live: float = math.nan        # Long leg delta (FIRE verticals)
    short_put_delta_live: float = math.nan   # IC put short delta
    short_call_delta_live: float = math.nan  # IC call short delta
    # GEX regime (populated during Schwab validation)
    net_gex: float = math.nan                # Net gamma exposure (positive=pinned, negative=volatile)
    gex_regime: str = ""                     # "pinned" or "volatile"
    gex_support: float = math.nan            # Highest put GEX wall below spot
    gex_resistance: float = math.nan         # Highest call GEX wall above spot
```

**Step 2: Verify no breakage**

Run: `cd c:/uw_root && python -c "from uwos.swing_trend_pipeline import SwingScore; s = SwingScore(); print(s.net_gex, s.gex_regime)"`
Expected: `nan ` (nan and empty string)

**Step 3: Commit**

```bash
git add uwos/swing_trend_pipeline.py
git commit -m "feat(gates): add delta and GEX fields to SwingScore dataclass"
```

---

## Task 2: Compute GEX Per Ticker in validate_with_schwab()

**Files:**
- Modify: `uwos/swing_trend_pipeline.py:1497-1612`

**Step 1: Add GEX computation helper function before `validate_with_schwab()` (before line 1497)**

```python
def _compute_ticker_gex(
    chain_map: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]],
    spot: float,
    target_expiry_str: str,
) -> Dict[str, Any]:
    """Compute net GEX and GEX walls from option chain data.

    GEX = gamma x openInterest x 100 x spot
    Calls contribute positive GEX (dealers long gamma).
    Puts contribute negative GEX (dealers short gamma on hedge side).
    Net GEX > 0 => mean-reverting ("pinned"), < 0 => trending ("volatile").
    """
    exp_data = chain_map.get(target_expiry_str)
    if not exp_data or spot <= 0:
        return {"net_gex": math.nan, "gex_regime": "", "gex_support": math.nan, "gex_resistance": math.nan}

    call_contracts = exp_data.get("C", {})
    put_contracts = exp_data.get("P", {})

    total_call_gex = 0.0
    total_put_gex = 0.0
    call_gex_by_strike: Dict[float, float] = {}
    put_gex_by_strike: Dict[float, float] = {}

    for strike, contract in call_contracts.items():
        gamma = contract.get("gamma")
        oi = contract.get("openInterest")
        if gamma is not None and oi is not None:
            try:
                g, o = float(gamma), float(oi)
                if math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0:
                    gex = g * o * 100.0 * spot
                    total_call_gex += gex
                    call_gex_by_strike[strike] = gex
            except (TypeError, ValueError):
                pass

    for strike, contract in put_contracts.items():
        gamma = contract.get("gamma")
        oi = contract.get("openInterest")
        if gamma is not None and oi is not None:
            try:
                g, o = float(gamma), float(oi)
                if math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0:
                    gex = g * o * 100.0 * spot
                    total_put_gex += gex
                    put_gex_by_strike[strike] = gex
            except (TypeError, ValueError):
                pass

    net_gex = total_call_gex - total_put_gex
    gex_regime = "pinned" if net_gex >= 0 else "volatile"

    # GEX walls: strikes with highest gamma concentration
    gex_support = math.nan
    gex_resistance = math.nan

    put_below = {k: v for k, v in put_gex_by_strike.items() if k < spot}
    if put_below:
        gex_support = max(put_below, key=put_below.get)

    call_above = {k: v for k, v in call_gex_by_strike.items() if k > spot}
    if call_above:
        gex_resistance = max(call_above, key=call_above.get)

    return {
        "net_gex": round(net_gex, 2),
        "gex_regime": gex_regime,
        "gex_support": gex_support,
        "gex_resistance": gex_resistance,
    }
```

**Step 2: Call GEX computation in the per-ticker loop, after chain_map is built**

In `validate_with_schwab()`, after line 1612 (after `chain_map` construction), before the `for s in ticker_score_list:` loop at line 1615, add:

```python
        # Compute GEX for this ticker (use the nearest expiry to first score's target)
        gex_expiry = best_exp_str_for_gex = None
        for s_gex in ticker_score_list:
            try:
                t_exp = dt.date.fromisoformat(s_gex.target_expiry)
                for exp_str in chain_map:
                    try:
                        exp_d = dt.date.fromisoformat(exp_str)
                        if abs((exp_d - t_exp).days) <= 7:
                            gex_expiry = exp_str
                            break
                    except (ValueError, TypeError):
                        continue
                if gex_expiry:
                    break
            except (ValueError, TypeError):
                continue

        ticker_gex = {"net_gex": math.nan, "gex_regime": "", "gex_support": math.nan, "gex_resistance": math.nan}
        if gex_expiry and spot is not None:
            ticker_gex = _compute_ticker_gex(chain_map, spot, gex_expiry)
```

**Step 3: Write GEX fields to each SwingScore in the per-score loop**

At the start of the `for s in ticker_score_list:` loop (after line 1617 `s.live_spot = spot`), add:

```python
            s.net_gex = ticker_gex["net_gex"]
            s.gex_regime = ticker_gex["gex_regime"]
            s.gex_support = ticker_gex["gex_support"]
            s.gex_resistance = ticker_gex["gex_resistance"]
```

**Step 4: Verify no breakage**

Run: `cd c:/uw_root && python -c "from uwos.swing_trend_pipeline import _compute_ticker_gex; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add uwos/swing_trend_pipeline.py
git commit -m "feat(gates): compute GEX per ticker from Schwab chain gamma x OI"
```

---

## Task 3: Extract Live Delta in validate_with_schwab()

**Files:**
- Modify: `uwos/swing_trend_pipeline.py:1615-1870`

**Step 1: Add delta extraction for IC trades (4-leg path)**

After IC pricing is complete and before `s.live_validated = True` (around line 1757), add delta extraction. Insert after the `s.live_strike_setup = ...` line:

```python
                # Extract live deltas for IC short legs
                ps_contract = put_contracts.get(snap_ps, {})
                cs_contract = call_contracts.get(snap_cs, {})
                ps_delta = ps_contract.get("delta")
                cs_delta = cs_contract.get("delta")
                if ps_delta is not None:
                    try:
                        s.short_put_delta_live = float(ps_delta)
                    except (TypeError, ValueError):
                        pass
                if cs_delta is not None:
                    try:
                        s.short_call_delta_live = float(cs_delta)
                    except (TypeError, ValueError):
                        pass
```

**Step 2: Add delta extraction for 2-leg verticals**

After the 2-leg `s.live_validated = True` (around line 1862), before the loop continues, add:

```python
            # Extract live deltas for vertical legs
            short_delta_raw = short_contract.get("delta")
            long_delta_raw = long_contract.get("delta")
            if short_delta_raw is not None:
                try:
                    s.short_delta_live = float(short_delta_raw)
                except (TypeError, ValueError):
                    pass
            if long_delta_raw is not None:
                try:
                    s.long_delta_live = float(long_delta_raw)
                except (TypeError, ValueError):
                    pass
```

**Step 3: Commit**

```bash
git add uwos/swing_trend_pipeline.py
git commit -m "feat(gates): extract live delta from Schwab contracts for FIRE and SHIELD"
```

---

## Task 4: Add Delta + GEX Fields to Shortlist CSV Output

**Files:**
- Modify: `uwos/swing_trend_pipeline.py:2309-2361`

**Step 1: Add new columns to the CSV dict**

In `generate_shortlist_csv()`, after the backtest columns (after line 2360 `"backtest_confidence": s.backtest_confidence,`), add:

```python
            # Live greeks
            "short_delta_live": round(s.short_delta_live, 4) if math.isfinite(s.short_delta_live) else "",
            "long_delta_live": round(s.long_delta_live, 4) if math.isfinite(s.long_delta_live) else "",
            "short_put_delta_live": round(s.short_put_delta_live, 4) if math.isfinite(s.short_put_delta_live) else "",
            "short_call_delta_live": round(s.short_call_delta_live, 4) if math.isfinite(s.short_call_delta_live) else "",
            # GEX regime
            "net_gex": round(s.net_gex, 2) if math.isfinite(s.net_gex) else "",
            "gex_regime": s.gex_regime,
            "gex_support": round(s.gex_support, 2) if math.isfinite(s.gex_support) else "",
            "gex_resistance": round(s.gex_resistance, 2) if math.isfinite(s.gex_resistance) else "",
```

**Step 2: Commit**

```bash
git add uwos/swing_trend_pipeline.py
git commit -m "feat(gates): add delta and GEX columns to shortlist CSV output"
```

---

## Task 5: Add FIRE + SHIELD Blockers in run_mode_a_two_stage.py

**Files:**
- Modify: `uwos/run_mode_a_two_stage.py:1215-1386`

**Step 1: Read new config values at the top of the approval section**

Find where the existing approval config is read (search for `require_live_shield_short_delta`). Near those lines, add:

```python
    # FIRE delta gate
    require_fire_long_delta = bool(approval_cfg.get("require_fire_long_delta", False))
    min_abs_long_delta_fire = fnum(approval_cfg.get("min_abs_long_delta_fire", 0.15))
    # GEX regime gate
    require_gex_regime = bool(approval_cfg.get("require_gex_regime", False))
```

**Step 2: Add FIRE delta blocker**

After the SHIELD delta check block (after line 1346), add the FIRE delta gate:

```python
            # FIRE long-leg delta gate: reject lottery tickets
            if track == "FIRE" and require_fire_long_delta:
                if strategy_local in {"Bull Call Debit", "Bear Put Debit"}:
                    long_delta = fnum(row.get("long_delta_live"))
                    if not np.isfinite(long_delta):
                        blockers.append("fire_delta_missing")
                    elif abs(long_delta) < min_abs_long_delta_fire:
                        blockers.append(f"fire_delta_low:{long_delta:+.2f}")
```

**Step 3: Add GEX regime blocker (both tracks)**

After the FIRE delta block, add:

```python
            # GEX regime gate
            if require_gex_regime:
                gex_regime = str(row.get("gex_regime", "")).strip().lower()
                if gex_regime:  # only block if GEX data is available
                    if track == "SHIELD" and gex_regime == "volatile":
                        blockers.append("shield_gex_volatile")
                    elif track == "FIRE" and gex_regime == "pinned":
                        net_gex_val = fnum(row.get("net_gex"))
                        # Only block FIRE if GEX is strongly pinned (not marginal)
                        if np.isfinite(net_gex_val) and net_gex_val > 0:
                            blockers.append("fire_gex_pinned")
                    # IC-specific: require pinned regime
                    if strategy_local in {"Iron Condor", "Iron Butterfly"} and gex_regime != "pinned":
                        blockers.append("ic_gex_not_pinned")
```

**Step 4: Update split_blockers() to classify new blockers**

In `split_blockers()` at line 1374-1382, add new quality patterns:

```python
            if (
                token.startswith("likelihood_")
                or token.startswith("edge_below")
                or token.startswith("signals_below")
                or token.startswith("shield_sigma")
                or token.startswith("credit_no_touch")
                or token.startswith("shield_core")
                or token.startswith("shield_delta")
                or token.startswith("fire_delta")
                or token.startswith("fire_gex")
                or token.startswith("shield_gex")
                or token.startswith("ic_gex")
            ):
                quality.append(token)
```

**Step 5: Commit**

```bash
git add uwos/run_mode_a_two_stage.py
git commit -m "feat(gates): add FIRE delta, SHIELD/FIRE GEX regime blockers"
```

---

## Task 6: Move shield_sigma_fail to Hard Blocker

**Files:**
- Modify: `uwos/run_mode_a_two_stage.py:1374-1382`

**Step 1: Remove `shield_sigma` from quality blocker classification**

In `split_blockers()`, remove the line `or token.startswith("shield_sigma")` from the quality conditions. The updated block becomes:

```python
            if (
                token.startswith("likelihood_")
                or token.startswith("edge_below")
                or token.startswith("signals_below")
                or token.startswith("credit_no_touch")
                or token.startswith("shield_core")
                or token.startswith("shield_delta")
                or token.startswith("fire_delta")
                or token.startswith("fire_gex")
                or token.startswith("shield_gex")
                or token.startswith("ic_gex")
            ):
                quality.append(token)
```

This makes `shield_sigma_fail` and `shield_sigma_unknown` fall through to the `hard` list (line 1385), preventing any Tactical promotion of ATM SHIELD trades.

**Step 2: Commit**

```bash
git add uwos/run_mode_a_two_stage.py
git commit -m "fix(gates): make shield_sigma_fail a hard blocker — no Tactical promotion for ATM SHIELD"
```

---

## Task 7: Add IVR Conviction Bonus/Penalty in eod_trade_scan_mode_a.py

**Files:**
- Modify: `uwos/eod_trade_scan_mode_a.py:705-714, 786-794, 875-878, 989-992, 1228-1232`

**Step 1: Add IVR adjustment after each conviction formula**

For each of the 5 strategy scoring blocks, add an IVR adjustment AFTER the `conv = int(round(100 * score))` line. The `iv_rank` variable is already available in scope (loaded at line 621).

**Bull Call Debit (after line 712):**
```python
                        conv = int(round(100 * score))
                        # IVR bonus: low IV = cheap options = good for buying
                        if np.isfinite(iv_rank):
                            if iv_rank < 25:
                                conv = min(100, conv + 3)
                            elif iv_rank > 70:
                                conv = max(0, conv - 5)
```

**Bear Put Debit (after line 793):**
```python
                        conv = int(round(100 * score))
                        # IVR bonus: low IV = cheap options = good for buying
                        if np.isfinite(iv_rank):
                            if iv_rank < 25:
                                conv = min(100, conv + 3)
                            elif iv_rank > 70:
                                conv = max(0, conv - 5)
```

**Bull Put Credit (after line 878):**
```python
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for selling
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)
```

**Bear Call Credit (after line 992):**
```python
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for selling
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)
```

**Iron Condor (after line 1232):**
```python
                        conv = int(round(100 * score))
                        # IVR bonus: high IV = rich premium = good for IC
                        if np.isfinite(iv_rank):
                            if iv_rank > 50:
                                conv = min(100, conv + 3)
                            elif iv_rank < 20:
                                conv = max(0, conv - 5)
```

**Step 2: Commit**

```bash
git add uwos/eod_trade_scan_mode_a.py
git commit -m "feat(gates): add IVR conviction bonus/penalty across all 5 strategy scoring blocks"
```

---

## Task 8: Update Config File

**Files:**
- Modify: `uwos/rulebook_config_goal_holistic_claude.yaml:112-169`

**Step 1: Enable delta gate and add new config keys**

Change `require_live_shield_short_delta: false` to `true` at line 162.

Add new keys in the `approval:` section (after line 163):

```yaml
  # FIRE delta gate: reject lottery tickets with low directional exposure
  require_fire_long_delta: true
  min_abs_long_delta_fire: 0.15
  # GEX regime gate: block regime-mismatched trades
  require_gex_regime: true
```

**Step 2: Commit**

```bash
git add uwos/rulebook_config_goal_holistic_claude.yaml
git commit -m "feat(gates): enable delta gate, add FIRE delta and GEX regime config keys"
```

---

## Task 9: Add GEX to Position Analyzer (Trade-History Pipeline)

**Files:**
- Modify: `uwos/schwab_position_analyzer.py:354-360`

**Step 1: Change strike_count to None for full chain (needed for GEX)**

At line 357, change:
```python
            chains_payload[ul] = svc.get_option_chain(ul, strike_count=12)
```
to:
```python
            chains_payload[ul] = svc.get_option_chain(ul, strike_count=None)
```

**Step 2: Add GEX computation after chain collection (after line 359)**

```python
    # 5b. Compute GEX per underlying from full chain
    gex_by_underlying = {}
    for ul, chain_data in chains_payload.items():
        ul_spot = None
        ul_quote = underlying_quotes.get(ul, {})
        for fld in ("mark", "last", "close"):
            v = ul_quote.get(fld)
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv) and fv > 0:
                        ul_spot = fv
                        break
                except (TypeError, ValueError):
                    pass
        if ul_spot is None or ul_spot <= 0:
            continue

        total_call_gex = 0.0
        total_put_gex = 0.0
        best_put_wall = (0.0, math.nan)   # (gex_value, strike)
        best_call_wall = (0.0, math.nan)

        for map_name, side in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
            exp_map = chain_data.get(map_name, {}) or {}
            for exp_key, strike_map in exp_map.items():
                for strike_key, contracts in strike_map.items():
                    if not contracts:
                        continue
                    c = contracts[0]
                    gamma = c.get("gamma")
                    oi = c.get("openInterest")
                    if gamma is None or oi is None:
                        continue
                    try:
                        g, o = float(gamma), float(oi)
                        strike_f = float(strike_key)
                    except (TypeError, ValueError):
                        continue
                    if not (math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0):
                        continue
                    gex = g * o * 100.0 * ul_spot
                    if side == "call":
                        total_call_gex += gex
                        if strike_f > ul_spot and gex > best_call_wall[0]:
                            best_call_wall = (gex, strike_f)
                    else:
                        total_put_gex += gex
                        if strike_f < ul_spot and gex > best_put_wall[0]:
                            best_put_wall = (gex, strike_f)

        net = total_call_gex - total_put_gex
        gex_by_underlying[ul] = {
            "net_gex": round(net, 2),
            "gex_regime": "pinned" if net >= 0 else "volatile",
            "gex_support": best_put_wall[1] if math.isfinite(best_put_wall[1]) else None,
            "gex_resistance": best_call_wall[1] if math.isfinite(best_call_wall[1]) else None,
        }
```

**Step 3: Add GEX to per-position computed metrics**

In the position enrichment loop (around line 470-476 where `computed` dict is built), add:

```python
            ul = pos.get("underlying", "")
            gex_info = gex_by_underlying.get(ul, {})
            computed["net_gex"] = gex_info.get("net_gex")
            computed["gex_regime"] = gex_info.get("gex_regime")
            computed["gex_support"] = gex_info.get("gex_support")
            computed["gex_resistance"] = gex_info.get("gex_resistance")
```

**Step 4: Add math import if not already present**

Verify `import math` exists at the top of the file.

**Step 5: Commit**

```bash
git add uwos/schwab_position_analyzer.py
git commit -m "feat(gates): compute GEX per underlying in position analyzer"
```

---

## Task 10: Add base_live_cols for New Fields in run_mode_a_two_stage.py

**Files:**
- Modify: `uwos/run_mode_a_two_stage.py:757-788`

**Step 1: Add missing columns to base_live_cols**

After `"long_call_ask_live",` (around line 782), add:

```python
    "long_delta_live",
    "net_gex",
    "gex_regime",
    "gex_support",
    "gex_resistance",
```

Note: `short_delta_live`, `short_put_delta_live`, `short_call_delta_live` are already in the list.

**Step 2: Commit**

```bash
git add uwos/run_mode_a_two_stage.py
git commit -m "feat(gates): add long_delta_live and GEX columns to base_live_cols"
```

---

## Task 11: Smoke Test — Run Pipeline and Verify

**Step 1: Run the daily pipeline for 2026-03-13**

```bash
cd c:/uw_root && python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/2026-03-13" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "c:/uw_root/out/2026-03-13-test" \
  --top-trades 20
```

**Step 2: Verify new columns appear in output**

```bash
cd c:/uw_root && python -c "
import pandas as pd
df = pd.read_csv('out/2026-03-13-test/live_trade_table_2026-03-13_final.csv')
for col in ['short_delta_live','long_delta_live','net_gex','gex_regime','gex_support','gex_resistance']:
    vals = df[col].dropna() if col in df.columns else pd.Series()
    print(f'{col}: {len(vals)} populated / {len(df)} total')
"
```

Expected: Delta and GEX columns populated for Schwab-validated trades.

**Step 3: Verify blockers fire correctly**

```bash
cd c:/uw_root && python -c "
import pandas as pd
df = pd.read_csv('out/2026-03-13-test/live_trade_table_2026-03-13_final.csv')
blockers = df['approval_blockers'].fillna('')
for pattern in ['shield_sigma_fail', 'shield_gex_volatile', 'fire_gex_pinned', 'fire_delta_low', 'shield_delta_fail']:
    count = blockers.str.contains(pattern).sum()
    print(f'{pattern}: {count} trades blocked')
"
```

**Step 4: Verify shield_sigma_fail is now a hard blocker**

Check that no trade with `shield_sigma_fail` appears in Tactical book:

```bash
cd c:/uw_root && python -c "
import pandas as pd
df = pd.read_csv('out/2026-03-13-test/live_trade_table_2026-03-13_final.csv')
sigma_fail = df[df['approval_blockers'].fillna('').str.contains('shield_sigma_fail')]
tactical = sigma_fail[sigma_fail['execution_book'] == 'Tactical']
print(f'Trades with sigma_fail in Tactical: {len(tactical)} (should be 0)')
"
```

Expected: 0 trades with sigma_fail in Tactical.

**Step 5: Commit test output cleanup**

```bash
rm -rf c:/uw_root/out/2026-03-13-test
```

---

## Task 12: Final Commit — Tag the Implementation

**Step 1: Create a summary commit**

```bash
git add -A
git status
```

Verify only expected files changed. Do NOT commit if unexpected files appear.

---

## Summary of All File Changes

| File | Lines Modified | Changes |
|------|---------------|---------|
| `uwos/swing_trend_pipeline.py` | 247-290, 1497, 1612-1620, 1757, 1862, 2309-2361 | SwingScore fields, GEX computation, delta extraction, CSV columns |
| `uwos/run_mode_a_two_stage.py` | 757-788, 1215-1386 | base_live_cols, FIRE/SHIELD blockers, split_blockers classification, sigma hard block |
| `uwos/eod_trade_scan_mode_a.py` | 712, 793, 878, 992, 1232 | IVR conviction bonus/penalty (5 blocks) |
| `uwos/schwab_position_analyzer.py` | 354-360, 470-476 | Full chain fetch, GEX computation, output JSON enrichment |
| `uwos/rulebook_config_goal_holistic_claude.yaml` | 162-163 | Enable delta gate, add FIRE delta + GEX config |
