"""Self-contained HTML calculator. Generated from sleeve constants."""

from __future__ import annotations

import json
from pathlib import Path

from compoundcore.sleeve import public_snapshot


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HTML = ROOT / "web" / "calculator.html"


def calculator_html() -> str:
    blob = json.dumps(public_snapshot(), indent=2)
    return _TEMPLATE.replace("__SNAPSHOT__", blob)


def write_calculator(path: Path | None = None) -> Path:
    dest = path or DEFAULT_HTML
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(calculator_html(), encoding="utf-8")
    return dest


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Compound Core calculator</title>
  <style>
    :root {
      --ink: #1b1f1a;
      --muted: #5c6358;
      --paper: #f4f1ea;
      --card: #fffcf6;
      --line: #d7d0c4;
      --accent: #1f5c45;
      --accent-2: #8a4b12;
      --warn: #7a2e1f;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--paper); color: var(--ink); font: 16px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; }
    main { max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { font-size: 1.85rem; letter-spacing: -0.02em; margin: 0 0 8px; }
    .lede { color: var(--muted); max-width: 46rem; margin: 0 0 24px; }
    .box { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 18px 12px; margin-bottom: 18px; }
    label { display: block; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 4px; }
    .inputs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    input[type="text"], input[type="number"] {
      width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px;
      font: 1.05rem/1.2 ui-monospace, "SF Mono", Menlo, Consolas, monospace; background: #fff;
    }
    .hint { font-size: 0.85rem; color: var(--muted); margin-top: 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    h2 { font-size: 1.15rem; margin: 0 0 10px; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); font-weight: 600; }
    td.num, th.num { text-align: right; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.92rem; }
    .meta { font-size: 0.9rem; color: var(--muted); margin: 10px 0 0; }
    .meta strong { color: var(--ink); }
    .bar { height: 8px; background: #ece6db; border-radius: 99px; overflow: hidden; margin: 8px 0 14px; display: flex; }
    .seg { height: 100%; }
    .disclaimer { border-left: 3px solid var(--warn); padding: 10px 14px; background: #f8eee9; color: #4d2a22; font-size: 0.92rem; }
    footer { margin-top: 28px; color: var(--muted); font-size: 0.85rem; }
    @media (max-width: 860px) {
      .inputs, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Compound Core calculator</h1>
    <p class="lede">Enter a dollar amount. Both sleeves split it the same way the playbook does — default 48/10/7/5/20/5/5 and the capped aggressive variant. Optional weekly buy and monthly contribution fill the recipe and the 5-year / 10-year paths. No orders are placed.</p>

    <div class="box">
      <div class="inputs">
        <div>
          <label for="amount">Amount to allocate ($)</label>
          <input id="amount" type="text" inputmode="decimal" value="100000" autocomplete="off">
        </div>
        <div>
          <label for="weekly">Weekly buy ($)</label>
          <input id="weekly" type="text" inputmode="decimal" value="0" autocomplete="off">
        </div>
        <div>
          <label for="monthly">Monthly add for projections ($)</label>
          <input id="monthly" type="text" inputmode="decimal" value="1000" autocomplete="off">
        </div>
      </div>
      <p class="hint">Weekly 0 still shows a per-$1,000 recipe. Monthly 0 is lump-sum only. Bands are ±25% of target with a 2 percentage-point floor.</p>
    </div>

    <p class="disclaimer"><strong>Not financial advice.</strong> Capital-market assumptions are hypothetical geometric rates, not guarantees. A trailing 40% year is not a 5-year or 10-year budget. This sleeve is the long-term core; trading desks never touch it.</p>

    <div class="grid" id="sleeves"></div>
    <footer>As-of <span id="asof"></span>. Fees, NVDA look-through, and forward rates are research snapshots, not live quotes.</footer>
  </main>
  <script id="snapshot" type="application/json">__SNAPSHOT__</script>
  <script>
    const SNAP = JSON.parse(document.getElementById("snapshot").textContent);
    const COLORS = {VOO:"#1f5c45",VGT:"#3d7a63",SMH:"#8a4b12",VB:"#6b7c3a",VXUS:"#2c4c7c",GLDM:"#c4a35a",VGSH:"#7a7f78"};
    document.getElementById("asof").textContent = SNAP.asof;

    function moneyNum(s) {
      const n = Number(String(s).replace(/[$,\s]/g, ""));
      return Number.isFinite(n) ? n : 0;
    }
    function usd(n) {
      const abs = Math.abs(n);
      const opts = abs >= 1000 ? {maximumFractionDigits: 0} : {minimumFractionDigits: 2, maximumFractionDigits: 2};
      return (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString("en-US", opts);
    }
    function k(n) {
      const v = Math.round(n / 1000);
      if (Math.abs(v) >= 1000) return "$" + (v / 1000).toFixed(2) + "M";
      return "$" + v + "k";
    }
    function pct(x, d) { return (x * 100).toFixed(d) + "%"; }
    function cents(amount) { return Math.round(amount * 100); }
    function allocate(amount, weights) {
      const total = cents(amount);
      const tickers = SNAP.tickers;
      const raw = {};
      const floors = {};
      tickers.forEach(t => { raw[t] = total * weights[t]; floors[t] = Math.floor(raw[t]); });
      let leftover = total - tickers.reduce((s, t) => s + floors[t], 0);
      const ranked = tickers.slice().sort((a, b) => {
        const fa = raw[a] - floors[a];
        const fb = raw[b] - floors[b];
        if (fb !== fa) return fb - fa;
        return tickers.indexOf(a) - tickers.indexOf(b);
      });
      for (let i = 0; i < leftover; i++) floors[ranked[i]] += 1;
      const out = {};
      tickers.forEach(t => { out[t] = floors[t] / 100; });
      return out;
    }
    function monthlyRate(annual) { return Math.pow(1 + annual, 1 / 12) - 1; }
    function fvDca(pv, pmt, annual, years) {
      const n = Math.round(years * 12);
      const r = monthlyRate(annual);
      const lump = pv * Math.pow(1 + annual, years);
      const ann = Math.abs(r) < 1e-15 ? pmt * n : pmt * (Math.pow(1 + r, n) - 1) / r;
      return lump + ann;
    }
    function real(nom, years) { return nom / Math.pow(1 + SNAP.inflation, years); }

    function render() {
      const amount = Math.max(0, moneyNum(document.getElementById("amount").value));
      const weekly = Math.max(0, moneyNum(document.getElementById("weekly").value));
      const monthly = Math.max(0, moneyNum(document.getElementById("monthly").value));
      const host = document.getElementById("sleeves");
      host.innerHTML = "";
      ["default", "aggressive"].forEach(name => {
        const s = SNAP.sleeves[name];
        const lump = allocate(amount, s.weights);
        const weekAmt = weekly > 0 ? weekly : 1000;
        const week = allocate(weekAmt, s.weights);
        const card = document.createElement("section");
        card.className = "box";
        const title = name === "default" ? "Default sleeve" : "Aggressive variant";
        const bar = SNAP.tickers.map(t =>
          `<span class="seg" style="width:${s.weights[t]*100}%;background:${COLORS[t]}" title="${t}"></span>`
        ).join("");
        const weekLabel = weekly > 0 ? "Weekly" : "Per $1,000/wk";
        let rows = SNAP.tickers.map(t => {
          const b = s.bands[t];
          return `<tr>
            <td><strong>${t}</strong><div class="meta" style="margin:0">${SNAP.roles[t]}</div></td>
            <td class="num">${pct(s.weights[t], 0)}</td>
            <td class="num">${usd(lump[t])}</td>
            <td class="num">${usd(week[t])}</td>
            <td class="num">${pct(b.low,1)}–${pct(b.high,1)}</td>
          </tr>`;
        }).join("");
        const p5 = s.rates["5y"];
        const p10 = s.rates["10y"];
        const paths = [
          ["Stress", p5.stress, p10.stress],
          ["Bear", p5.bear, p10.bear],
          ["Base", p5.base, p10.base],
          ["Bull", p5.bull, p10.bull],
          ["Fantasy 40%/yr", SNAP.fantasy, SNAP.fantasy],
        ];
        const proj = paths.map(([label, a5, a10]) => {
          const strong = label === "Base";
          return `<tr>
            <td>${strong ? "<strong>"+label+"</strong>" : label}</td>
            <td class="num">${k(fvDca(amount, monthly, a5, 5))}</td>
            <td class="num">${k(fvDca(amount, monthly, a10, 10))}</td>
          </tr>`;
        }).join("");
        const voo = fvDca(amount, monthly, SNAP.voo_only_10y, 10);
        const base10 = fvDca(amount, monthly, p10.base, 10);
        const stress10 = fvDca(amount, monthly, p10.stress, 10);
        card.innerHTML = `
          <h2>${title}</h2>
          <div class="bar">${bar}</div>
          <table>
            <thead><tr><th>Ticker</th><th class="num">Weight</th><th class="num">Dollars</th><th class="num">${weekLabel}</th><th class="num">Band</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <p class="meta">Fee <strong>${(s.fee*100).toFixed(3)}%/yr</strong>. Look-through NVDA <strong>${pct(s.nvda,1)}</strong> (${usd(amount*s.nvda)}). US share of equities <strong>${pct(s.us_of_equity,0)}</strong>. SMH −45% ≈ <strong>${usd(amount*s.smh_crash_hit)}</strong>. VGSH dry powder <strong>${usd(lump.VGSH)}</strong>.</p>
          <h2 style="margin-top:18px">5-year / 10-year paths</h2>
          <table>
            <thead><tr><th>Path</th><th class="num">5-year</th><th class="num">10-year</th></tr></thead>
            <tbody>
              ${proj}
              <tr><td>VOO-only at VG midpoint 5.2% 10y</td><td class="num">—</td><td class="num">${k(voo)}</td></tr>
            </tbody>
          </table>
          <p class="meta">Base 10y real (2% inflation) ≈ <strong>${k(real(base10,10))}</strong>. Stress 10y real ≈ <strong>${k(real(stress10,10))}</strong>. ${monthly ? "Includes "+usd(monthly)+" per month." : "Lump sum only."}</p>
        `;
        host.appendChild(card);
      });
    }
    ["amount", "weekly", "monthly"].forEach(id => {
      document.getElementById(id).addEventListener("input", render);
    });
    render();
  </script>
</body>
</html>
"""
