"""Self-contained HTML report for CodexSwing v0.3 candidate slates."""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return ("{:." + str(digits) + "%}").format(float(value))
    except (TypeError, ValueError):
        return "—"


def _money(value: Any) -> str:
    try:
        return "${:,.2f}".format(float(value))
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return ("{:,." + str(digits) + "f}").format(float(value))
    except (TypeError, ValueError):
        return "—"


def _list(values: Sequence[Any]) -> str:
    if not values:
        return "<li>None recorded</li>"
    return "".join("<li>{}</li>".format(_e(value)) for value in values)


def _safe_href(value: Any) -> str:
    text = str(value or "").strip()
    return _e(text) if text.startswith(("https://", "http://")) else "#"


def _stage_class(stage: Any) -> str:
    value = str(stage or "")
    if value == "MANUAL_READY":
        return "ready"
    if value == "TACTICAL_READY":
        return "tactical"
    return "blocked"


def _context_section(payload: Mapping[str, Any]) -> str:
    raw_context = payload.get("market_context")
    context = raw_context if isinstance(raw_context, Mapping) else {}
    raw_items = context.get("items")
    items = raw_items if isinstance(raw_items, list) else []
    rows = []
    for item in items[:30]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            "<tr><td><a href=\"{}\" rel=\"noreferrer\">{}</a></td><td>{}</td><td>{}</td></tr>".format(
                _safe_href(item.get("url")),
                _e(item.get("title")),
                _e(item.get("source_name") or item.get("source_country") or item.get("provider")),
                _e(item.get("published_at_utc")),
            )
        )
    if not rows:
        rows.append('<tr><td colspan="3">No time-aligned public items were available. This contributed zero to every score.</td></tr>')
    failures = context.get("upstream_failures")
    failure_values = failures if isinstance(failures, list) else []
    return """
      <section class="card">
        <span class="eyebrow">Shadow evidence · never a numeric vote</span>
        <h2>Public trend and geopolitical context</h2>
        <p><strong>Status:</strong> {status}. <strong>Provider:</strong> {provider}. {reason}</p>
        {failures}
        <div class="table-wrap"><table><thead><tr><th>Headline</th><th>Source</th><th>Published UTC</th></tr></thead><tbody>{rows}</tbody></table></div>
      </section>
    """.format(
        status=_e(context.get("status") or "UNAVAILABLE"),
        provider=_e(context.get("provider") or "None"),
        reason=_e(context.get("reason") or ""),
        failures=(
            "<p><strong>Upstream fallback note:</strong> {}</p>".format(
                _e("; ".join(str(value) for value in failure_values))
            )
            if failure_values
            else ""
        ),
        rows="".join(rows),
    )


def _candidate_row(idea: Mapping[str, Any]) -> str:
    option = idea.get("selected_option")
    values = option if isinstance(option, Mapping) else {}
    profitability = values.get("profitability")
    pop = profitability if isinstance(profitability, Mapping) else {}
    promotion = values.get("promotion")
    readiness = promotion if isinstance(promotion, Mapping) else {}
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    return """
      <tr>
        <td><strong>{ticker}</strong><br><small>{direction}</small></td>
        <td><span class="pill {stage_class}">{stage}</span></td>
        <td>{strategy}</td>
        <td><strong>{pop}</strong><br><small>{rating} · score {score}/100</small></td>
        <td>{hist}<br><small>n-eff {effective}</small></td>
        <td>{modeled}</td>
        <td>{ev}</td>
        <td>{risk}</td>
        <td>{limit}</td>
        <td>{blocker}</td>
      </tr>
    """.format(
        ticker=_e(idea.get("ticker")),
        direction=_e(idea.get("direction")),
        stage=_e(idea.get("promotion_stage")),
        stage_class=_stage_class(idea.get("promotion_stage")),
        strategy=_e(values.get("strategy") or "No current vertical"),
        pop=_pct(pop.get("estimated_probability_profitable")),
        rating=_e(pop.get("confidence_rating") or "INSUFFICIENT"),
        score=_e(pop.get("confidence_score_0_to_100") or 0),
        hist=_pct(pop.get("historical_holdout_pop")),
        effective=_e(pop.get("historical_effective_sample") or 0),
        modeled=_pct(pop.get("current_contract_modeled_pop")),
        ev=_money(values.get("modeled_expected_pnl_dollars")),
        risk=_money(values.get("maximum_loss_dollars")),
        limit=_e(values.get("entry_limit_display")),
        blocker=_e(blockers[0] if blockers else "None"),
    )


def _idea_card(idea: Mapping[str, Any]) -> str:
    option = idea.get("selected_option")
    values = option if isinstance(option, Mapping) else {}
    profitability = values.get("profitability")
    pop = profitability if isinstance(profitability, Mapping) else {}
    promotion = values.get("promotion")
    readiness = promotion if isinstance(promotion, Mapping) else {}
    forecasts = idea.get("orats_forecasts")
    forecast_values = forecasts if isinstance(forecasts, Mapping) else {}
    analog = idea.get("analog_evidence")
    analog_values = analog if isinstance(analog, Mapping) else {}
    stock = idea.get("stock_expression")
    stock_values = stock if isinstance(stock, Mapping) else {}
    sources = idea.get("source_contributions")
    source_values = sources if isinstance(sources, Mapping) else {}
    source_rows = []
    for name, raw in source_values.items():
        values_for_source = raw if isinstance(raw, Mapping) else {}
        seeded = values_for_source.get("seeded")
        seeded_values = seeded if isinstance(seeded, list) else []
        source_rows.append(
            "<tr><th>{}</th><td>{}</td><td>{}</td></tr>".format(
                _e(name),
                _e("; ".join(str(value) for value in seeded_values)),
                _e(values_for_source.get("reason") or values_for_source.get("current_session") or values_for_source.get("quote_session") or "current record"),
            )
        )
    if values and int(values.get("leg_count") or 2) == 1:
        legs = """
          <ol class="legs">
            <li><strong>BUY TO OPEN 1</strong> {long_symbol} — {right} {long_strike:g}, bid {long_bid}, ask {long_ask}, delta {long_delta}</li>
          </ol>
        """.format(
            long_symbol=_e(values.get("long_symbol")),
            right=_e(values.get("right")),
            long_strike=float(values.get("long_strike") or 0),
            long_bid=_num(values.get("long_bid")),
            long_ask=_num(values.get("long_ask")),
            long_delta=_num(values.get("long_delta")),
        )
    elif values:
        legs = """
          <ol class="legs">
            <li><strong>BUY 1</strong> {long_symbol} — {right} {long_strike:g}, bid {long_bid}, ask {long_ask}, delta {long_delta}</li>
            <li><strong>SELL 1</strong> {short_symbol} — {right} {short_strike:g}, bid {short_bid}, ask {short_ask}, delta {short_delta}</li>
          </ol>
        """.format(
            long_symbol=_e(values.get("long_symbol")),
            right=_e(values.get("right")),
            long_strike=float(values.get("long_strike") or 0),
            long_bid=_num(values.get("long_bid")),
            long_ask=_num(values.get("long_ask")),
            long_delta=_num(values.get("long_delta")),
            short_symbol=_e(values.get("short_symbol")),
            short_strike=float(values.get("short_strike") or 0),
            short_bid=_num(values.get("short_bid")),
            short_ask=_num(values.get("short_ask")),
            short_delta=_num(values.get("short_delta")),
        )
    else:
        legs = "<p>No exact current vertical passed the chain selector.</p>"
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    full_shortfalls = (
        readiness.get("full_evidence_shortfalls")
        if isinstance(readiness.get("full_evidence_shortfalls"), list)
        else []
    )
    diagnostics = readiness.get("portfolio_diagnostics")
    portfolio = diagnostics if isinstance(diagnostics, Mapping) else {}
    entry = values.get("entry_plan")
    entry_plan = entry if isinstance(entry, Mapping) else {}
    return """
      <article class="card">
        <header>
          <div><span class="eyebrow">{direction} · {ticker}</span><h2>{strategy}</h2></div>
          <span class="pill {stage_class}">{stage}</span>
        </header>
        <div class="metrics">
          <div><span>Estimated profitable POP</span><strong>{estimated_pop}</strong><small>{rating} · {score}/100</small></div>
          <div><span>ORATS holdout POP</span><strong>{historical_pop}</strong><small>Wilson lower {wilson}; effective n {effective}</small></div>
          <div><span>Current modeled POP</span><strong>{modeled_pop}</strong><small>scenario model, not calibrated</small></div>
          <div><span>Expected P/L after costs</span><strong>{ev}</strong><small>at {limit}</small></div>
          <div><span>Maximum loss</span><strong>{risk}</strong><small>defined risk, one contract package</small></div>
          <div><span>Quote quality</span><strong>{quote_width}</strong><small>max leg width; OI {oi}, volume {volume}</small></div>
        </div>
        <h3>Exact current expression</h3>
        {legs}
        <p><strong>Entry:</strong> {limit}; expiration {expiration} ({dte} DTE). On {entry_session}, require underlying trigger {trigger}, do not chase an opening gap beyond {gap_price}, and require the option limit to remain available. If entered, the fixed replay exit is by the {exit_session} regular-session close.</p>
        <h3>Why it is or is not executable by you</h3>
        <ul>{blockers}</ul>
        <p><strong>Evidence tier:</strong> {evidence_tier}. <strong>Maximum contracts:</strong> {max_contracts}. Full-evidence shortfalls: {full_shortfalls}</p>
        <table class="compact">
          <tr><th>Discovery</th><td>{discovered}</td><th>Backtest</th><td>{backtest}</td></tr>
          <tr><th>Current contract</th><td>{contract}</td><th>Portfolio</th><td>{portfolio_gate}</td></tr>
        </table>
        <h3>ORATS forecast semantics</h3>
        <p>Future realized/statistical vol: <strong>{realized}%</strong>. Future implied vol: <strong>{implied}%</strong>. Current 30-day implied vol: <strong>{current_iv}%</strong>. These are deliberately separate inputs.</p>
        <h3>Stock lane (separate from option validation)</h3>
        <p>{stock_side} stock trigger {stock_trigger}; invalidation {stock_stop}; empirical analog POP {stock_pop}, Wilson lower {stock_lower}. Status: <strong>{stock_status}</strong>.</p>
        <h3>What seeded this analysis</h3>
        <table><thead><tr><th>Source</th><th>Fields and decisions seeded</th><th>Boundary</th></tr></thead><tbody>{source_rows}</tbody></table>
        <h3>Portfolio diagnostics</h3>
        <p>Liquidation value {liquidation}; available funds {available}; current ticker concentration {concentration}; tactical one-contract risk cap {tactical_cap}; working-order conflict {working_conflict}.</p>
        <details><summary>Analog evidence and rejection counts</summary><pre>{analog}\n\nOption rejection counts:\n{rejections}</pre></details>
      </article>
    """.format(
        direction=_e(idea.get("direction")),
        ticker=_e(idea.get("ticker")),
        strategy=_e(values.get("strategy") or "No selected option"),
        stage=_e(idea.get("promotion_stage")),
        stage_class=_stage_class(idea.get("promotion_stage")),
        estimated_pop=_pct(pop.get("estimated_probability_profitable")),
        rating=_e(pop.get("confidence_rating") or "INSUFFICIENT"),
        score=_e(pop.get("confidence_score_0_to_100") or 0),
        historical_pop=_pct(pop.get("historical_holdout_pop")),
        wilson=_pct(pop.get("historical_wilson_95_lower_bound")),
        effective=_e(pop.get("historical_effective_sample") or 0),
        modeled_pop=_pct(pop.get("current_contract_modeled_pop")),
        ev=_money(values.get("modeled_expected_pnl_dollars")),
        limit=_e(values.get("entry_limit_display")),
        risk=_money(values.get("maximum_loss_dollars")),
        quote_width=_pct(values.get("maximum_leg_spread_pct")),
        oi=_e(values.get("minimum_open_interest") or 0),
        volume=_e(values.get("minimum_volume") or 0),
        legs=legs,
        expiration=_e(values.get("expiration")),
        dte=_e(values.get("dte")),
        entry_session=_e(entry_plan.get("entry_session")),
        trigger=_num(entry_plan.get("underlying_trigger")),
        gap_price=_num(entry_plan.get("maximum_open_gap_price")),
        exit_session=_e(entry_plan.get("planned_exit_session")),
        blockers=_list(blockers),
        evidence_tier=_e(readiness.get("evidence_tier")),
        max_contracts=_e(readiness.get("recommended_max_contracts") or (1 if values else None)),
        full_shortfalls=_e("; ".join(str(value) for value in full_shortfalls) or "None"),
        discovered=_e((readiness.get("gates") or {}).get("discovered")),
        backtest=_e((readiness.get("gates") or {}).get("backtest_pass")),
        contract=_e((readiness.get("gates") or {}).get("current_contract_pass")),
        portfolio_gate=_e((readiness.get("gates") or {}).get("portfolio_pass")),
        realized=_num(forecast_values.get("realized_vol_forecast_20d_pct")),
        implied=_num(forecast_values.get("implied_vol_forecast_20d_pct")),
        current_iv=_num(forecast_values.get("current_implied_vol_30d_pct")),
        stock_side=_e(stock_values.get("side")),
        stock_trigger=_num(stock_values.get("entry_trigger")),
        stock_stop=_num(stock_values.get("invalidation")),
        stock_pop=_pct(stock_values.get("empirical_pop")),
        stock_lower=_pct(stock_values.get("empirical_pop_wilson_lower")),
        stock_status=_e(stock_values.get("status")),
        source_rows="".join(source_rows),
        liquidation=_money(portfolio.get("liquidation_value_dollars")),
        available=_money(portfolio.get("available_funds_dollars")),
        concentration=_pct(portfolio.get("ticker_concentration")),
        tactical_cap=_money(portfolio.get("tactical_risk_cap_dollars")),
        working_conflict=_e(portfolio.get("working_order_conflict")),
        analog=_e(analog_values),
        rejections=_e(idea.get("option_rejection_counts")),
    )


def render_idea_html(payload: Mapping[str, Any]) -> str:
    ideas = [item for item in payload.get("ideas") or () if isinstance(item, Mapping)]
    rows = "".join(_candidate_row(item) for item in ideas)
    cards = "".join(_idea_card(item) for item in ideas)
    context_section = _context_section(payload)
    actionable = int(payload.get("actionable_trade_count") or 0)
    full_ready = int(payload.get("manual_ready_trade_count") or 0)
    tactical_ready = int(payload.get("tactical_ready_trade_count") or 0)
    status_class = "ready" if actionable else "blocked"
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodexSwing {date}</title>
<style>
:root{{--ink:#12212f;--muted:#667482;--paper:#f5f7f8;--card:#fff;--line:#dce2e7;--green:#08775d;--red:#ad3d3d;--blue:#185d8f;--amber:#9a6100}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1480px;margin:auto;padding:28px}}h1{{font-size:34px;margin:.15rem 0}}h2{{margin:.2rem 0;font-size:24px}}h3{{margin:24px 0 8px}}p{{max-width:1050px}}.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}.hero,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:18px;box-shadow:0 2px 10px #1020300a}}.hero{{border-left:7px solid var(--red)}}.hero.ready{{border-left-color:var(--green)}}.status{{font-weight:750;font-size:18px;color:var(--red)}}.hero.ready .status{{color:var(--green)}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:10px;vertical-align:top}}th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:14px 0 22px}}.pill{{display:inline-block;padding:4px 9px;border-radius:999px;font-size:12px;font-weight:750}}.pill.ready{{background:#d9f4e9;color:var(--green)}}.pill.tactical{{background:#fff0c9;color:var(--amber)}}.pill.blocked{{background:#f8e6df;color:var(--red)}}.card header{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:18px 0}}.metrics div{{border:1px solid var(--line);border-radius:10px;padding:12px}}.metrics span,.metrics small{{display:block;color:var(--muted)}}.metrics strong{{font-size:21px}}.legs{{background:#f2f6f9;border-radius:10px;padding:12px 12px 12px 35px}}.compact th,.compact td{{width:25%}}details{{margin-top:16px}}pre{{white-space:pre-wrap;word-break:break-word;background:#12212f;color:#eaf1f4;padding:14px;border-radius:9px}}footer{{color:var(--muted);font-size:13px;padding:10px 0 30px}}@media(max-width:800px){{main{{padding:14px}}.metrics{{grid-template-columns:1fr 1fr}}h1{{font-size:27px}}}}@media(max-width:520px){{.metrics{{grid-template-columns:1fr}}.card header{{display:block}}}}
</style></head><body><main>
<section class="hero {status_class}"><span class="eyebrow">CodexSwing v0.4 · {date}</span><h1>{headline}</h1><div class="status">{status}</div><p>{notice}</p><p><strong>{actionable} actionable</strong>: {full_ready} full-evidence and {tactical_ready} one-contract tactical, across {ideas} finalists drawn from {raw_universe} ORATS underlyings. Tactical means positive train/validation/holdout economics with a confidence interval that still overlaps zero; it is capped at 0.05% NAV. Manual submission only; no order was submitted.</p></section>
<h2>Decision slate</h2><div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Stage</th><th>Structure</th><th>Estimated POP / confidence</th><th>Historical POP</th><th>Modeled POP</th><th>EV</th><th>Max loss</th><th>Limit</th><th>First blocker</th></tr></thead><tbody>{rows}</tbody></table></div>
{context_section}
{cards}
<footer>Primary data: ORATS delayed/historical API and Schwab read-only API. Public context is shadow-only. Broker order authorization: {authorized}. Broker order submitted: {submitted}.</footer>
</main></body></html>""".format(
        date=_e(payload.get("as_of_date")),
        status_class=status_class,
        headline="Actionable manual trade candidates" if actionable else "No trade has cleared an actionable tier",
        status=_e(payload.get("status")),
        notice=_e(payload.get("risk_notice")),
        actionable=actionable,
        full_ready=full_ready,
        tactical_ready=tactical_ready,
        ideas=len(ideas),
        raw_universe="{:,}".format(int(payload.get("raw_orats_underlying_count") or 0)),
        rows=rows or '<tr><td colspan="10">No complete ideas.</td></tr>',
        context_section=context_section,
        cards=cards,
        authorized=_e(payload.get("broker_order_authorized")),
        submitted=_e(payload.get("broker_order_submitted")),
    )
