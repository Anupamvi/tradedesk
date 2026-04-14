"""
Exhaustive verdict test suite — tests every combination of position state.

Matrix: credit/debit/equity × ITM/ATM/OTM × DTE ranges × P&L ranges × earnings × delta × intraday change

This runs BEFORE any code ships. Not after.
"""
import sys

PASS = 0
FAIL = 0

def check(name, expected, pos):
    global PASS, FAIL
    from uwos.trade_monitor import compute_verdict
    v, reason = compute_verdict(pos)
    if v == expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name} -> got {v}, expected {expected}: {reason}", flush=True)


def credit_put(strike, ul, delta, pct_max, dte, pnl=0, pnl_pct=0, change=0, earn=999):
    return {"asset_type": "OPTION", "symbol": "T", "qty": -1, "put_call": "PUT",
            "strike": strike, "greeks": {"delta": delta},
            "underlying_quote": {"last": ul, "change_pct": change},
            "computed": {"pct_of_max_profit": pct_max, "unrealized_pnl": pnl,
                         "unrealized_pnl_pct": pnl_pct, "dte": dte, "days_to_earnings": earn}}

def credit_call(strike, ul, delta, pct_max, dte, pnl=0, pnl_pct=0, change=0, earn=999):
    return {"asset_type": "OPTION", "symbol": "T", "qty": -1, "put_call": "CALL",
            "strike": strike, "greeks": {"delta": delta},
            "underlying_quote": {"last": ul, "change_pct": change},
            "computed": {"pct_of_max_profit": pct_max, "unrealized_pnl": pnl,
                         "unrealized_pnl_pct": pnl_pct, "dte": dte, "days_to_earnings": earn}}

def debit_call(strike, ul, delta, dte, pnl=0, pnl_pct=0, change=0, earn=999):
    return {"asset_type": "OPTION", "symbol": "T", "qty": 1, "put_call": "CALL",
            "strike": strike, "greeks": {"delta": delta},
            "underlying_quote": {"last": ul, "change_pct": change},
            "computed": {"pct_of_max_profit": None, "unrealized_pnl": pnl,
                         "unrealized_pnl_pct": pnl_pct, "dte": dte, "days_to_earnings": earn}}

def debit_put(strike, ul, delta, dte, pnl=0, pnl_pct=0, change=0, earn=999):
    return {"asset_type": "OPTION", "symbol": "T", "qty": 1, "put_call": "PUT",
            "strike": strike, "greeks": {"delta": delta},
            "underlying_quote": {"last": ul, "change_pct": change},
            "computed": {"pct_of_max_profit": None, "unrealized_pnl": pnl,
                         "unrealized_pnl_pct": pnl_pct, "dte": dte, "days_to_earnings": earn}}

def equity(pnl, pnl_pct, change=0):
    return {"asset_type": "EQUITY", "symbol": "T", "qty": 100, "put_call": "", "strike": None,
            "greeks": None, "underlying_quote": {"last": 100, "change_pct": change},
            "computed": {"pct_of_max_profit": None, "unrealized_pnl": pnl,
                         "unrealized_pnl_pct": pnl_pct, "dte": None, "days_to_earnings": None},
            "market_value": 10000, "avg_cost": 100}


print("=" * 60, flush=True)
print("  VERDICT TEST SUITE", flush=True)
print("=" * 60, flush=True)

# ================================================================
# CREDIT PUT TESTS
# ================================================================
print("\n--- CREDIT PUT ---", flush=True)

# Near max profit
check("95% max 30 DTE",       "CLOSE",  credit_put(100, 120, -0.10, 95, 30))
check("90% max 45 DTE",       "CLOSE",  credit_put(100, 115, -0.15, 90, 45))
check("85% max 60 DTE",       "CLOSE",  credit_put(100, 112, -0.18, 85, 60))
check("84% max 60 DTE",       "CLOSE",  credit_put(100, 112, -0.18, 84, 60))  # 84 > 75% = CLOSE

# ITM + assignment risk
check("ITM 14% delta-0.90 3DTE", "CLOSE", credit_put(100, 86, -0.90, -200, 3))
check("ITM 14% delta-0.90 15DTE","ASSESS",credit_put(100, 86, -0.90, -200, 15))

# ITM + DTE < 14 = ROLL
check("ITM 5% 10 DTE",         "ROLL",   credit_put(100, 95, -0.60, -30, 10))
check("ITM 5% 13 DTE",         "ROLL",   credit_put(100, 95, -0.60, -30, 13))

# Pin risk
check("0.5% from strike 2 DTE", "CLOSE", credit_put(100, 100.5, -0.48, 40, 2))
check("0.5% from strike 10 DTE","HOLD",  credit_put(100, 100.5, -0.35, 40, 10)) # no pin risk at 10 DTE

# ITM + deep + any DTE
check("ITM 8% delta-0.58 68DTE","ASSESS",credit_put(140, 129, -0.58, -43, 68))  # PLTR scenario
check("ITM 10% delta-0.46 251DTE","ASSESS",credit_put(18, 16.2, -0.46, -24, 251))  # SOFI scenario
check("ITM 2% delta-0.52 40DTE","ASSESS",credit_put(100, 98, -0.52, -10, 40))
check("ITM 1% delta-0.40 40DTE","ASSESS",credit_put(100, 99, -0.40, -5, 40))  # ITM at all = ASSESS

# Expiration week gamma
check("30% max 5 DTE",         "CLOSE",  credit_put(100, 110, -0.20, 30, 5))
check("45% max 6 DTE",         "CLOSE",  credit_put(100, 108, -0.25, 45, 6))
check("55% max 5 DTE",         "CLOSE",  credit_put(100, 108, -0.25, 55, 5))  # 55>50 but DTE<7

# Earnings proximity
check("40% max earnings 5d",   "CLOSE",  credit_put(100, 110, -0.20, 40, 30, earn=5))
check("4% max earnings 3d",    "ASSESS", credit_put(100, 105, -0.30, 4, 30, earn=3))  # too little profit for CLOSE
check("40% max earnings 7d",   "CLOSE",  credit_put(100, 110, -0.20, 40, 30, earn=7))
check("40% max earnings 8d",   "HOLD",   credit_put(100, 110, -0.20, 40, 30, earn=8))
check("-20% max earnings 5d",  "ASSESS", credit_put(100, 103, -0.35, -20, 30, earn=5))

# 75% target
check("76% max 30 DTE",        "CLOSE",  credit_put(100, 115, -0.15, 76, 30))
check("74% max 30 DTE",        "HOLD",   credit_put(100, 114, -0.18, 74, 30))

# 50% max + low DTE
check("55% max 8 DTE",         "CLOSE",  credit_put(100, 110, -0.20, 55, 8))
check("55% max 12 DTE",        "HOLD",   credit_put(100, 110, -0.20, 55, 12))

# High delta approaching ATM
check("delta -0.48 20 DTE",    "ASSESS", credit_put(100, 101, -0.48, 10, 20))
check("delta -0.44 20 DTE",    "HOLD",   credit_put(100, 102, -0.44, 15, 20))

# Deep loss
check("-85% max 60 DTE",       "ASSESS", credit_put(100, 95, -0.60, -85, 60))
check("-79% max 60 DTE ITM",   "ASSESS", credit_put(100, 97, -0.55, -79, 60))  # ul<strike = ITM, delta>0.50

# Normal hold
check("40% max 40 DTE OTM",    "HOLD",   credit_put(100, 110, -0.25, 40, 40))

# ================================================================
# CREDIT CALL TESTS
# ================================================================
print("\n--- CREDIT CALL ---", flush=True)

check("ITM 10% call 60 DTE",   "ASSESS", credit_call(100, 110, 0.70, -80, 60))
check("95% max call",          "CLOSE",  credit_call(100, 80, 0.05, 95, 30))
check("ITM call 5 DTE",        "ROLL",   credit_call(100, 105, 0.65, -50, 5))

# ================================================================
# DEBIT CALL TESTS
# ================================================================
print("\n--- DEBIT CALL ---", flush=True)

# OTM > 5% + DTE < 35
check("OTM 8% 20 DTE",         "CLOSE",  debit_call(200, 185, 0.15, 20, -300, -55))
check("OTM 6% 30 DTE",         "CLOSE",  debit_call(200, 188, 0.20, 30, -200, -35))
check("OTM 6% 40 DTE",         "HOLD",   debit_call(200, 188, 0.20, 40, -100, -15))  # DTE > 35

# OTM any + DTE < 14
check("OTM 1% 10 DTE",         "CLOSE",  debit_call(200, 198, 0.45, 10, -50, -10))
check("OTM 1% 20 DTE",         "HOLD",   debit_call(200, 198, 0.45, 20, -50, -10))

# Down > 60%
check("down 65% 50 DTE",       "CLOSE",  debit_call(200, 180, 0.10, 50, -650, -65))

# Down > 40%
check("down 45% 50 DTE",       "ASSESS", debit_call(200, 185, 0.15, 50, -450, -45))

# Earnings IV crush
check("debit earnings 3d +20%","CLOSE",  debit_call(200, 195, 0.40, 30, 100, 20, earn=3))
check("debit earnings 3d -30%","ASSESS", debit_call(200, 210, 0.60, 30, -150, -30, earn=3))
check("debit earnings 10d",    "HOLD",   debit_call(200, 195, 0.40, 30, 100, 20, earn=10))

# ITM working
check("ITM +40% 30 DTE",       "HOLD",   debit_call(200, 210, 0.65, 30, 200, 40))

# ================================================================
# DEBIT PUT TESTS
# ================================================================
print("\n--- DEBIT PUT ---", flush=True)

check("OTM 10% 15 DTE",        "CLOSE",  debit_put(80, 100, -0.10, 15, -300, -70))
check("ITM working",           "HOLD",   debit_put(100, 90, -0.65, 30, 200, 40))

# ================================================================
# EQUITY TESTS
# ================================================================
print("\n--- EQUITY ---", flush=True)

# Intraday crash
check("today -8%",             "CLOSE",  equity(-2000, -20, change=-8))
check("today -6%",             "ASSESS", equity(-2000, -20, change=-6))
check("today -4%",             "HOLD",   equity(-2000, -20, change=-4))

# Cumulative thresholds
check("down 65%",              "CLOSE",  equity(-6500, -65))
check("down 55%",              "CLOSE",  equity(-5500, -55))
check("down 45%",              "CLOSE",  equity(-4500, -45))
check("down 35%",              "ASSESS", equity(-3500, -35))
check("down 25%",              "ASSESS", equity(-2500, -25))
check("down 24%",              "HOLD",   equity(-2400, -24))
check("down 10%",              "HOLD",   equity(-1000, -10))

# Take profit
check("up 120%",               "ASSESS", equity(12000, 120))
check("up 100%",               "ASSESS", equity(10000, 100))
check("up 99%",                "HOLD",   equity(9900, 99))
check("up 55%",                "HOLD",   equity(5500, 55))

# ================================================================
# BOUNDARY TESTS (audit v6)
# ================================================================
print("\n--- BOUNDARY TESTS ---", flush=True)

# Credit: 50% max at DTE boundary
check("50% max 10 DTE",        "CLOSE",  credit_put(100, 110, -0.20, 50, 10))
check("50% max 11 DTE",        "HOLD",   credit_put(100, 110, -0.20, 50, 11))

# Credit call: ITM assignment risk
check("call ITM delta+0.90 3DTE","CLOSE", credit_call(100, 115, 0.90, -200, 3))
check("call pin risk 2 DTE",    "CLOSE", credit_call(100, 99.5, 0.48, 40, 2))

# Debit: OTM 4% with DTE 30 = ASSESS (3-5% range)
check("debit OTM 4% 30 DTE",   "ASSESS", debit_call(200, 192, 0.25, 30, -150, -30))

# Debit: exactly -60% = CLOSE (boundary)
check("debit exactly -60%",    "CLOSE",  debit_call(200, 180, 0.10, 50, -600, -60))

# Equity: exactly -40% = CLOSE
check("equity exactly -40%",   "CLOSE",  equity(-4000, -40))

# Equity: exactly -25% = ASSESS
check("equity exactly -25%",   "ASSESS", equity(-2500, -25))

# ================================================================
# NULL/EDGE CASES
# ================================================================
print("\n--- NULL/EDGE ---", flush=True)

check("all None credit",       "HOLD",   credit_put(None, None, 0, 0, None))
check("zero everything",       "CLOSE",  credit_put(0, 0, 0, 0, 0))  # DTE=0 triggers expiry week
check("negative DTE",          "HOLD",   credit_put(100, 110, -0.20, 40, -1))

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'=' * 60}", flush=True)
total = PASS + FAIL
print(f"  PASS: {PASS}/{total}  FAIL: {FAIL}/{total}", flush=True)
if FAIL == 0:
    print("  ALL TESTS PASSED", flush=True)
else:
    print(f"  {FAIL} FAILURES — FIX BEFORE SHIPPING", flush=True)
print(f"{'=' * 60}", flush=True)
sys.exit(1 if FAIL else 0)
