import pandas as pd
import glob
import os
import sys
import re
import math
import numpy as np
import argparse
import zipfile
from datetime import datetime, timedelta

# Force UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 📜 CONFIGURATION (V53 - TRUE MARKET MATH)
# ==========================================
# ==========================================
# 🛠️ VIX AUTO-FETCHER
# ==========================================
def get_live_vix():
    try:
        import yfinance as yf
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        vix = yf.Ticker("^VIX")
        # Try different keys usually found in yfinance data
        val = vix.info.get('regularMarketPrice') or vix.info.get('currentPrice') or vix.history(period='1d')['Close'].iloc[-1]
        print(f"   🌊 LIVE VIX FETCHED: {val:.2f}")
        return float(val)
    except:
        print("   ⚠️ VIX FETCH FAILED. Using Default: 15.0")
        return 15.0

# ==========================================
# 📜 CONFIGURATION (V53 - DYNAMIC VIX)
# ==========================================
CURRENT_VIX = get_live_vix()  # <--- Now it's automatic

# --- EDIT #1: PASTE THIS WHOLE BLOCK (NEW CODE) ---
def calculate_kelly_size(account_balance, win_rate, risk_reward_ratio):
    # Kelly Formula: W - (1-W)/R
    kelly_pct = win_rate - ((1 - win_rate) / risk_reward_ratio)
    # Safety: Half-Kelly
    safe_pct = kelly_pct * 0.5
    # Constraints: Max 10% risk
    final_pct = max(0.0, min(safe_pct, 0.10)) 
    dollar_amount = account_balance * final_pct
    return round(dollar_amount, 2)

# GATES
RULE_MIN_VOLUME = 1500 
RULE_MIN_OI = 100
RULE_MIN_PRICE = 12.0
RULE_MIN_CAP = 1_000_000_000 
RULE_MIN_IV = 15.0

# DTE WINDOWS
DTE_MIN_FIRE = 14
DTE_MAX_FIRE = 45
DTE_MIN_SHIELD = 21
DTE_MAX_SHIELD = 60
DTE_BRIDGE = 45 

# CREDIT RULES
CREDIT_RATIO_STD = 0.25
CREDIT_RATIO_LOW_VOL = 0.20
CREDIT_RATIO_DIRECTIONAL = 0.15  # V54: Lower threshold for DP-supported directional trades

# LISTS
INDEX_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'SPX', 'NDX', 'RUT', 'VIX']
VIP_LIST = ['AAPL', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'AMZN', 'GOOG', 'GOOGL', 'MSFT', 'META']
HIGH_BETA_LIST = ['TSLA', 'MSTR', 'COIN', 'NVDA', 'MU', 'SMCI'] 

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = "temp_workspace"
DATA_DIR = None
DATE_TAG = None
OUTPUT_FILE = "FINAL_MASTER_STRATEGY.md"

# ==========================================
# 🛠️ DATA UTILS
# ==========================================
def scan_directory(data_dir):
    if not data_dir or not os.path.exists(data_dir):
        return None
    base = os.path.basename(os.path.normpath(data_dir))
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if date_pattern.fullmatch(base):
        return base
    all_files = glob.glob(os.path.join(data_dir, "*"))
    dates = []
    for f in all_files:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if match:
            dates.append(match.group(1))
    return max(dates) if dates else None

def scan_date_dirs(root_dir):
    if not os.path.exists(root_dir):
        return []
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    dates = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and date_pattern.fullmatch(name):
            dates.append(name)
    return sorted(dates)

def resolve_data_location(date_arg=None, data_dir_arg=None):
    root = os.path.abspath(DATA_ROOT)
    if data_dir_arg:
        data_dir = data_dir_arg
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(root, data_dir)
        if not os.path.exists(data_dir):
            return None, None
        date_tag = date_arg or scan_directory(data_dir)
        return data_dir, date_tag
    if date_arg:
        candidate = os.path.join(root, date_arg)
        if os.path.isdir(candidate):
            return candidate, date_arg
        data_dir = os.path.join(root, DEFAULT_DATA_DIR)
        return data_dir, date_arg
    date_dirs = scan_date_dirs(root)
    if date_dirs:
        date_tag = max(date_dirs)
        return os.path.join(root, date_tag), date_tag
    data_dir = os.path.join(root, DEFAULT_DATA_DIR)
    return data_dir, scan_directory(data_dir)

def find_file_smart(keywords, date_tag, data_dir):
    if not data_dir or not os.path.exists(data_dir):
        return None
    if isinstance(keywords, str):
        keywords = [keywords]
    all_files = glob.glob(os.path.join(data_dir, "*"))
    matches = []
    for f in all_files:
        if not os.path.isfile(f):
            continue
        name = os.path.basename(f).lower()
        if not (name.endswith(".csv") or name.endswith(".zip")):
            continue
        if date_tag and date_tag not in name:
            continue
        if any(k in name for k in keywords):
            matches.append(f)
    if not matches:
        if date_tag:
            for f in all_files:
                if not os.path.isfile(f):
                    continue
                name = os.path.basename(f).lower()
                if not (name.endswith(".csv") or name.endswith(".zip")):
                    continue
                if any(k in name for k in keywords):
                    matches.append(f)
        if not matches:
            return None
    return max(matches, key=os.path.getmtime)

def parse_market_cap(val):
    if pd.isnull(val): return 0.0
    val = str(val).upper().replace('$', '').strip()
    multiplier = 1.0
    if val.endswith('T'): multiplier = 1_000_000_000_000; val = val[:-1]
    elif val.endswith('B'): multiplier = 1_000_000_000; val = val[:-1]
    elif val.endswith('M'): multiplier = 1_000_000; val = val[:-1]
    try: return float(val) * multiplier
    except: return 0.0

def select_zip_member(members, role=None):
    if role:
        role_l = role.lower()
        role_hints = {
            "chain": ["chain", "oi"],
            "whale": ["whale"],
            "screener": ["screener", "stock"],
            "dark_pool": ["dp", "dark_pool", "dark-pool", "dark"],
            "hot": ["hot"]
        }
        hints = role_hints.get(role_l, [role_l])
        for m in members:
            if any(h in m.lower() for h in hints):
                return m
    return sorted(members, key=lambda x: len(x), reverse=True)[0]

def read_csv_auto(path, role=None):
    if not path:
        return None
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV files found in zip: {path}")
            chosen = select_zip_member(members, role=role)
            with zf.open(chosen) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def fetch_live_price(symbol):
    try:
        import yfinance as yf
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        t = yf.Ticker(symbol)
        if 'currentPrice' in t.info: return float(t.info['currentPrice'])
        hist = t.history(period="1d")
        if not hist.empty: return float(hist['Close'].iloc[-1])
    except: pass 
    return 0.0

def parse_occ_symbols(df, occ_col='option_symbol'):
    """Parse OCC/OPRA option symbols: ROOT + YYMMDD + C/P + STRIKE*1000 (padded)"""
    def parse_single(sym):
        if pd.isnull(sym) or len(str(sym)) < 15: return None, None, None, None
        sym = str(sym).upper()
        # Find the C or P that separates date from strike
        for i in range(len(sym)-1, 5, -1):
            if sym[i] in ['C', 'P']:
                root = sym[:i-6]  # Everything before YYMMDD
                date_part = sym[i-6:i]  # YYMMDD
                opt_type = sym[i]  # C or P
                strike_str = sym[i+1:]  # Strike * 1000 padded
                try:
                    expiry = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]}"
                    strike = float(strike_str) / 1000.0
                    return root, expiry, opt_type, strike
                except: return None, None, None, None
        return None, None, None, None
    
    parsed = df[occ_col].apply(parse_single)
    sym_parsed = parsed.apply(lambda x: x[0] if x[0] else None)
    exp_parsed = parsed.apply(lambda x: x[1])
    type_parsed = parsed.apply(lambda x: x[2])

    if 'Symbol' in df.columns:
        existing_symbol = df['Symbol']
    else:
        existing_symbol = pd.Series([""] * len(df), index=df.index)
    df['Symbol'] = sym_parsed.fillna(existing_symbol)

    if 'Expiry' in df.columns:
        df['Expiry'] = df['Expiry'].where(df['Expiry'].notna(), exp_parsed)
    else:
        df['Expiry'] = exp_parsed

    if 'Type' in df.columns:
        df['Type'] = df['Type'].where(df['Type'].notna(), type_parsed)
    else:
        df['Type'] = type_parsed
    if 'Strike' not in df.columns or df['Strike'].isnull().all():
        df['Strike'] = parsed.apply(lambda x: x[3])
    return df

def smart_normalize(df, file_type="generic"):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        if col in ['underlying_symbol', 'symbol', 'ticker', 'root']: rename_map[col] = 'Symbol'
        if col in ['last_ask', 'ask']: rename_map[col] = 'Ask'
        if col in ['last_bid', 'bid']: rename_map[col] = 'Bid'
        if col in ['volume', 'curr_vol', 'vol']: rename_map[col] = 'Volume'
        if col in ['curr_oi', 'open_interest', 'oi']: rename_map[col] = 'OI'
        if col in ['dte', 'days_to_expiry']: rename_map[col] = 'DTE'
        if col in ['strike']: rename_map[col] = 'Strike'
        if col in ['avg_vol', '30d_avg_vol']: rename_map[col] = 'AvgVol'
        if col in ['iv_rank', 'ivr']: rename_map[col] = 'IVRank' 
        if col in ['oi_change', 'oi_chg']: rename_map[col] = 'OIChange' 
        if col in ['side', 'trade_side']: rename_map[col] = 'Side' 
        if col in ['put_call_ratio', 'pcr']: rename_map[col] = 'PCR'
        if col in ['market_cap', 'marketcap', 'cap']: rename_map[col] = 'MarketCap'
        if col in ['implied_volatility', 'iv', 'iv30']: rename_map[col] = 'IV'
        if col in ['premium', 'value', 'amount']: rename_map[col] = 'Premium'
        if col in ['option_type', 'type']: rename_map[col] = 'Type'
        if file_type == "dark_pool" and col in ['price', 'level', 'print_price']: rename_map[col] = 'StockPrice'
        elif file_type == "chain" and col in ['price', 'last', 'last_fill']: rename_map[col] = 'OptionLast'
        elif file_type == "chain" and col in ['stock_price', 'underlying_last']: rename_map[col] = 'StockPrice'

    df.rename(columns=rename_map, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Symbol' in df.columns: df['Symbol'] = df['Symbol'].astype(str).str.upper().str.strip()
    
    # FIX V55: Parse OCC option symbols (e.g., ORCL260220P00175000) to extract Symbol/Expiry/Type/Strike
    if 'option_symbol' in [c.lower() for c in df.columns]:
        occ_col = [c for c in df.columns if c.lower() == 'option_symbol'][0]
        df = parse_occ_symbols(df, occ_col)
    
    return df

def calculate_expiry(row, file_date_str):
    if pd.notnull(row.get('Expiry')): return row['Expiry']
    if pd.notnull(row.get('DTE')) and file_date_str:
        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        return (file_date + timedelta(days=row['DTE'])).strftime('%Y-%m-%d')
    return "N/A"

def get_1sigma_move(price, iv, dte):
    if price <= 0 or iv <= 0 or dte <= 0: return 0
    return price * (iv / 100.0) * math.sqrt(dte / 365.0)

# ==========================================
# 🧠 CORE LOGIC (V52)
# ==========================================
REJECT_LOG = {}

def log_fail(reason, ticker=None):
    REJECT_LOG[reason] = REJECT_LOG.get(reason, 0) + 1
    return None

def get_dynamic_width(price):
    if price > 2000: return 50.0 
    if price > 1000: return 25.0
    if price > 400: return 25.0 
    if price > 150: return 15.0  
    if price > 75: return 10.0
    if price > 25: return 5.0
    return 2.5

def calculate_conviction(row, momentum, iv_rank, is_vip, is_index):
    score = 50 
    vol = row.get('Volume', 0)
    avg_vol = row.get('AvgVol', 0)
    if avg_vol > 0 and vol > (3 * avg_vol): score += 15
    elif vol > 5000: score += 10
    
    oi_chg = row.get('OIChange', 0)
    if oi_chg > 20: score += 10 
    
    if iv_rank > 60: score += 10
    if momentum == "YES": score += 15
    if is_vip or is_index: score += 10
    return min(score, 95)

def analyze_flow_unanimity(df_whales, ticker):
    if df_whales is None or ticker not in df_whales['Symbol'].values: return 0.0
    w = df_whales[df_whales['Symbol'] == ticker]
    if w.empty: return 0.0
    net_bull_score, total_prem = 0.0, 0.0
    for _, row in w.iterrows():
        prem = float(row.get('Premium', 0))
        otype = str(row.get('Type', '')).upper()
        side = str(row.get('Side', '')).upper()
        is_bullish = False
        if 'C' in otype: is_bullish = False if 'BID' in side else True 
        elif 'P' in otype: is_bullish = True if 'BID' in side else False 
        if is_bullish: net_bull_score += prem
        else: net_bull_score -= prem
        total_prem += prem
    if total_prem == 0: return 0.0
    return net_bull_score / total_prem

def determine_flow_sentiment(row):
    side = str(row.get('Side', '')).upper()
    trade_type = str(row.get('Type', '')).upper()
    if 'C' in trade_type: return "BEARISH" if 'BID' in side else "BULLISH"
    if 'P' in trade_type: return "BULLISH" if 'BID' in side else "BEARISH"
    return "NEUTRAL"

def build_trade(row, flow_score, dp_lvl, momentum, iv_rank, earnings_data, account_size):
    ticker = row['Symbol']
    strike = float(row['Strike'])
    stock_price = float(row.get('StockPrice', 0))
    is_index = ticker in INDEX_ETFS
    is_vip = ticker in VIP_LIST
    is_high_beta = ticker in HIGH_BETA_LIST
    iv_percent = float(row.get('IV', 0)) if pd.notnull(row.get('IV')) else 0
    
    if pd.isna(stock_price) or stock_price <= 0: return log_fail(f"G0: Invalid Price", ticker)
    if not is_index and ticker not in VIP_LIST and stock_price < RULE_MIN_PRICE: 
        return log_fail(f"G0: Price < ${RULE_MIN_PRICE}", ticker)
        
    # V55: Use correct column names from chain data
    ask = row.get('last_ask', row.get('Ask', 0))
    bid = row.get('last_bid', row.get('Bid', 0))
    if ask > 0 and bid > 0:
        raw_price = (ask + bid) / 2
        spread_tol = 0.50 if is_index else 0.40
        if (ask - bid) > (raw_price * spread_tol): return log_fail("G1: Wide Spread", ticker)
    elif 'avg_price' in row and row.get('avg_price', 0) > 0:
        raw_price = float(row['avg_price'])
    elif 'OptionLast' in row:
        raw_price = float(row['OptionLast'])
    else: return log_fail("G1: No Price", ticker)

    if raw_price > (strike * 1.5) and not is_index: return log_fail("G1: Deep ITM", ticker)

    if row['Expiry'] == "N/A": return log_fail("G2: No Expiry", ticker)
    expiry_dt = datetime.strptime(row['Expiry'], '%Y-%m-%d')
    file_dt = datetime.strptime(DATE_TAG, "%Y-%m-%d")
    days_from_file = (expiry_dt - file_dt).days
    
    if days_from_file < 1: return log_fail("G2: Past/Zero DTE", ticker)

    # V52: VOLATILITY PIVOT (REAL DATA)
    is_bullish = False
    is_bearish = False
    if momentum == "YES": is_bullish = True 
    
    if flow_score > 0.5: is_bullish, is_bearish = True, False
    elif flow_score < -0.5: is_bullish, is_bearish = False, True
    
    # V53: DARK POOL BIAS (Break Ties for Neutral Stocks)
    # Below DP Wall = Bullish Support (price held above key level)
    # Above DP Wall = Bearish (overextended, potential reversal)
    if not is_bullish and not is_bearish and dp_lvl > 0:
        if stock_price < (dp_lvl * 1.02): is_bullish = True  # Near/Below Wall = Bullish Support
        elif stock_price > (dp_lvl * 1.05): is_bearish = True # Far Above Wall = Bearish (overextended)
    
    strategy_class = "NEUTRAL"
    if is_bullish or is_bearish:
        if iv_rank >= 45: strategy_class = "CREDIT_DIRECTIONAL"  # V54: Lowered for DP-supported trades
        else: strategy_class = "DEBIT_DIRECTIONAL"
    else:
        strategy_class = "CONDOR"



    if strategy_class == "CONDOR":
        if days_from_file < DTE_MIN_SHIELD or days_from_file > DTE_MAX_SHIELD: return log_fail("G2: Shield DTE", ticker)
    else:
        if days_from_file < DTE_MIN_FIRE or days_from_file > DTE_MAX_FIRE: return log_fail("G2: Fire DTE", ticker)

    has_earnings_risk, er_note = False, ""
    if not is_index and ticker in earnings_data:
        er_date = earnings_data[ticker]
        if file_dt <= er_date <= expiry_dt:
            if days_from_file < DTE_BRIDGE: return log_fail("G3: Earnings Risk", ticker)
            has_earnings_risk, er_note = True, f"ER: {er_date.strftime('%m-%d')}"
        if 0 <= (er_date - file_dt).days <= 7: return log_fail("G3: Earnings Imminent", ticker)

    width = get_dynamic_width(stock_price)
    net_premium_est = raw_price * 0.70  # V55: More realistic credit spread estimate 
    action_icon, action_text, strategy_type, strike_setup = "", "", "", ""
    net_val, net_type = 0.0, ""
    max_profit, max_loss, breakeven = 0.0, 0.0, 0.0
    notes = []
    
    # LOGIC BRANCHING
    if strategy_class == "CONDOR":
        track = "SHIELD"
        if momentum == "YES" or abs(flow_score) > 0.6: return log_fail("G11: Trend exists", ticker)
        
        strategy_type, action_icon, action_text = "Iron Condor", "🟨", "Sell Iron Condor"
        short_call, short_put = strike + width, strike - width
        
        if abs(strike - stock_price) / stock_price < 0.04: return log_fail("G6: Too Close", ticker)
        sigma = get_1sigma_move(stock_price, iv_percent if iv_percent > 0 else 40, days_from_file)
        if short_call < (stock_price + sigma) or short_put > (stock_price - sigma): return log_fail("G5: IC Inside 1-Sigma", ticker)
        
        strike_setup = f"Sell {short_call}C/{short_put}P | Buy {short_call+width}C/{short_put-width}P"
        net_val, net_type = net_premium_est * 2, "Credit"
        
        min_credit = width * CREDIT_RATIO_STD
        if CURRENT_VIX < 16.0: min_credit = width * CREDIT_RATIO_LOW_VOL
        if net_val < min_credit: return log_fail("G7: Low Credit", ticker)
        
        max_profit, max_loss = net_val * 100, (width - net_val) * 100
        stop_price, take_profit = net_val * 2.5, net_val * 0.50
        notes.append(f"TP: ${take_profit:.2f}")

    elif strategy_class == "CREDIT_DIRECTIONAL":
        track = "FIRE"
        net_val, net_type = net_premium_est, "Credit"
        min_credit = width * CREDIT_RATIO_DIRECTIONAL  # V54: Lower for directional trades
        if net_val < min_credit: return log_fail("G7: Low Credit", ticker)
        
        if is_bullish:
            if strike >= stock_price: return log_fail("G10: Sell Put ITM", ticker)
            strategy_type, action_icon, action_text = "Bull Put Credit Spread", "🟩", "Sell Put Credit"
            strike_setup = f"Sell {strike}P / Buy {strike-width}P"
            breakeven = strike - net_val
        else:
            if strike <= stock_price: return log_fail("G10: Sell Call ITM", ticker)
            strategy_type, action_icon, action_text = "Bear Call Credit Spread", "🟥", "Sell Call Credit"
            strike_setup = f"Sell {strike}C / Buy {strike+width}C"
            breakeven = strike + net_val
            
        max_profit, max_loss = net_val * 100, (width - net_val) * 100
        stop_price, take_profit = net_val * 3.0, net_val * 0.50 
        notes.append(f"TP: ${take_profit:.2f}")

    elif strategy_class == "DEBIT_DIRECTIONAL":
        track = "FIRE"
        net_val, net_type = net_premium_est, "Debit"
        if net_val >= width * 0.6: return log_fail("G8: Bad R/R", ticker) 
        if net_val < width * 0.1: return log_fail("G8: Lotto", ticker)
        
        if is_bullish:
            if strike <= stock_price: return log_fail("G10: Call ITM", ticker)
            if strike > (stock_price * 1.10): return log_fail("G10: Call Too OTM", ticker)
            strategy_type, action_icon, action_text = "Bull Call Debit Spread", "🟩", "Buy Call Debit"
            strike_setup = f"Buy {strike}C / Sell {strike+width}C"
            breakeven = strike + net_val
        else:
            if strike >= stock_price: return log_fail("G10: Put ITM", ticker)
            if strike < (stock_price * 0.90): return log_fail("G10: Put Too OTM", ticker)
            strategy_type, action_icon, action_text = "Bear Put Debit Spread", "🟥", "Buy Put Debit"
            strike_setup = f"Buy {strike}P / Sell {strike-width}P"
            breakeven = strike - net_val
            
        max_profit, max_loss = (width - net_val) * 100, net_val * 100
        stop_price, take_profit = net_val * 0.50, net_val * 2.0
        notes.append(f"TP: ${take_profit:.2f}")

    conviction = calculate_conviction(row, momentum, iv_rank, is_vip, is_index)
    size_tag = "Core (1.0x)"
    if strategy_class == "DEBIT_DIRECTIONAL": size_tag = "Spec (0.5x)"
    if conviction < 60: size_tag = "Lotto (0.25x)"
    if has_earnings_risk: size_tag = "Spec (0.5x) [ER]"; notes.append(er_note)

    if conviction >= 70: confidence, optimal = "High 🔥", "Yes – Prime"
    elif conviction >= 50: confidence, optimal = "Medium ⚖️", "Yes – Good"
    else: confidence, optimal = "Low 💤", "Watch Only"

    notes.insert(0, f"Strat: {strategy_class}")
    if dp_lvl > 0: notes.append(f"DP Lvl: {dp_lvl}")
    if is_index: notes.append("Index Shield")

    delta_proxy = round(abs(1 - (strike / stock_price)), 2)
    if 'Delta' in row: delta_proxy = row['Delta']
# ====================================================
    # 💰 NEW: KELLY CRITERION SIZING LOGIC
    # ====================================================
    # 1. Calculate Risk/Reward Ratio (Avoid division by zero)
    rr_ratio = 1.0
    if max_loss > 0:
        rr_ratio = max_profit / max_loss

    # 2. Convert Conviction (0-100) to Win Rate (0.0-1.0)
    # If Conviction is 75, we treat it as 75% Win Rate
    estimated_win_rate = conviction / 100.0

    # 3. Calculate Exact Dollar Amount
    # Uses account_size passed into run()
    kelly_usd = calculate_kelly_size(account_size, estimated_win_rate, rr_ratio)
    
    # 4. Update the Size Tag to show the money
    size_tag = f"${kelly_usd} ({size_tag})"
    # ====================================================
    return {
        "#": 0, "Ticker": ticker, "Stock Price": stock_price, "Delta": delta_proxy, 
        "Action": f"{action_icon} {action_text}", "Strategy Type": strategy_type, 
        "Strike Setup": strike_setup, "Expiry": row['Expiry'], "DTE": days_from_file,
        "Net Credit/Debit": f"{net_val:.2f} ({net_type})", "Max Profit": f"${max_profit:.0f}", 
        "Max Loss": f"${max_loss:.0f}", "Breakeven": f"{breakeven:.2f}", 
        "Conviction %": conviction, "Confidence": confidence, "Optimal": optimal, 
        "Size": size_tag, "Notes": "; ".join(notes), "Source": f"UW + MC ({DATE_TAG})"
    }

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
def run(date_arg=None, data_dir_arg=None, account_size=50000):
    global DATA_DIR, DATE_TAG, REJECT_LOG
    REJECT_LOG = {}
    DATA_DIR, DATE_TAG = resolve_data_location(date_arg, data_dir_arg)
    print(f"\nSTRATEGY ENGINE V52 (TRUE MARKET MATH) | VIX: {CURRENT_VIX}")
    if DATA_DIR:
        print(f"   -> Data Dir: {DATA_DIR}")
    if not DATE_TAG:
        print("CRITICAL: No Date Tag Found")
        return

    f_chain  = find_file_smart(["chain", "oi", "oi-changes"], DATE_TAG, DATA_DIR)
    f_whales = find_file_smart(["whale"], DATE_TAG, DATA_DIR)
    f_screen = find_file_smart(["screener", "stock-screener"], DATE_TAG, DATA_DIR)
    f_dp     = find_file_smart(["dp", "dark_pool", "dark-pool"], DATE_TAG, DATA_DIR)
    f_hot    = find_file_smart(["hot"], DATE_TAG, DATA_DIR)

    if not f_chain:
        print(f"CRITICAL: No chain file found for {DATE_TAG} in {DATA_DIR}")
        return

    valid_tickers = {}
    earnings_map = {}
    if f_screen:
        s = smart_normalize(read_csv_auto(f_screen, role="screener"), "screener")
        if 'MarketCap' in s.columns:
            s['MC_Float'] = s['MarketCap'].apply(parse_market_cap)
            for _, row in s.iterrows():
                sym = row['Symbol']
                if 'Date' in row and pd.notnull(row['Date']):
                    try:
                        earnings_map[sym] = pd.to_datetime(row['Date'])
                    except:
                        pass

                mc = row.get('MC_Float', 0)
                if sym not in INDEX_ETFS and sym not in VIP_LIST and mc < RULE_MIN_CAP:
                    continue
                iv = float(str(row.get('IV', 0)).replace('%','')) if pd.notnull(row.get('IV')) else 0
                if iv < RULE_MIN_IV:
                    continue
                valid_tickers[sym] = {"IV": iv, "Cap": mc}

    df = read_csv_auto(f_chain, role="chain")
    df = smart_normalize(df, "chain")
    df['Expiry'] = df.apply(lambda x: calculate_expiry(x, DATE_TAG), axis=1)

    if valid_tickers and 'Symbol' in df.columns:
        allowed = set(valid_tickers.keys()).union(INDEX_ETFS).union(VIP_LIST)
        df = df[df['Symbol'].isin(allowed)].copy()

    # FIX V55: Volume OR OI filter (high OI = institutional interest even if low daily volume)
    if 'OI' in df.columns and 'Volume' in df.columns:
        df = df[(df['Volume'] >= 500) | (df['OI'] >= 5000)].copy()  # OI >= 5000 for quality
    elif 'OI' in df.columns:
        df = df[df['OI'] >= 5000].copy()
    elif 'Volume' in df.columns:
        df = df[df['Volume'] >= RULE_MIN_VOLUME].copy()
    if 'StockPrice' in df.columns:
        df = df[df['StockPrice'] > 0].copy()

    df_whales = None
    if f_whales:
        df_whales = smart_normalize(read_csv_auto(f_whales, role="whale"), "whales")

    dp_levels = {}
    if f_dp:
        d = smart_normalize(read_csv_auto(f_dp, role="dark_pool"), "dark_pool")
        if 'StockPrice' in d.columns:
            dp_levels = d.groupby('Symbol')['StockPrice'].mean().to_dict()

    hot_list = []
    if f_hot:
        h = smart_normalize(read_csv_auto(f_hot, role="hot"), "hot")
        if 'Symbol' in h.columns:
            hot_list = h['Symbol'].unique().tolist()

    print("   -> Architecting Trades...")
    trades = []

    for sym, group in df.groupby('Symbol'):
        is_vip = sym in VIP_LIST
        avg_price = group['StockPrice'].mean() if 'StockPrice' in group else 0
        if avg_price == 0:
            avg_price = fetch_live_price(sym)
            if avg_price == 0:
                continue
            group['StockPrice'] = avg_price

        if not is_vip and avg_price < RULE_MIN_PRICE:
            continue

        # SCAN DEEPER: Process all rows to avoid missing valid DTEs buried by high-volume weeklies
        if 'Volume' in group.columns:
            top_candidates = group.sort_values(by='Volume', ascending=False)
        elif 'OI' in group.columns:
            top_candidates = group.sort_values(by='OI', ascending=False)
        else:
            top_candidates = group
        flow_score = analyze_flow_unanimity(df_whales, sym)
        dp_lvl = dp_levels.get(sym, 0)
        mom = "YES" if sym in hot_list or is_vip else "NO"

        # V52 FIX: GET ACTUAL IV RANK - Use Median of group to be robust
        iv_rank = 50.0  # Default
        if 'IVRank' in group.columns:
            iv_series = pd.to_numeric(
                group['IVRank'].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            ).dropna()
            if not iv_series.empty:
                iv_rank = float(iv_series.median())

        # Collect ALL valid trades for this symbol
        symbol_trades = []
        for _, best in top_candidates.iterrows():
            if best.get('StockPrice', 0) == 0:
                best['StockPrice'] = avg_price
            trade = build_trade(best, flow_score, dp_lvl, mom, iv_rank, earnings_map, account_size)
            if trade:
                symbol_trades.append(trade)

        # COLLECT ALL VALID TRADES (No Limits)
        # If multiple setups work for one symbol (e.g. Iron Condor AND Bull Put), show them both.
        if symbol_trades:
            trades.extend(symbol_trades)

    cols = ['#', 'Ticker', 'Stock Price', 'Delta', 'Action', 'Strategy Type', 'Strike Setup',
            'Expiry', 'DTE', 'Net Credit/Debit', 'Max Profit', 'Max Loss', 'Breakeven',
            'Conviction %', 'Confidence', 'Optimal', 'Size', 'Notes', 'Source']

    if trades:
        final_df = pd.DataFrame(trades).sort_values(by='Conviction %', ascending=False)
        final_df['#'] = range(1, len(final_df) + 1)
        final_df = final_df[cols]
    else:
        final_df = pd.DataFrame(columns=cols)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# EXPERT TRADE TABLE ({DATE_TAG})\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        try:
            f.write(final_df.to_markdown(index=False))
        except Exception:
            f.write(final_df.to_string(index=False))

        f.write("\n\n## GATE CARD AUDIT\n")
        f.write("| Gate | Rejections |\n|---|---|\n")
        for k, v in REJECT_LOG.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n\nWARNING: Check 10-Q/8-K for top 3 candidates. Verify prices on broker.")

    # === V53 UPGRADE: INSTITUTIONAL MEMORY ===
    # 1. Save Daily Snapshot
    if DATA_DIR and not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    daily_filename = os.path.join(DATA_DIR if DATA_DIR else ".", f"daily_log_{DATE_TAG}.csv")
    final_df.to_csv(daily_filename, index=False)

    # 2. Append to Master Database (The "Longitudinal Brain")
    master_file = "INSTITUTIONAL_MASTER_DB.csv"

    # Add a 'Date Signal' column so we know when this trade was generated
    final_df['Signal Date'] = DATE_TAG

    if not os.path.exists(master_file):
        final_df.to_csv(master_file, index=False)
    else:
        # Append only new data (prevent duplicates if you run twice in one day)
        existing = pd.read_csv(master_file)
        # Filter out rows that already exist for this date
        existing = existing[existing['Signal Date'] != DATE_TAG]
        updated_master = pd.concat([existing, final_df], ignore_index=True)
        updated_master.to_csv(master_file, index=False)

    print(f"   MEMORY UPDATED: {len(final_df)} trades added to {master_file}")
    print(f"\nSUCCESS. {len(trades)} Trades Generated.")
    print(f"   -> Saved to: {OUTPUT_FILE}")
    print("\nGATE FAILURES:")
    for k, v in REJECT_LOG.items():
        print(f"   - {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy engine for a specific date or data folder.")
    parser.add_argument("--date", help="YYYY-MM-DD (e.g., 2026-01-26)")
    parser.add_argument("--data-dir", help="Path to daily folder or temp workspace")
    parser.add_argument("--account-size", type=float, default=50000, help="Account size for Kelly sizing")
    args = parser.parse_args()
    run(date_arg=args.date, data_dir_arg=args.data_dir, account_size=args.account_size)
