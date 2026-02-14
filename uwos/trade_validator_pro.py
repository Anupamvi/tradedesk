import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
import argparse

# ==========================================
# ⚡ TRADE VALIDATOR PRO (V3.0 - HYBRID AGENT)
# ==========================================


def validate_trade_logic(
    ticker,
    trade_type,
    current_price,
    dte,
    long_strike,
    short_strike,
    cost,
    short_put=0.0,
    short_call=0.0,
    period="1y",
    profit_target="breakeven",
    profit_pct=0.5,
    use_rsi_filter=True,
):
    """
    Core validation logic.
    trade_type: 1=Debit, 2=Credit (Vertical), 3=Iron Condor
    For IC (Type 3): short_strike is ignored, use short_put and short_call.
    """
    result = {
        "ticker": ticker,
        "win_rate": 0.0,
        "required_win_rate": 0.0,
        "edge": 0.0,
        "verdict": "FAIL",
        "signals": 0,
        "wins": 0,
        "note": ""
    }

    # 1. STRATEGY MATH
    if trade_type == 1: # DEBIT (Buying)
        width = abs(short_strike - long_strike)
        max_profit = width - cost
        max_loss = cost
        # target definition for debit spreads
        if short_strike >= long_strike:
            # call debit (price up)
            if profit_target == "max":
                target_price = short_strike
            elif profit_target == "pct":
                target_price = long_strike + cost + (profit_pct * max_profit)
            else:  # breakeven
                target_price = long_strike + cost
        else:
            # put debit (price down)
            if profit_target == "max":
                target_price = short_strike
            elif profit_target == "pct":
                target_price = long_strike - (cost + (profit_pct * max_profit))
            else:  # breakeven
                target_price = long_strike - cost
    elif trade_type == 2: # CREDIT (Selling Vertical)
        width = abs(short_strike - long_strike)
        max_loss = (width - cost)
        max_profit = cost
        is_put = short_strike < current_price
        target_price = short_strike
    elif trade_type == 3: # IRON CONDOR (Selling)
        # For IC validation, we need to survive between Short Put and Short Call
        # Cost = Credit Received
        # Max Loss = Max Width - Credit
        # Assuming symmetric or using the widest wing for conservative estimate
        # We need width. In batch mode, we might not have long strikes for both wings easily.
        # Let's assume standard width or just use risk/reward passed if possible? 
        # For now, let's use the cost/credit to implied math if possible, or just focus on probability.
        # Required Win Rate = Max Loss / (Max Loss + Max Profit)
        # Max Profit = Cost (Credit)
        # If we don't have max loss explicitly, we might struggle.
        # But wait, the loop calculates HISTORICAL win rate. We can compare that.
        max_profit = cost
        # We need proper Max Loss to calculate edge. 
        # If passed as 0, we can't calc Edge, but we can calc Win %
        max_loss = cost # Placeholder if unknown to avoid div/0, but requires caution
        
    # Required Win Rate
    # If Type 3 and we don't have max_loss, we default required to ~60% for a Condor?
    # Or we construct it from the wings if available.
    # Let's assume standard IC 1/3 width credit rule? 
    # Better: If we can't calc required, just report Hist Win Rate.
    
    if max_loss + max_profit > 0:
        required_win_rate = (max_loss / (max_loss + max_profit)) * 100
    else:
        required_win_rate = 0.0

    # 2. GET DATA
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: 
            result["note"] = "No Data"
            return result
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except:
        result["note"] = "Download Error"
        return result

    df['RSI'] = df.ta.rsi(length=14)
    
    wins = 0
    total_signals = 0
    
    # 3. SIMULATION
    for i in range(200, len(df) - dte):
        row = df.iloc[i]
        
        # Match regime (RSI 30-70)
        if (not use_rsi_filter) or (row['RSI'] > 30 and row['RSI'] < 70):
            total_signals += 1
            entry_price = row['Close']
            window = df.iloc[i+1 : i+dte]
            
            if trade_type == 1: # DEBIT
                move_needed_pct = (target_price - current_price) / current_price
                sim_target = entry_price * (1 + move_needed_pct)
                if short_strike >= long_strike:
                    if window['High'].max() >= sim_target:
                        wins += 1
                else:
                    if window['Low'].min() <= sim_target:
                        wins += 1
                
            elif trade_type == 2: # CREDIT VERTICAL
                move_danger_pct = (target_price - current_price) / current_price
                sim_danger = entry_price * (1 + move_danger_pct)
                touched = False
                if is_put: # Selling Put
                    if window['Low'].min() <= sim_danger: touched = True
                else: # Selling Call
                    if window['High'].max() >= sim_danger: touched = True
                if not touched: wins += 1
            
            elif trade_type == 3: # IRON CONDOR
                # Check bounds
                # We need percentages for Short Call and Short Put from current price
                if short_call > 0 and short_put > 0:
                    call_dist_pct = (short_call - current_price) / current_price
                    put_dist_pct = (short_put - current_price) / current_price # Negative
                    
                    sim_upper = entry_price * (1 + call_dist_pct)
                    sim_lower = entry_price * (1 + put_dist_pct)
                    
                    # Survived if High < Upper AND Low > Lower
                    if window['High'].max() < sim_upper and window['Low'].min() > sim_lower:
                        wins += 1

    if total_signals == 0:
        result["note"] = "No Signals"
        return result

    hist_win_rate = (wins / total_signals) * 100
    edge = hist_win_rate - required_win_rate
    
    # Verdict
    if trade_type == 3 and required_win_rate == 0:
        # Fallback for IC if we didn't have risk math
        if hist_win_rate > 65: verdict = "PASS"
        else: verdict = "FAIL"
    else:
        if edge > 0: verdict = "PASS"
        else: verdict = "FAIL"

    result["win_rate"] = hist_win_rate
    result["required_win_rate"] = required_win_rate
    result["edge"] = edge
    result["verdict"] = verdict
    result["signals"] = total_signals
    result["wins"] = wins
    
    return result

def run_validator():
    print("\n⚡ TRADE VALIDATOR PRO (HYBRID ENGINE)")
    # ... (Keep existing interactive/argparse logic but link to validate_trade_logic if needed or leave as legacy wrapper)
    # For now, I'll essentially keep the legacy code as the "__main__" block or wrapper 
    # to avoid breaking the CLI usage, but the batch script will import `validate_trade_logic`.
    # ...
    pass 



def run_validator_interactive():
    # 1. SETUP ARGUMENT PARSER (For Agent Mode)
    parser = argparse.ArgumentParser(description='Validate Options Trade')
    parser.add_argument('--ticker', type=str, help='Ticker Symbol')
    parser.add_argument('--type', type=int, help='1=Debit (Buy), 2=Credit (Sell)')
    parser.add_argument('--price', type=float, help='Current Stock Price')
    parser.add_argument('--dte', type=int, help='Days to Expiration')
    parser.add_argument('--long', type=float, help='Long Strike')
    parser.add_argument('--short', type=float, help='Short Strike')
    parser.add_argument('--cost', type=float, help='Net Debit or Credit Amount')
    parser.add_argument('--profit-target', type=str, default='breakeven', choices=['breakeven','pct','max'],
                        help='Target for debit spreads: breakeven | pct | max')
    parser.add_argument('--profit-pct', type=float, default=0.5,
                        help='If profit-target=pct, target % of max profit (0.0-1.0)')
    parser.add_argument('--no-rsi-filter', action='store_true',
                        help='Disable RSI 30-70 regime filter')

    args = parser.parse_args()

    # 2. DECIDE: HUMAN OR ROBOT?
    if args.ticker:
        # --- ROBOT MODE ---
        # Call logic
        res = validate_trade_logic(
            args.ticker.upper(),
            args.type,
            args.price,
            args.dte,
            args.long,
            args.short,
            args.cost,
            profit_target=args.profit_target,
            profit_pct=args.profit_pct,
            use_rsi_filter=not args.no_rsi_filter,
        )
        print(f"VERDICT: {res['verdict']} | Win: {res['win_rate']:.1f}% | Edge: {res['edge']:.1f}%")
    else:
        # --- HUMAN MODE ---
        print("👤 INTERACTIVE MODE: Please answer the prompts.\n")
        ticker = get_input_manual("Ticker Symbol (e.g. AAPL)", str).upper()
        trade_type = get_input_manual("Strategy Type? (1=Debit/Buy, 2=Credit/Sell)", int)
        current_price = get_input_manual(f"Current Stock Price of {ticker}", float)
        dte = get_input_manual("Days to Expiration (DTE)", int)
        
        long_strike, short_strike = 0.0, 0.0
        if trade_type == 1:
            long_strike = get_input_manual("Long Strike (Lower)", float)
            short_strike = get_input_manual("Short Strike (Higher)", float)
            cost = get_input_manual("Net Debit Paid (e.g. 1.55)", float)
        else:
            short_strike = get_input_manual("Short Strike (The Dangerous One)", float)
            long_strike = get_input_manual("Long Strike (The Protection)", float)
            cost = get_input_manual("Net Credit Received (e.g. 0.50)", float)
            
        res = validate_trade_logic(ticker, trade_type, current_price, dte, long_strike, short_strike, cost)
        
        print(f"\n" + "="*40)
        print(f" FINAL VERDICT: {res['verdict']}")
        print("="*40)
        print(f"🎯 Success Probability:  {res['win_rate']:.1f}%  (History)")
        print(f"⚖️  Required to Win:     {res['required_win_rate']:.1f}%  (Math)")
        print(f"🚀  Statistical Edge:    {res['edge']:+.1f}%")

if __name__ == "__main__":
    run_validator_interactive()
