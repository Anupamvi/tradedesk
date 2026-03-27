#!/usr/bin/env python3
"""
Schwab Account Puller for Anu Options Engine
=============================================
Run this on YOUR local machine. It reads your Schwab API tokens,
pulls positions + trade history + account data, and writes a JSON
file you can upload to the scan engine.

Usage:
    python schwab_pull.py
    python schwab_pull.py --token-path "C:/uw_root/tokens"
    python schwab_pull.py --output positions.json

Output: schwab_positions_YYYY-MM-DD.json (upload this to Claude)

Requirements:
    pip install requests

Your credentials NEVER leave your machine. This script only writes
a local JSON file that YOU choose to upload.
"""

import json, os, sys, glob, time, argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ================================================================
# CONFIGURATION
# ================================================================

DEFAULT_TOKEN_PATH = r"C:\uw_root\tokens"
SCHWAB_API_BASE = "https://api.schwabapi.com"
SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"

# ================================================================
# TOKEN DETECTION — auto-detect format
# ================================================================

def find_tokens(token_path):
    """Auto-detect token file format at the given path."""
    path = Path(token_path)
    
    tokens = {
        'access_token': None,
        'refresh_token': None,
        'client_id': None,
        'client_secret': None,
        'token_expiry': None,
    }
    
    # Case 1: Path is a file
    if path.is_file():
        return _parse_token_file(path, tokens)
    
    # Case 2: Path is a directory — scan for token files
    if path.is_dir():
        # Try common filenames
        candidates = [
            'token.json', 'tokens.json', 'schwab_token.json',
            'credentials.json', 'auth.json', 'api_token.json',
            '.token', 'access_token.json', 'schwab.json',
            'token.dat', 'token.txt',
        ]
        
        # Also try any .json file in the directory
        json_files = list(path.glob('*.json'))
        dat_files = list(path.glob('*.dat'))
        txt_files = list(path.glob('*.txt'))
        env_files = list(path.glob('.env*'))
        
        all_files = []
        for name in candidates:
            f = path / name
            if f.exists():
                all_files.insert(0, f)  # prioritize known names
        all_files.extend(json_files)
        all_files.extend(dat_files)
        all_files.extend(txt_files)
        all_files.extend(env_files)
        
        # Deduplicate
        seen = set()
        unique = []
        for f in all_files:
            if str(f) not in seen:
                seen.add(str(f))
                unique.append(f)
        
        print(f"\nScanning {path} for token files...")
        print(f"Found {len(unique)} candidate files:")
        for f in unique[:10]:
            print(f"  {f.name} ({f.stat().st_size} bytes)")
        
        for f in unique:
            result = _parse_token_file(f, tokens.copy())
            if result and (result.get('access_token') or result.get('refresh_token')):
                print(f"\n✅ Tokens found in: {f.name}")
                return result
        
        print("\n❌ No valid tokens found in directory.")
        print("   Expected a JSON file with 'access_token' or 'refresh_token'")
        return None
    
    print(f"\n❌ Path not found: {token_path}")
    return None


def _parse_token_file(filepath, tokens):
    """Try to parse a single file for Schwab tokens."""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
    except Exception as e:
        return None
    
    # Try JSON
    try:
        data = json.loads(content)
        return _extract_from_dict(data, tokens)
    except json.JSONDecodeError:
        pass
    
    # Try .env format (KEY=VALUE)
    if '=' in content:
        data = {}
        for line in content.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, _, val = line.partition('=')
                data[key.strip().lower()] = val.strip().strip('"').strip("'")
        if data:
            return _extract_from_dict(data, tokens)
    
    # Try: file is just a raw token string
    if len(content) > 20 and '\n' not in content:
        tokens['access_token'] = content
        return tokens
    
    return None


def _extract_from_dict(data, tokens):
    """Extract token fields from a dictionary (handles nested formats)."""
    
    # Flatten nested structures (schwab-py format has 'token' wrapper)
    if 'token' in data and isinstance(data['token'], dict):
        inner = data['token']
        data = {**data, **inner}
    
    # Map common field names
    field_map = {
        'access_token': ['access_token', 'accessToken', 'access'],
        'refresh_token': ['refresh_token', 'refreshToken', 'refresh'],
        'client_id': ['client_id', 'clientId', 'app_key', 'appKey', 'api_key', 'apiKey', 'consumer_key'],
        'client_secret': ['client_secret', 'clientSecret', 'app_secret', 'appSecret', 'api_secret'],
        'token_expiry': ['creation_timestamp', 'expires_at', 'expiry', 'token_expiry'],
    }
    
    for target, sources in field_map.items():
        for src in sources:
            # Try exact match
            if src in data:
                tokens[target] = data[src]
                break
            # Try case-insensitive
            for k, v in data.items():
                if k.lower() == src.lower():
                    tokens[target] = v
                    break
    
    return tokens


def refresh_access_token(tokens):
    """Use refresh_token to get a new access_token."""
    if not tokens.get('refresh_token'):
        print("❌ No refresh_token available. Cannot refresh.")
        return None
    if not tokens.get('client_id') or not tokens.get('client_secret'):
        print("❌ Need client_id and client_secret to refresh token.")
        print("   These should be in your token file or set as env vars:")
        print("   SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET")
        
        # Try environment variables
        tokens['client_id'] = tokens.get('client_id') or os.environ.get('SCHWAB_CLIENT_ID')
        tokens['client_secret'] = tokens.get('client_secret') or os.environ.get('SCHWAB_CLIENT_SECRET')
        
        if not tokens.get('client_id') or not tokens.get('client_secret'):
            return None
    
    print("Refreshing access token...")
    try:
        resp = requests.post(SCHWAB_AUTH_URL, data={
            'grant_type': 'refresh_token',
            'refresh_token': tokens['refresh_token'],
        }, auth=(tokens['client_id'], tokens['client_secret']))
        
        if resp.status_code == 200:
            new_tokens = resp.json()
            tokens['access_token'] = new_tokens['access_token']
            if 'refresh_token' in new_tokens:
                tokens['refresh_token'] = new_tokens['refresh_token']
            print("✅ Token refreshed successfully")
            return tokens
        else:
            print(f"❌ Token refresh failed: {resp.status_code}")
            print(f"   {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ Token refresh error: {e}")
        return None


# ================================================================
# SCHWAB API CALLS
# ================================================================

def schwab_get(endpoint, access_token, params=None):
    """Make authenticated GET request to Schwab API."""
    headers = {'Authorization': f'Bearer {access_token}'}
    url = f"{SCHWAB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 401:
            return {'error': 'UNAUTHORIZED', 'status': 401}
        if resp.status_code != 200:
            return {'error': resp.text[:200], 'status': resp.status_code}
        return resp.json()
    except Exception as e:
        return {'error': str(e)}


def get_accounts(access_token):
    """Get all linked accounts."""
    return schwab_get('/trader/v1/accounts', access_token)


def get_positions(access_token, account_hash):
    """Get positions for an account."""
    return schwab_get(f'/trader/v1/accounts/{account_hash}',
                      access_token, params={'fields': 'positions'})


def get_transactions(access_token, account_hash, days=90):
    """Get recent transactions (for Health Gate)."""
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00.000Z')
    end = datetime.now().strftime('%Y-%m-%dT23:59:59.000Z')
    return schwab_get(f'/trader/v1/accounts/{account_hash}/transactions',
                      access_token, params={
                          'startDate': start,
                          'endDate': end,
                          'types': 'TRADE',
                      })


def get_orders(access_token, account_hash, days=30):
    """Get recent orders (pending fills)."""
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00.000Z')
    end = datetime.now().strftime('%Y-%m-%dT23:59:59.000Z')
    return schwab_get(f'/trader/v1/accounts/{account_hash}/orders',
                      access_token, params={
                          'fromEnteredTime': start,
                          'toEnteredTime': end,
                      })


# ================================================================
# FORMAT OUTPUT FOR SCAN ENGINE
# ================================================================

def _parse_option_symbol(symbol):
    """Parse Schwab option symbol to extract strike, expiry, type.
    
    Format: 'META  260821P00510000' or 'GRAB  270115C00012000'
    = TICKER (padded to 6) + YYMMDD + C/P + strike×1000 (8 digits)
    """
    import re
    symbol = symbol.strip()
    m = re.match(r'^([A-Z]+)\s*(\d{6})([CP])(\d{8})$', symbol)
    if not m:
        return 0, '', ''
    
    date_str = m.group(2)  # YYMMDD
    opt_type = 'CALL' if m.group(3) == 'C' else 'PUT'
    strike = int(m.group(4)) / 1000  # 00510000 → 510.0
    
    try:
        from datetime import datetime
        expiry = datetime.strptime('20' + date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except:
        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    
    return strike, expiry, opt_type


def format_for_engine(account_data, transactions, orders):
    """Transform Schwab API data into scan engine format."""
    output = {
        'pull_timestamp': datetime.now().isoformat(),
        'source': 'schwab_api',
        'account': {},
        'positions': [],
        'open_orders': [],
        'closed_trades': [],  # For Health Gate
        'health_gate': {},
    }
    
    # Account info — handle both dict and list responses
    if isinstance(account_data, list):
        account_data = account_data[0] if account_data else {}
    acct = account_data.get('securitiesAccount', account_data.get('account', account_data))
    balances = acct.get('currentBalances', {})
    output['account'] = {
        'account_type': acct.get('type', ''),
        'nlv': balances.get('liquidationValue', 0),
        'buying_power': balances.get('buyingPower', 0),
        'cash': balances.get('cashBalance', 0),
        'margin_equity': balances.get('equity', 0),
    }
    
    # Positions
    for pos in acct.get('positions', []):
        inst = pos.get('instrument', {})
        
        position = {
            'ticker': inst.get('underlyingSymbol', inst.get('symbol', '')),
            'symbol': inst.get('symbol', ''),
            'asset_type': inst.get('assetType', ''),
            'quantity': pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
            'avg_cost': pos.get('averagePrice', 0),
            'market_value': pos.get('marketValue', 0),
            'day_pnl': pos.get('currentDayProfitLoss', 0),
            'total_pnl': pos.get('longOpenProfitLoss', pos.get('shortOpenProfitLoss', 0)),
            'total_pnl_pct': pos.get('longOpenProfitLossPercentage',
                                     pos.get('shortOpenProfitLossPercentage', 0)),
        }
        
        # Options-specific fields
        if inst.get('assetType') == 'OPTION':
            # Parse strike and expiry from symbol (e.g., "META  260821P00510000")
            sym = inst.get('symbol', '')
            parsed_strike, parsed_expiry, parsed_type = _parse_option_symbol(sym)
            
            position.update({
                'option_type': inst.get('putCall', parsed_type or ''),
                'strike': inst.get('strikePrice', 0) or parsed_strike,
                'expiry': inst.get('expirationDate', '') or parsed_expiry,
                'underlying': inst.get('underlyingSymbol', ''),
                'multiplier': pos.get('multiplier', 100),
            })
            
            # Classify position type for engine
            qty = position['quantity']
            opt = position['option_type']
            if qty > 0 and opt == 'CALL': position['engine_type'] = 'long_call'
            elif qty > 0 and opt == 'PUT': position['engine_type'] = 'long_put'
            elif qty < 0 and opt == 'CALL': position['engine_type'] = 'short_call'
            elif qty < 0 and opt == 'PUT': position['engine_type'] = 'short_put'
        else:
            position['engine_type'] = 'shares' if position['quantity'] > 0 else 'short_shares'
        
        output['positions'].append(position)
    
    # Open orders
    if isinstance(orders, list):
        for order in orders:
            if order.get('status') in ('WORKING', 'PENDING_ACTIVATION', 'QUEUED', 'ACCEPTED'):
                legs = order.get('orderLegCollection', [])
                output['open_orders'].append({
                    'order_id': order.get('orderId'),
                    'status': order.get('status'),
                    'order_type': order.get('orderType'),
                    'price': order.get('price', order.get('stopPrice', 0)),
                    'entered': order.get('enteredTime'),
                    'legs': [{
                        'symbol': leg.get('instrument', {}).get('symbol', ''),
                        'action': leg.get('instruction', ''),
                        'quantity': leg.get('quantity', 0),
                    } for leg in legs],
                })
    
    # Closed trades (for Health Gate)
    fire_results = []
    if isinstance(transactions, list):
        for txn in transactions:
            if txn.get('type') == 'TRADE':
                items = txn.get('transactionItems', []) or [txn]
                for item in items:
                    inst = item.get('instrument', {})
                    if inst.get('assetType') == 'OPTION':
                        output['closed_trades'].append({
                            'date': txn.get('transactionDate', ''),
                            'symbol': inst.get('symbol', ''),
                            'underlying': inst.get('underlyingSymbol', ''),
                            'action': item.get('instruction', ''),
                            'quantity': item.get('amount', 0),
                            'price': item.get('price', 0),
                            'cost': item.get('cost', 0),
                            'net_amount': txn.get('netAmount', 0),
                        })
    
    # Health Gate computation
    # Group trades by underlying + expiry to identify spread P&Ls
    output['health_gate'] = {
        'total_option_trades': len(output['closed_trades']),
        'note': 'Group by underlying+expiry to compute FIRE spread P&Ls for Health Gate',
    }
    
    return output


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Pull Schwab account data for Anu Options Engine')
    parser.add_argument('--token-path', default=DEFAULT_TOKEN_PATH,
                        help=f'Path to Schwab token file/directory (default: {DEFAULT_TOKEN_PATH})')
    parser.add_argument('--output', default=None,
                        help='Output JSON filename (default: schwab_positions_YYYY-MM-DD.json)')
    parser.add_argument('--days', type=int, default=90,
                        help='Days of trade history to pull (default: 90)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Schwab Account Puller — Anu Options Engine")
    print("=" * 60)
    
    # 1. Find tokens
    print(f"\nToken path: {args.token_path}")
    tokens = find_tokens(args.token_path)
    
    if not tokens:
        print("\n❌ Could not find tokens. Make sure the path is correct.")
        print("   Expected: JSON file with access_token/refresh_token")
        sys.exit(1)
    
    # 1b. Load .env for client credentials if missing
    if not tokens.get('client_id') or not tokens.get('client_secret'):
        env_locations = [
            Path(args.token_path).parent / '.env',   # C:\uw_root\.env
            Path(args.token_path) / '.env',           # C:\uw_root\tokens\.env
            Path('.env'),                              # current directory
        ]
        for env_path in env_locations:
            if env_path.exists():
                print(f"Loading credentials from {env_path}...")
                try:
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if '=' in line and not line.startswith('#'):
                                key, _, val = line.partition('=')
                                key = key.strip()
                                val = val.strip().strip('"').strip("'")
                                kl = key.lower()
                                if kl in ('schwab_client_id', 'client_id', 'app_key', 'api_key',
                                          'schwab_api_key', 'schwab_app_key'):
                                    tokens['client_id'] = val
                                elif kl in ('schwab_client_secret', 'client_secret', 'app_secret', 'api_secret',
                                            'schwab_app_secret', 'schwab_secret'):
                                    tokens['client_secret'] = val
                    if tokens.get('client_id') and tokens.get('client_secret'):
                        print(f"  ✅ client_id + client_secret loaded from .env")
                        break
                except Exception as e:
                    print(f"  ⚠️ Could not read {env_path}: {e}")

    
    # Show what we found (redacted)
    print(f"\nToken status:")
    print(f"  access_token:  {'✅ found (' + tokens['access_token'][:8] + '...)' if tokens.get('access_token') else '❌ missing'}")
    print(f"  refresh_token: {'✅ found (' + tokens['refresh_token'][:8] + '...)' if tokens.get('refresh_token') else '❌ missing'}")
    print(f"  client_id:     {'✅ found' if tokens.get('client_id') else '❌ missing'}")
    print(f"  client_secret: {'✅ found' if tokens.get('client_secret') else '❌ missing'}")
    
    # 2. Test access token, refresh if needed
    access_token = tokens.get('access_token')
    
    if access_token:
        print("\nTesting access token...")
        test = get_accounts(access_token)
        if test.get('error') == 'UNAUTHORIZED':
            print("Access token expired. Attempting refresh...")
            tokens = refresh_access_token(tokens)
            if tokens:
                access_token = tokens['access_token']
            else:
                print("❌ Could not refresh token. You may need to re-authenticate.")
                sys.exit(1)
        elif 'error' in test:
            print(f"❌ API error: {test['error']}")
            sys.exit(1)
    else:
        # No access token — try refresh
        tokens = refresh_access_token(tokens)
        if tokens:
            access_token = tokens['access_token']
        else:
            print("❌ No access token and cannot refresh. Re-authenticate required.")
            sys.exit(1)
    
    # 3. Get accounts
    print("\nFetching accounts...")
    accounts = get_accounts(access_token)
    
    if not isinstance(accounts, list) or len(accounts) == 0:
        print(f"❌ No accounts found. Response: {json.dumps(accounts)[:200]}")
        sys.exit(1)
    
    print(f"Found {len(accounts)} account(s)")
    
    all_data = []
    for acct in accounts:
        # Handle nested account structures
        if isinstance(acct, dict):
            account_hash = acct.get('hashValue', acct.get('accountHash', ''))
            acct_num = str(acct.get('accountNumber', 
                          acct.get('securitiesAccount', {}).get('accountNumber', 'unknown')))
        else:
            continue
        print(f"\n--- Account: ...{acct_num[-4:]} (hash: {account_hash[:8]}...) ---")
        
        # 4. Get positions
        print("  Fetching positions...")
        pos_data = get_positions(access_token, account_hash)
        if 'error' in pos_data if isinstance(pos_data, dict) else False:
            print(f"  ⚠️ Positions error: {pos_data['error'][:100]}")
        else:
            # Debug: show structure
            if isinstance(pos_data, list):
                print(f"  Response: list with {len(pos_data)} items")
            elif isinstance(pos_data, dict):
                print(f"  Response keys: {list(pos_data.keys())[:5]}")
        
        # 5. Get transactions
        print(f"  Fetching {args.days}-day trade history...")
        txn_data = get_transactions(access_token, account_hash, args.days)
        
        # 6. Get open orders
        print("  Fetching open orders...")
        order_data = get_orders(access_token, account_hash)
        
        # 7. Format for engine
        formatted = format_for_engine(pos_data, txn_data, order_data)
        formatted['account']['account_number_last4'] = acct_num[-4:]
        all_data.append(formatted)
    
    # 8. Write output
    output_file = args.output or f"schwab_positions_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    final_output = {
        'generated': datetime.now().isoformat(),
        'generator': 'schwab_pull.py v1.0 — Anu Options Engine',
        'accounts': all_data,
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Output written to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    for i, acct_data in enumerate(all_data):
        print(f"\nAccount ...{acct_data['account'].get('account_number_last4', '????')}:")
        print(f"  NLV:          ${acct_data['account'].get('nlv', 0):,.2f}")
        print(f"  Buying power: ${acct_data['account'].get('buying_power', 0):,.2f}")
        print(f"  Positions:    {len(acct_data['positions'])}")
        print(f"  Open orders:  {len(acct_data['open_orders'])}")
        print(f"  Trade history: {len(acct_data['closed_trades'])} option trades")
        
        # Position summary
        for pos in acct_data['positions']:
            qty = pos['quantity']
            sym = pos.get('symbol', pos.get('ticker', ''))
            pnl = pos.get('total_pnl', 0)
            if pos['asset_type'] == 'OPTION':
                print(f"    {'LONG' if qty > 0 else 'SHORT'} {abs(qty)}× "
                      f"{pos.get('underlying', sym)} {pos.get('expiry','')[:10]} "
                      f"${pos.get('strike',0):.0f} {pos.get('option_type','')} "
                      f"P&L: ${pnl:+,.0f}")
            else:
                print(f"    {abs(qty):.1f} shares {sym} "
                      f"avg ${pos.get('avg_cost',0):.2f} P&L: ${pnl:+,.0f}")
    
    print(f"\n📤 Upload '{output_file}' to Claude for scan engine ingestion.")
    print(f"   The engine will use this for: positions, portfolio delta,")
    print(f"   Health Gate, concentration checks, and NLV-based sizing.")


if __name__ == '__main__':
    main()
