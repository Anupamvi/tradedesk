# xhigh gates 2026-08-31

```json
{
  "quote": {
    "min_last": 15,
    "min_bid": 0,
    "max_spread_frac": 0.01,
    "max_spread_abs": 0.1
  },
  "orats": {
    "min_mkt_cap": 2000,
    "min_avg_opt_vol_20d": 200,
    "cheap_iv_hv": 0.9,
    "cheap_iv_pctile": 40
  },
  "dte_min": 25,
  "dte_max": 45,
  "earnings_buffer_days": 3,
  "max_width_frac": 0.07,
  "csp": {
    "otm_min": 0.08,
    "otm_max": 0.15,
    "ideal_otm": 0.1,
    "min_credit_frac": 0.004,
    "max_spread_frac_of_credit": 0.35
  },
  "put_credit": {
    "short_otm_min": 0.08,
    "short_otm_max": 0.15,
    "min_credit": 0.2
  },
  "call_debit": {
    "long_otm_min": -0.02,
    "long_otm_max": 0.04,
    "short_otm_min": 0.05,
    "short_otm_max": 0.08,
    "min_debit": 0.4,
    "min_rr": 1.2,
    "max_rr": 4.0
  },
  "call_credit": {
    "short_otm_min": 0.05,
    "short_otm_max": 0.08,
    "min_credit": 0.2
  },
  "put_debit": {
    "long_otm_min": -0.04,
    "long_otm_max": 0.02,
    "min_debit": 0.4,
    "min_rr": 1.2,
    "max_rr": 4.0
  },
  "swing": {
    "chase_atr": 2.5,
    "quote_width_abs": 0.25,
    "quote_width_frac": 0.1,
    "atr_n": 14,
    "stop_atr": 1.0,
    "max_stop_frac": 0.08
  },
  "score": {
    "conf_floor": 40,
    "conf_max": 85
  }
}
```
