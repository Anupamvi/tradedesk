import json
import tempfile
import unittest
from pathlib import Path

from corat.ledger import read_events, record_plan, record_trade_event, review_open_trades, trade_states


class LedgerTest(unittest.TestCase):
    def run_file(self, root):
        path=root/"run.json"
        path.write_text(json.dumps({"as_of":"2026-08-27","candidates":[{
            "ticker":"AAA","vehicle":"STOCK","vehicle_reason":"fixture","score":80,"confidence":"MEDIUM",
            "setup":{"name":"BREAKOUT + CONFIRMATION","direction":"BULLISH","reason":"original","invalidation":"below stop"},
            "stock_plan":{"stop":95,"target_1":110,"target_2":120},"option":{},"history":{"expectancy":0.02},
        }]}),encoding="utf-8")
        return path

    def test_plan_and_state_transitions_are_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/"ledger.jsonl"
            event=record_plan(ledger,self.run_file(root),"AAA",trade_id="t1")
            record_trade_event(ledger,"t1","SUBMITTED")
            record_trade_event(ledger,"t1","FILLED",price=100,quantity=10)
            record_trade_event(ledger,"t1","OPEN")
            self.assertEqual(len(read_events(ledger)),4)
            self.assertEqual(trade_states(read_events(ledger))[0]["status"],"OPEN")

    def test_invalid_transition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/"ledger.jsonl"
            record_plan(ledger,self.run_file(root),"AAA",trade_id="t1")
            with self.assertRaises(ValueError):
                record_trade_event(ledger,"t1","CLOSED",price=99)

    def test_review_uses_original_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/"ledger.jsonl"
            record_plan(ledger,self.run_file(root),"AAA",trade_id="t1")
            record_trade_event(ledger,"t1","SUBMITTED")
            record_trade_event(ledger,"t1","FILLED",price=100,quantity=10)
            record_trade_event(ledger,"t1","OPEN")
            current={"as_of":"2026-08-28","candidates":[{"ticker":"AAA","technical":{"price":94},"setup":{"direction":"BULLISH"},"status":"WAIT FOR TRIGGER","blockers":[]}]}
            review=review_open_trades(read_events(ledger),current)
            self.assertEqual(review["reviews"][0]["action"],"EXIT")
            self.assertIn("Original technical invalidation",review["reviews"][0]["reason"])

    def test_add_requires_predefined_zone_capacity_and_current_actionability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/"ledger.jsonl"
            record_plan(
                ledger,
                self.run_file(root),
                "AAA",
                trade_id="t1",
                predefined_scaling={
                    "enabled": True,
                    "add_entry_low": 98,
                    "add_entry_high": 102,
                    "add_quantity": 5,
                    "maximum_total_quantity": 20,
                },
            )
            record_trade_event(ledger,"t1","SUBMITTED")
            record_trade_event(ledger,"t1","FILLED",price=100,quantity=10)
            record_trade_event(ledger,"t1","OPEN",price=100,quantity=10)
            current={"as_of":"2026-08-28","candidates":[{
                "ticker":"AAA","technical":{"price":100},"setup":{"direction":"BULLISH"},
                "status":"ACTIONABLE NOW","blockers":[],"hard_rejections":[],
            }]}
            review=review_open_trades(read_events(ledger),current)["reviews"][0]
            self.assertEqual(review["action"],"ADD")
            self.assertEqual(review["recommended_add_quantity"],5)

    def test_invalid_scaling_plan_is_rejected_before_ledger_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ledger=root/"ledger.jsonl"
            with self.assertRaisesRegex(ValueError,"scaling plan"):
                record_plan(
                    ledger,
                    self.run_file(root),
                    "AAA",
                    trade_id="t1",
                    predefined_scaling={"enabled":True,"add_entry_low":100},
                )
            self.assertFalse(ledger.exists())
