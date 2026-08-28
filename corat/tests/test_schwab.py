import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from corat.schwab import SchwabClient, load_credentials, quote_is_fresh, quote_to_bar


class FakeResponse:
    def __init__(self, payload):
        self.payload=json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def getcode(self): return 200
    def read(self): return self.payload


class SchwabTest(unittest.TestCase):
    def env(self, root):
        token=root/"token.json"
        token.write_text(json.dumps({"token":{"access_token":"SCHWAB_SECRET","expires_at":time.time()+3600,"refresh_token":"unused"}}),encoding="utf-8")
        env=root/".env"
        env.write_text("SCHWAB_TOKEN_PATH=token.json\n",encoding="utf-8")
        return env,token

    def test_relative_token_path_resolves_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,token=self.env(Path(tmp))
            before=token.read_bytes()
            creds=load_credentials(env)
            self.assertEqual(creds["access_token"],"SCHWAB_SECRET")
            self.assertEqual(before,token.read_bytes())

    def test_quote_cache_excludes_access_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); env,token=self.env(root)
            payload={"SPY":{"quote":{"lastPrice":700,"openPrice":695,"highPrice":702,"lowPrice":694,"totalVolume":1000000,"quoteTimeInLong":int(time.time()*1000)}}}
            client=SchwabClient(env,"https://api.schwab.test",root/"cache")
            with mock.patch("corat.schwab.urllib.request.urlopen",return_value=FakeResponse(payload)):
                result=client.fetch_quotes(["SPY"])
            self.assertEqual(len(result.quotes),1)
            cache=Path(result.traces[0].cache_path).read_text()
            self.assertNotIn("SCHWAB_SECRET",cache)
            self.assertEqual(result.traces[0].source,"SCHWAB")

    def test_quote_bar_and_freshness(self):
        payload={"quote":{"lastPrice":100,"openPrice":98,"highPrice":101,"lowPrice":97,"totalVolume":5000,"quoteTimeInLong":int(time.time()*1000)}}
        bar=quote_to_bar("AAA",payload,"2026-08-27")
        self.assertEqual(bar.close,100)
        self.assertEqual(bar.source,"Schwab read-only quote")
        self.assertTrue(quote_is_fresh(payload,30))

