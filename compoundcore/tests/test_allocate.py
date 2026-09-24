import unittest

from compoundcore.allocate import allocate_cents, allocate_dollars, distribution


class TestAllocate(unittest.TestCase):
    def test_hundred_k_default(self):
        d = allocate_dollars(100000, "default")
        self.assertEqual(d["VOO"], 48000.0)
        self.assertEqual(d["VGT"], 10000.0)
        self.assertEqual(d["SMH"], 7000.0)
        self.assertEqual(d["VB"], 5000.0)
        self.assertEqual(d["VXUS"], 20000.0)
        self.assertEqual(d["GLDM"], 5000.0)
        self.assertEqual(d["VGSH"], 5000.0)
        self.assertEqual(sum(d.values()), 100000.0)

    def test_cents_sum_on_awkward_amount(self):
        cents = allocate_cents(33333.33, "default")
        self.assertEqual(sum(cents.values()), 3333333)
        cents2 = allocate_cents(1.00, "aggressive")
        self.assertEqual(sum(cents2.values()), 100)

    def test_weekly_recipe(self):
        dist = distribution(250000, "default", weekly=500)
        self.assertEqual(dist["rows"][0]["dollars"], 120000.0)
        week = {row["ticker"]: row["weekly"] for row in dist["rows"]}
        self.assertEqual(week["VOO"], 240.0)
        self.assertEqual(week["SMH"], 35.0)
        self.assertEqual(sum(week.values()), 500.0)

    def test_per_thousand(self):
        dist = distribution(0, "default")
        per = {row["ticker"]: row["per_1000"] for row in dist["rows"]}
        self.assertEqual(per["VOO"], 480.0)
        self.assertEqual(sum(per.values()), 1000.0)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            allocate_dollars(-1, "default")
        with self.assertRaises(ValueError):
            distribution(100, "default", weekly=-5)
