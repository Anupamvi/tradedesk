import dataclasses
import unittest

from cultra.requesting import (
    ContingencyKind,
    ENDPOINT_RULES,
    Endpoint,
    EndpointPolicyError,
    PlanningError,
    RequestPlan,
    RunType,
    SecretMaterialError,
    build_reference_eod_plan,
    make_planned_request,
)


def tickers(count):
    return ["T%03d" % index for index in range(count)]


def options(count):
    return ["T%03d  270101C%08d" % (index, index + 1000) for index in range(count)]


class RequestPlanningTests(unittest.TestCase):
    def test_reference_funnel_is_deterministic_and_bounded(self):
        kwargs = dict(
            run_id="run-reference",
            core_symbols=tickers(80),
            summary_symbols=tickers(40),
            monies_symbols=tickers(20),
            option_symbols=options(200),
            expected_vintage="2026-08-29",
        )
        plan = build_reference_eod_plan(**kwargs)
        reversed_plan = build_reference_eod_plan(
            **dict(
                kwargs,
                core_symbols=list(reversed(kwargs["core_symbols"])),
                summary_symbols=list(reversed(kwargs["summary_symbols"])),
                monies_symbols=list(reversed(kwargs["monies_symbols"])),
                option_symbols=list(reversed(kwargs["option_symbols"])),
            )
        )
        self.assertEqual(18, plan.base_count)
        self.assertEqual(18, plan.logical_count)
        self.assertEqual(0, plan.retry_reserve)
        self.assertEqual(18, plan.worst_charged_attempts)
        self.assertEqual(plan.plan_hash, reversed_plan.plan_hash)
        counts = {}
        for item in plan.requests:
            counts[item.endpoint] = counts.get(item.endpoint, 0) + 1
        self.assertEqual(8, counts[Endpoint.CORES])
        self.assertEqual(4, counts[Endpoint.SUMMARIES])
        self.assertEqual(2, counts[Endpoint.MONIES_IMPLIED])
        self.assertEqual(2, counts[Endpoint.MONIES_FORECAST])
        self.assertEqual(2, counts[Endpoint.EXACT_OPTIONS])

    def test_funnel_rejects_oversize_instead_of_truncating(self):
        with self.assertRaises(PlanningError):
            build_reference_eod_plan(
                run_id="run-too-large",
                core_symbols=tickers(601),
                expected_vintage="2026-08-29",
                retry_reserve=0,
            )

    def test_daily_funnel_accepts_60_and_rejects_the_83_call_shape(self):
        plan = build_reference_eod_plan(
            run_id="cap-funnel",
            core_symbols=tickers(600),
            expected_vintage="2026-08-29",
        )
        self.assertEqual(60, plan.logical_count)
        self.assertEqual(60, plan.worst_charged_attempts)
        self.assertLessEqual(
            sum(item.max_response_bytes for item in plan.requests), 250_000_000
        )
        with self.assertRaisesRegex(PlanningError, "logical request cap"):
            build_reference_eod_plan(
                run_id="rejected-absolute-funnel",
                core_symbols=tickers(600),
                summary_symbols=tickers(120),
                monies_symbols=tickers(40),
                option_symbols=options(250),
                expected_vintage="2026-08-29",
            )

    def test_exact_contract_is_get_and_documented_not_probed(self):
        rule = ENDPOINT_RULES[Endpoint.EXACT_OPTIONS]
        self.assertEqual("GET", rule.method)
        self.assertEqual("DOCUMENTED_NOT_PROBED", rule.contract_status)

    def test_historical_exact_strikes_are_distinct_by_expiry_and_strike(self):
        def item(identifier, strike):
            return make_planned_request(
                logical_request_id=identifier,
                endpoint=Endpoint.HIST_STRIKES_OPTIONS,
                run_type=RunType.HISTORICAL_BACKFILL,
                entities=("AAPL",),
                fields=("ticker", "tradeDate", "strike"),
                field_profile="HIST_EXACT_STRIKE_SERIES_V1",
                purpose="exact series overlap test",
                expected_vintage="2026-08-29",
                expected_rows=100,
                expected_bytes=1000,
                retry_limit=0,
                params={"expirDate": "2026-10-16", "strike": strike},
            )

        plan = RequestPlan(
            run_id="different-exact-strikes",
            run_type=RunType.HISTORICAL_BACKFILL,
            requests=(item("strike-100", 100), item("strike-105", 105)),
            target=2,
            hard_cap=2,
            retry_reserve=0,
            campaign_id="exact-history-campaign",
            campaign_hard_cap=90,
        )
        self.assertEqual(2, plan.logical_count)

    def test_get_url_limit_is_rejected_before_any_permit(self):
        long_fields = tuple(
            "F%02d%s" % (index, "X" * 61) for index in range(64)
        )
        item = make_planned_request(
            logical_request_id="oversized-url",
            endpoint=Endpoint.EXACT_OPTIONS,
            run_type=RunType.EOD,
            entities=options(100),
            fields=long_fields,
            field_profile="URL_PREFLIGHT_V1",
            purpose="encoded URL preflight test",
            expected_vintage="2026-08-29",
            expected_rows=100,
            expected_bytes=1000,
            max_response_bytes=1_000_000,
            retry_limit=0,
        )
        with self.assertRaisesRegex(PlanningError, "encoded URL ceiling"):
            RequestPlan(
                run_id="oversized-url-plan",
                run_type=RunType.EOD,
                requests=(item,),
                target=1,
                hard_cap=1,
                retry_reserve=0,
            )

    def test_conditional_strikes_needs_authorization_and_bounds(self):
        common = dict(
            logical_request_id="conditional-strikes",
            endpoint=Endpoint.STRIKES,
            run_type=RunType.EOD,
            entities=["AAPL"],
            fields=["ticker", "tradeDate"],
            field_profile="STRIKES_NARROW_V1",
            purpose="explicit narrow validation",
            expected_vintage="2026-08-29",
            expected_rows=100,
            expected_bytes=1000,
            retry_limit=0,
        )
        with self.assertRaises(EndpointPolicyError):
            make_planned_request(**common)
        with self.assertRaises(EndpointPolicyError):
            make_planned_request(**dict(common, conditional_authorized=True))
        request = make_planned_request(
            **dict(
                common,
                conditional_authorized=True,
                params={
                    "ticker": "AAPL",
                    "fields": "ticker,tradeDate",
                    "dte_min": 20,
                    "dte_max": 60,
                    "delta_min": 0.1,
                    "delta_max": 0.9,
                },
            )
        )
        self.assertEqual(Endpoint.STRIKES, request.endpoint)

    def test_conditional_strikes_rejects_unfrozen_or_unbounded_filters(self):
        valid_params = {
            "ticker": "AAPL",
            "fields": "ticker,tradeDate",
            "dte_min": 20,
            "dte_max": 60,
            "delta_min": -0.9,
            "delta_max": 0.9,
        }
        common = dict(
            logical_request_id="strict-strikes",
            endpoint=Endpoint.STRIKES,
            run_type=RunType.EOD,
            entities=["AAPL"],
            fields=["ticker", "tradeDate"],
            field_profile="STRIKES_NARROW_V1",
            purpose="strict filter validation",
            expected_vintage="2026-08-29",
            expected_rows=100,
            expected_bytes=1000,
            retry_limit=0,
            conditional_authorized=True,
        )
        invalid = (
            dict(valid_params, ticker="MSFT"),
            dict(valid_params, fields="ticker"),
            dict(valid_params, fields=7),
            dict(valid_params, dte_min=-1),
            dict(valid_params, dte_min=61),
            dict(valid_params, dte_max=3651),
            dict(valid_params, delta_min=float("nan")),
            dict(valid_params, delta_min=0.5, delta_max=-0.5),
            dict(valid_params, delta_max=1.01),
            dict(valid_params, unexpected="not frozen"),
        )
        for index, params in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(EndpointPolicyError):
                make_planned_request(**dict(common, params=params))
        with self.assertRaises(EndpointPolicyError):
            make_planned_request(
                **dict(common, params=valid_params, expected_rows=100_001)
            )

    def test_ad_hoc_requests_cannot_bypass_frozen_batch_size(self):
        with self.assertRaises(EndpointPolicyError):
            make_planned_request(
                logical_request_id="oversize-batch",
                endpoint=Endpoint.CORES,
                run_type=RunType.EOD,
                entities=tickers(11),
                fields=["ticker"],
                field_profile="CORE_V1",
                purpose="batch-bound test",
                expected_vintage="2026-08-29",
                expected_rows=11,
                expected_bytes=1000,
                retry_limit=0,
            )

    def test_post_retries_require_a_frozen_idempotency_contract(self):
        common = dict(
            logical_request_id="backtest-post",
            endpoint=Endpoint.BACKTEST,
            run_type=RunType.BACKTEST_VALIDATION,
            entities=["JOB1"],
            fields=["status"],
            field_profile="BACKTEST_JOB_V1",
            purpose="POST retry policy test",
            expected_vintage="2026-08-29",
            expected_rows=1,
            expected_bytes=1000,
        )
        with self.assertRaises(PlanningError):
            make_planned_request(**dict(common, retry_limit=1))
        request = make_planned_request(**dict(common, retry_limit=0))
        self.assertEqual("POST", request.method)
        self.assertIsNone(request.idempotency_contract)

    @staticmethod
    def _core_item(
        identifier,
        entities,
        *,
        contingency=False,
        kind=None,
        parent=None,
        depth=0,
    ):
        return make_planned_request(
            logical_request_id=identifier,
            endpoint=Endpoint.CORES,
            run_type=RunType.EOD,
            entities=entities,
            fields=["ticker"],
            field_profile="CORE_V1",
            purpose="contingency-shape test",
            expected_vintage="2026-08-29",
            expected_rows=len(entities),
            expected_bytes=1000,
            retry_limit=0,
            contingency=contingency,
            contingency_kind=kind,
            contingency_parent_id=parent,
            split_depth=depth,
        )

    def test_automatic_contingency_execution_is_disabled(self):
        base = self._core_item("base", ["AAPL", "MSFT", "NVDA", "TSLA"])
        recovery = self._core_item(
            "recovery",
            ["AAPL", "MSFT"],
            contingency=True,
            kind=ContingencyKind.GROUPED_MISSING_RECOVERY,
        )
        split = self._core_item(
            "split-one",
            ["AAPL", "MSFT", "NVDA", "TSLA"],
            contingency=True,
            kind=ContingencyKind.SPLIT_CHILD,
            parent="base",
            depth=1,
        )
        split_two = self._core_item(
            "split-two",
            ["AAPL", "MSFT"],
            contingency=True,
            kind=ContingencyKind.SPLIT_CHILD,
            parent="split-one",
            depth=2,
        )
        with self.assertRaisesRegex(PlanningError, "recovery contingencies are disabled"):
            RequestPlan(
                run_id="bounded-contingencies",
                run_type=RunType.EOD,
                requests=(base, recovery, split, split_two),
                target=18,
                hard_cap=4,
                retry_reserve=0,
            )

    def test_contingency_shape_rejects_second_recovery(self):
        base = self._core_item("base", ["AAPL", "MSFT", "NVDA"])
        recoveries = tuple(
            self._core_item(
                "recovery-%d" % index,
                ["AAPL", "MSFT"],
                contingency=True,
                kind=ContingencyKind.GROUPED_MISSING_RECOVERY,
            )
            for index in range(2)
        )
        with self.assertRaises(PlanningError):
            RequestPlan(
                run_id="too-many-recoveries",
                run_type=RunType.EOD,
                requests=(base,) + recoveries,
                target=18,
                hard_cap=3,
                retry_reserve=0,
            )

    def test_contingency_shape_rejects_seventh_split_child_and_depth_three(self):
        first_entities = tickers(10)
        second_entities = tickers(4)
        second_entities = ["X" + value for value in second_entities]
        bases = (
            self._core_item("base-one", first_entities),
            self._core_item("base-two", second_entities),
        )
        children = []
        for index in range(5):
            children.append(
                self._core_item(
                    "child-%d" % index,
                    first_entities[index * 2 : index * 2 + 2],
                    contingency=True,
                    kind=ContingencyKind.SPLIT_CHILD,
                    parent="base-one",
                    depth=1,
                )
            )
        for index in range(2):
            children.append(
                self._core_item(
                    "child-x-%d" % index,
                    second_entities[index * 2 : index * 2 + 2],
                    contingency=True,
                    kind=ContingencyKind.SPLIT_CHILD,
                    parent="base-two",
                    depth=1,
                )
            )
        with self.assertRaises(PlanningError):
            RequestPlan(
                run_id="too-many-splits",
                run_type=RunType.EOD,
                requests=bases + tuple(children),
                target=18,
                hard_cap=9,
                retry_reserve=0,
            )
        with self.assertRaises(PlanningError):
            self._core_item(
                "depth-three",
                ["AAPL", "MSFT"],
                contingency=True,
                kind=ContingencyKind.SPLIT_CHILD,
                parent="depth-two",
                depth=3,
            )

    def test_offline_endpoints_cannot_enter_daily_run(self):
        with self.assertRaises(EndpointPolicyError):
            make_planned_request(
                logical_request_id="hist",
                endpoint=Endpoint.HIST_CORES,
                run_type=RunType.EOD,
                entities=["AAPL"],
                fields=["ticker", "tradeDate"],
                field_profile="HIST_CORE_V1",
                purpose="backfill",
                expected_vintage="2026-08-29",
                expected_rows=10,
                expected_bytes=1000,
                retry_limit=0,
            )

    def test_secret_key_is_rejected_before_plan_freeze(self):
        with self.assertRaises(SecretMaterialError):
            make_planned_request(
                logical_request_id="bad-secret",
                endpoint=Endpoint.CORES,
                run_type=RunType.EOD,
                entities=["AAPL"],
                fields=["ticker"],
                field_profile="CORE_V1",
                purpose="test",
                expected_vintage="2026-08-29",
                expected_rows=1,
                expected_bytes=100,
                retry_limit=0,
                params={"token": "must-not-enter"},
            )

    def test_overlapping_batches_are_rejected(self):
        def item(identifier, values):
            return make_planned_request(
                logical_request_id=identifier,
                endpoint=Endpoint.CORES,
                run_type=RunType.EOD,
                entities=values,
                fields=["ticker"],
                field_profile="CORE_V1",
                purpose="test",
                expected_vintage="2026-08-29",
                expected_rows=len(values),
                expected_bytes=1000,
                retry_limit=0,
            )

        with self.assertRaises(PlanningError):
            RequestPlan(
                run_id="overlap",
                run_type=RunType.EOD,
                requests=(item("one", ["AAPL", "MSFT"]), item("two", ["MSFT", "NVDA"])),
                target=25,
                hard_cap=2,
                retry_reserve=0,
            )

    def test_plan_is_frozen_and_automatic_retry_reserve_is_rejected(self):
        plan = build_reference_eod_plan(
            run_id="immutable",
            core_symbols=["AAPL"],
            expected_vintage="2026-08-29",
            retry_reserve=0,
            hard_cap=1,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.run_id = "changed"
        with self.assertRaises(PlanningError):
            RequestPlan(
                run_id="bad-envelope",
                run_type=RunType.EOD,
                requests=plan.requests,
                target=25,
                hard_cap=1,
                retry_reserve=1,
            )

    def test_small_reference_plan_has_no_automatic_retry_reserve(self):
        plan = build_reference_eod_plan(
            run_id="small-default",
            core_symbols=["AAPL"],
            expected_vintage="2026-08-29",
        )
        self.assertEqual(1, plan.logical_count)
        self.assertEqual(0, plan.retry_reserve)
        self.assertEqual(1, plan.worst_charged_attempts)

        exact = build_reference_eod_plan(
            run_id="exact-field",
            core_symbols=[],
            option_symbols=["AAPL  270101C00100000"],
            expected_vintage="2026-08-29",
        )
        self.assertIn("optionSymbol", exact.requests[0].fields)


if __name__ == "__main__":
    unittest.main()
