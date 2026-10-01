"""Offline contracts for the international daily-activity turn: no upstream calls.

Everything here drives ``app.daily_chat`` and ``app.acp_client`` against fakes, so
nothing reaches WorkBuddy. The properties worth pinning are the safety ones: the
turn is one-shot per day, a sent-but-unconfirmed attempt is never replayed, no
upstream text escapes into results, and the ACP channel declares no client
capabilities (otherwise the sandbox would wait for callbacks that never come).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import unittest
from unittest.mock import patch

import converter
from app import acp_client, daily_chat
from app.control_store import ControlStore
from tests import test_credential_actions as fixtures


class FakeConsole:
    """In-memory stand-in for the three console endpoints plus one sandbox turn."""

    def __init__(self, *, conversation_id="conv-1", sandbox_status="completed",
                 link="https://sandbox.invalid/acp", token="sandbox-token",
                 session_id="sess-1", cwd="/workspace", fail_create=False):
        self.conversation_id = conversation_id
        self.sandbox_status = sandbox_status
        self.session = {"link": link, "token": token, "session_id": session_id, "cwd": cwd}
        self.fail_create = fail_create
        self.created = []
        self.statuses = []
        self.posts = []
        self.methods = []
        self.channel_opens = 0

    def install(self, test):
        # Lambdas, not bound methods: patch.object installs the object as a class
        # attribute, and a bound method is not a descriptor, so instance.open()
        # would call it with no arguments instead of the channel.
        console = self
        test.enterContext(patch.object(daily_chat, "create_conversation", self.create))
        test.enterContext(patch.object(daily_chat, "sandbox_of", self.sandbox_of))
        test.enterContext(patch.object(daily_chat, "status_of", self.status_of))
        test.enterContext(patch.object(acp_client.AcpChannel, "open",
                                       lambda channel: console.open(channel)))
        test.enterContext(patch.object(acp_client.AcpChannel, "post",
                                       lambda channel, method, params, request_id:
                                           console.post(channel, method, params, request_id)))
        test.enterContext(patch.object(acp_client.AcpChannel, "drain",
                                       lambda channel, seconds: console.drain(channel, seconds)))
        test.enterContext(patch.object(acp_client.AcpChannel, "close",
                                       lambda channel: console.close(channel)))
        test.enterContext(patch.object(daily_chat.credits, "fetch_request_usage",
                                       side_effect=self.usage))

    def usage(self, token, days=1, uid="", domain=""):
        return {"by_day": {}, "total_credits": 0.0, "requests": 0, "partial": False}

    def create(self, client, headers):
        self.created.append(headers)
        if self.fail_create:
            raise daily_chat.ChatFailure("http_error", 503)
        return self.conversation_id

    def sandbox_of(self, client, headers, conversation_id):
        return dict(self.session, conversation_id=conversation_id)

    def status_of(self, client, headers, conversation_id):
        self.statuses.append(conversation_id)
        return self.sandbox_status

    def open(self, channel_self):
        self.channel_opens += 1
        channel_self.connection_id = "conn-1"
        channel_self._closed = False
        return channel_self

    def post(self, channel_self, method, params, request_id):
        self.methods.append((method, request_id))
        self.posts.append({"method": method, "params": params, "id": request_id})

    def drain(self, channel_self, seconds):
        self.drained = getattr(self, "drained", 0) + 1
        if self.drained == 1:
            return [{"sessionUpdate": "usage_update",
                     "cost": {"amount": 3, "currency": "credits"}}]
        return []

    def close(self, channel_self):
        channel_self._closed = True

    def run_turn(self, link, token, session_id, cwd, prompt):
        """Kept only so the old assertions on turn arguments still have a place."""
        self.channel_opens += 1
        self.posts.append({"link": link, "token": token, "session_id": session_id,
                           "cwd": cwd, "prompt": prompt})
        return FakeChannel(self)

    def channel_post(self, method, params, request_id):
        self.posts.append({"method": method, "params": params, "id": request_id})


class FakeChannel:
    """Stands in for the ACP channel: one drained usage update, then done."""

    def __init__(self, console):
        self.console = console
        self.closed = False
        self.drained = 0

    def drain(self, seconds):
        self.drained += 1
        if self.drained == 1:
            return [{"sessionUpdate": "usage_update",
                     "cost": {"amount": 3, "currency": "credits"}}]
        return []

    def close(self):
        self.closed = True


class DailyChatTests(fixtures.CredentialActionTests):
    add_account = fixtures.CredentialActionTests.add_account
    configure = fixtures.CredentialActionTests.configure
    handle_upstream = fixtures.CredentialActionTests.handle_upstream

    def setUp(self):
        fixtures.CredentialActionTests.setUp(self)
        self.intl = self.entries["intl-work"]
        self.console = FakeConsole()
        self.console.install(self)

    def turn(self, entry=None, **kwargs):
        return daily_chat.perform("synthetic-token", (entry or self.intl).get("profile"),
            uid="synthetic-uid", domain="workbuddy.ai",
            store=self.control, identity=(entry or self.intl)["account_key"], **kwargs)

    # -- profile gating ------------------------------------------------------

    def test_only_international_workbuddy_accounts_are_supported(self):
        self.assertEqual(daily_chat.supported("intl-work"), True)
        for profile in ("intl-cli", "cn-cli", "cn-work"):
            self.assertFalse(daily_chat.supported(profile), profile)
        self.assertTrue(daily_chat.unavailable()["skipped"])

    def test_domestic_profile_never_reaches_the_console(self):
        self.assertFalse(self.turn(self.entries["cn-work"])["ok"])
        self.assertEqual(self.console.created, [])

    # -- happy path ----------------------------------------------------------

    def test_completed_turn_is_confirmed_and_records_cost(self):
        result = self.turn()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["state"], "done")
        self.assertEqual(result["sandbox_status"], "completed")
        self.assertEqual(result["conversation_id"], "conv-1")
        # Cost comes from the protocol-native usage_update, not from a subtraction.
        self.assertEqual(result["acp_usage"], 3.0)
        self.assertEqual(len(self.console.created), 1)
        record = self.control.daily_chat_record(self.intl["account_key"], result["day"])
        self.assertEqual(record["phase"], "confirmed")
        self.assertTrue(self.control.daily_chat_done(self.intl["account_key"], result["day"]))

    def test_console_headers_carry_no_cli_fingerprint(self):
        self.turn()
        headers = self.console.created[0]
        self.assertEqual(headers["x-user-id"], "synthetic-uid")
        self.assertEqual(headers["referer"], daily_chat.HOST + "/app")
        self.assertTrue(headers["authorization"].startswith("Bearer "))
        # A CLI X-IDE-* fingerprint is exactly what fails to earn the reward.
        for name in headers:
            self.assertFalse(name.lower().startswith("x-ide-"), name)

    def test_acp_sequence_uses_ordered_ids_and_closed_capabilities(self):
        self.turn()
        self.assertEqual([method for method, _ in self.console.methods],
                         ["initialize", "session/load", "session/prompt"])
        self.assertEqual([request_id for _, request_id in self.console.methods], [1, 2, 3])
        by_method = {method: params for method, params in self.console.methods
                     for params in [next(p["params"] for p in self.console.posts
                                         if p["method"] == method)]}
        self.assertEqual(by_method["initialize"]["protocolVersion"], 1)
        capabilities = by_method["initialize"]["clientCapabilities"]
        self.assertIs(capabilities["fs"]["readTextFile"], False)
        self.assertIs(capabilities["fs"]["writeTextFile"], False)
        self.assertIs(capabilities["terminal"], False)
        self.assertEqual(by_method["session/load"]["sessionId"], "sess-1")
        self.assertEqual(by_method["session/load"]["cwd"], "/workspace")
        self.assertEqual(by_method["session/prompt"]["prompt"],
                         [{"type": "text", "text": daily_chat.PROMPT}])

    # -- one-shot discipline -------------------------------------------------

    def test_second_turn_the_same_day_sends_nothing(self):
        self.turn()
        self.assertEqual(len(self.console.created), 1)
        self.assertEqual(len(self.console.methods), 3)
        again = self.turn()
        self.assertTrue(again["ok"])
        self.assertEqual(again["state"], "done")
        self.assertTrue(again["skipped"])
        self.assertEqual(len(self.console.created), 1, "a confirmed day is never reopened")
        self.assertEqual(len(self.console.methods), 3)

    def test_unconfirmed_send_is_never_replayed(self):
        self.control.reserve_daily_chat(self.intl["account_key"], "2026-09-30")
        self.control.transition_daily_chat(self.intl["account_key"], "2026-09-30",
            self.control.daily_chat_record(self.intl["account_key"], "2026-09-30")["attempt_id"], "sent")
        result = self.turn()
        self.assertEqual(result["state"], "pending")
        self.assertTrue(result["skipped"])
        self.assertEqual(self.console.created, [], "an unconfirmed send must not be replayed")

    def test_failed_today_allows_tomorrow(self):
        self.console.sandbox_status = "failed"
        result = self.turn()
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "rejected")
        record = self.control.daily_chat_record(self.intl["account_key"], result["day"])
        # 'sent' means the request went out: it stays unconfirmed for the rest of the day.
        self.assertEqual(record["phase"], "sent")
        self.assertFalse(self.control.daily_chat_done(self.intl["account_key"], result["day"]))

    def test_cancelled_day_can_be_retried_manually(self):
        day = "2026-09-30"
        attempt = self.control.reserve_daily_chat(self.intl["account_key"], day)["attempt_id"]
        self.control.transition_daily_chat(self.intl["account_key"], day, attempt, "cancelled")
        reserved = self.control.reserve_daily_chat(self.intl["account_key"], day)
        self.assertIsNotNone(reserved)

    # -- failure containment -------------------------------------------------

    def test_upstream_http_failure_cancels_unsent_reservation(self):
        self.console.fail_create = True
        result = self.turn()
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "http_error")
        record = self.control.daily_chat_record(self.intl["account_key"], result["day"])
        self.assertEqual(record["phase"], "cancelled")
        self.assertEqual(self.console.channel_opens, 0, "no sandbox turn was opened")
    def test_timeout_leaves_the_reservation_uncancelled(self):
        self.console.sandbox_status = "working"
        with patch.object(daily_chat, "TURN_SECONDS", 1.0), patch.object(daily_chat, "POLL_SECONDS", 0.4):
            result = self.turn()
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "timeout")
        record = self.control.daily_chat_record(self.intl["account_key"], result["day"])
        # The turn was sent, so it stays unconfirmed; the next attempt is tomorrow.
        self.assertEqual(record["phase"], "sent")
        self.assertFalse(self.control.daily_chat_done(self.intl["account_key"], result["day"]))

    def test_unwritable_store_stops_before_any_request(self):
        with patch.object(self.control, "daily_chat_record", side_effect=OSError("synthetic")):
            result = self.turn()
        self.assertEqual(result["state"], "storage_error")
        self.assertEqual(self.console.created, [])

    def test_generation_change_cancels_before_sending(self):
        result = self.turn(can_write=lambda: False)
        self.assertEqual(result["state"], "changed")
        self.assertTrue(result["skipped"])
        self.assertEqual(self.console.created, [])

    def test_no_upstream_text_reaches_the_result(self):
        with patch.object(daily_chat, "create_conversation",
                          side_effect=daily_chat.ChatFailure("rejected", 429, 11140)):
            result = self.turn()
        # Keys such as sandbox_status are our own vocabulary; only *values* must be clean.
        def values(node):
            if isinstance(node, dict):
                for value in node.values():
                    yield from values(value)
            elif isinstance(node, list):
                for item in node:
                    yield from values(item)
            elif isinstance(node, str):
                yield node
        for text in values(result):
            for forbidden in ("synthetic-token", "synthetic-uid", "sandbox.invalid", "Bearer"):
                self.assertNotIn(forbidden, text, forbidden)
        self.assertEqual(result["code"], 11140)
        self.assertEqual(result["http_status"], 429)

    # -- manual action and preference ---------------------------------------

    def test_manual_action_runs_one_turn_and_audits_it(self):
        response = self.client.post("/admin/credentials/" + self.intl["account_key"] + "/daily-chat")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["results"][0]
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["state"], "done")
        self.assertEqual(len(self.console.created), 1)

    def test_preference_defaults_off_and_persists(self):
        row = next(r for r in self.client.get("/admin/credentials").json()["credentials"]
                   if r["id"] == self.intl["account_key"])
        self.assertIs(row["auto_daily_chat"], False)
        self.assertIs(row["daily_chat_supported"], True)
        self.assertEqual(row["daily_chat"]["state"], "available")
        response = self.client.patch("/admin/credentials/" + self.intl["account_key"],
                                     json={"auto_daily_chat": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(converter.model_policy.credential_auto_daily_chat(converter.CONFIG, self.intl))

    def test_preference_is_rejected_for_domestic_accounts(self):
        response = self.client.patch("/admin/credentials/" + self.entries["cn-work"]["account_key"],
                                     json={"auto_daily_chat": True})
        self.assertEqual(response.status_code, 400, response.text)

    def test_unknown_action_is_still_rejected(self):
        response = self.client.post(self.url + "/daily-checkin")
        self.assertEqual(response.status_code, 404, response.text)

    # -- store contracts -----------------------------------------------------

    def test_reserve_rejects_a_second_reservation_the_same_day(self):
        day = "2026-09-30"
        first = self.control.reserve_daily_chat(self.intl["account_key"], day)
        self.assertIsNotNone(first)
        self.assertIsNone(self.control.reserve_daily_chat(self.intl["account_key"], day))

    def test_transition_rejects_an_unknown_attempt(self):
        day = "2026-09-30"
        attempt = self.control.reserve_daily_chat(self.intl["account_key"], day)["attempt_id"]
        with self.assertRaises(ValueError):
            self.control.transition_daily_chat(self.intl["account_key"], day, "not-the-attempt", "sent")
        self.assertIsNotNone(attempt)

    def test_day_must_be_a_plain_date(self):
        for bad in ("2026-9-30", "20260930", 20260930, "2026-09-30T00:00:00", ""):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.control.daily_chat_record(self.intl["account_key"], bad)

    def test_prune_drops_only_old_days(self):
        identity = self.intl["account_key"]
        for day in ("2020-01-01", "2999-01-01"):
            attempt = self.control.reserve_daily_chat(identity, day)["attempt_id"]
            self.control.transition_daily_chat(identity, day, attempt, "sent")
            self.control.transition_daily_chat(identity, day, attempt, "confirmed")
        self.control.prune_daily_chats(keep_days=30)
        self.assertIsNone(self.control.daily_chat_record(identity, "2020-01-01"))
        self.assertIsNotNone(self.control.daily_chat_record(identity, "2999-01-01"))


class AcpClientTests(unittest.TestCase):
    """Protocol-shape guards that need no network at all."""

    def test_link_must_be_http_and_keeps_its_query(self):
        for bad in ("", "ftp://x/y", "notaurl", "https:///path"):
            with self.assertRaises(acp_client.AcpError, msg=bad):
                acp_client.AcpChannel(bad, "t").open()
        channel = acp_client.AcpChannel.__new__(acp_client.AcpChannel)
        channel.parts, channel.path = acp_client._target("https://sandbox.invalid/acp?a=1&b=2")
        self.assertEqual(channel.path, "/acp?a=1&b=2")

    def test_unknown_method_is_refused_before_any_connection(self):
        channel = acp_client.AcpChannel.__new__(acp_client.AcpChannel)
        channel._closed = False
        channel.connection_id = "conn-1"
        with self.assertRaises(acp_client.AcpError):
            channel.post("session/delete", {}, 4)

    def test_drain_harvests_usage_updates_only(self):
        channel = acp_client.AcpChannel.__new__(acp_client.AcpChannel)
        channel._closed, channel._response, channel._buffer, channel._line = False, object(), bytearray(), bytearray()
        channel._buffer.extend(
            b"data: {\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"update\":"
            b"{\"sessionUpdate\":\"usage_update\",\"cost\":{\"amount\":7,\"currency\":\"credits\"}}}}\n"
            b"data: {\"method\":\"session/update\",\"params\":{\"update\":{\"sessionUpdate\":"
            b"\"agent_message_chunk\",\"content\":{\"text\":\"hi\"}}}}\n"
            b": heartbeat\n\n")
        self.assertEqual(channel._events(), [{"sessionUpdate": "usage_update",
                                              "cost": {"amount": 7, "currency": "credits"}}])
