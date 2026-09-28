"""Test credential refresh, concurrency, discovery and cooldowns with temporary files and mocks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Allow direct execution.

import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx

import converter


class CredentialRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.enterContext(patch.dict(os.environ, {"CODEBUDDY_AUTH_DIR": temporary}))
        self.enterContext(patch.dict(converter.CONFIG, {"log_path": None, "cred_pool": None, "cred": None}))
        self.logs = self.enterContext(patch.object(converter, "_log"))
        self.calls = 0
        real_client = httpx.Client
        transport = httpx.MockTransport(self.refresh_response)
        self.enterContext(patch.object(converter.httpx, "Client",
                                       side_effect=lambda **kw: real_client(transport=transport, **kw)))

    def refresh_response(self, request):
        self.assertEqual(request.url.path, "/v2/plugin/auth/token/refresh")
        self.calls += 1
        return httpx.Response(200, json={"code": 0, "data": {
            "accessToken": "new-synthetic-token", "expiresIn": 7200}})

    def credential(self, name="account.info", uid="synthetic", expires_in=86400, age=25 * 3600):
        now = time.time()
        path = self.root / name
        path.write_text(json.dumps({"account": {"uid": uid}, "auth": {
            "accessToken": "old-synthetic-token", "refreshToken": "synthetic-refresh",
            "domain": "www.codebuddy.cn", "expiresAt": (now + expires_in) * 1000,
            "lastRefreshTime": (now - age) * 1000}}), encoding="utf-8")
        return path

    def test_daily_keepalive_really_refreshes_once_and_persists(self):
        path = self.credential()
        pool = converter.CredentialPool([path])
        pool.refresh_due()
        pool.refresh_due()
        self.assertEqual(self.calls, 1)
        auth = json.loads(path.read_text())["auth"]
        self.assertEqual(auth["accessToken"], "new-synthetic-token")
        self.assertEqual(auth["refreshToken"], "synthetic-refresh")
        self.assertGreater(auth["lastRefreshTime"], (time.time() - 5) * 1000)
        success = [c for c in self.logs.call_args_list if "并回写" in c.args[0]]
        self.assertEqual(len(success), 1)
        if os.name != "nt":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_five_minute_margin_refreshes_before_get_headers_expiry(self):
        path = self.credential(expires_in=300, age=0)
        pool = converter.CredentialPool([path])
        self.assertFalse(pool.first().summary()["token_expired"])
        pool.refresh_due()
        self.assertEqual(self.calls, 1)
        self.assertIn("new-synthetic-token", pool.first().get_headers()["Authorization"])

    def test_fresh_or_keepalive_disabled_does_not_refresh(self):
        fresh = self.credential(age=0)
        converter.CredentialPool([fresh]).refresh_due()
        stale = self.credential("stale.info", uid="stale")
        converter.CredentialPool([stale]).refresh_due(keepalive_s=0)
        self.assertEqual(self.calls, 0)

    def test_concurrent_foreground_and_background_refresh_only_once(self):
        path = self.credential(expires_in=-1)
        pool = converter.CredentialPool([path])
        manager = pool.first()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(manager.get_headers if i % 2 else pool.refresh_due)
                       for i in range(20)]
            for future in futures:
                future.result(timeout=5)
        self.assertEqual(self.calls, 1)
        self.assertIs(pool.first(), manager)

    def test_failed_persistence_is_not_logged_as_success_and_backs_off(self):
        path = self.credential()
        original = path.read_bytes()
        pool = converter.CredentialPool([path])
        with patch.object(converter, "atomic_write_credential", side_effect=OSError("synthetic disk failure")):
            pool.refresh_due()
            pool.refresh_due()
        self.assertEqual(self.calls, 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertGreater(pool._entries[0]["keepalive_after"], time.time())
        self.assertGreater(pool._entries[0]["fail_until"], time.time())
        self.assertFalse(any("并回写" in c.args[0] for c in self.logs.call_args_list))

    def test_rescan_preserves_manager_lock_and_auth_cooldown(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path], scan=True)
        manager = pool.first()
        lock = manager._lock
        pool.note_status(manager, 401)
        until = pool._entries[0]["fail_until"]
        for model in ("glm-5.3-flash", None):
            self.assertIsNone(pool.pick("session", model))
        self.assertIs(pool.first(), manager)
        self.assertIs(manager._lock, lock)
        self.assertEqual(pool._entries[0]["fail_until"], until)

    def test_status_from_inflight_manager_still_applies_after_rescan(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path], scan=True)
        manager = pool.pick("session", "glm-5.3-flash")
        pool.pick("session", "glm-5.3-flash")
        pool.note_status(manager, 429, model="glm-5.3-flash")
        self.assertIsNone(pool.pick("session", "glm-5.3-flash"))
        self.assertIs(pool.pick("another", "deepseek-v4-flash"), manager)

    def test_explicit_import_reloads_without_replacing_manager(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path], scan=True)
        manager = pool.first()
        pool.cooldown(manager)
        pool.note_status(manager, 429, model="glm-5.3-flash")
        data = json.loads(path.read_text())
        data["auth"]["accessToken"] = "reimported-synthetic-token"
        path.write_text(json.dumps(data))
        pool.reload([path])
        self.assertIs(pool.first(), manager)
        self.assertEqual(pool._entries[0]["fail_until"], 0)
        self.assertIn("reimported-synthetic-token", manager.get_headers()["Authorization"])
        self.assertIsNone(pool.pick("session", "glm-5.3-flash"))  # Relogin preserves quota cooldown.

    def test_duplicate_notice_is_not_emitted_on_every_request(self):
        path = self.credential(age=0)
        duplicate = self.credential("duplicate.info", age=0)
        pool = converter.CredentialPool([path, duplicate], scan=True)
        for _ in range(5):
            pool.pick("session", "glm-5.3-flash")
        notices = [c for c in self.logs.call_args_list if "忽略重复" in c.args[0]]
        self.assertEqual(len(notices), 1)
        self.assertEqual(len(pool.entries()), 1)
        duplicate.unlink()
        pool._rescan()
        self.assertFalse(pool._ignored_duplicates)

    def during_refresh(self, pool, operation):
        manager = pool.first()
        original = manager._refresh_locked
        started, release, operation_started = threading.Event(), threading.Event(), threading.Event()
        before = manager.path.read_bytes()

        def gated():
            started.set()
            self.assertTrue(release.wait(5))
            return original()

        def operate():
            operation_started.set()
            return operation()

        with patch.object(manager, "_refresh_locked", side_effect=gated), ThreadPoolExecutor(max_workers=2) as executor:
            refresh = executor.submit(pool.refresh_due)
            self.assertTrue(started.wait(5))
            pending = executor.submit(operate)
            try:
                self.assertTrue(operation_started.wait(5))
                with self.assertRaises(TimeoutError):
                    pending.result(timeout=0.05)
                self.assertEqual(manager.path.read_bytes(), before)
            finally:
                release.set()
            refresh.result(timeout=5)
            return pending.result(timeout=5)

    def test_relogin_waits_for_refresh_and_new_login_wins(self):
        path = self.credential()
        pool = converter.CredentialPool([path])
        converter.CONFIG["cred_pool"] = pool
        cred = json.loads(path.read_text())
        cred["auth"].update(accessToken="imported-synthetic-token", lastRefreshTime=time.time() * 1000)
        result = self.during_refresh(pool, lambda: converter._save_oauth_credential(cred))
        self.assertEqual(result, path)
        self.assertEqual(json.loads(path.read_text())["auth"]["accessToken"], "imported-synthetic-token")

    def test_independent_managers_share_file_lock_and_recheck(self):
        path = self.credential(expires_in=-1)
        pool = converter.CredentialPool([path])
        independent = converter.CredentialManager(path)
        headers = self.during_refresh(pool, independent.get_headers)
        self.assertEqual(self.calls, 1)
        self.assertIn("new-synthetic-token", headers["Authorization"])

    def test_delete_waits_for_refresh_and_is_not_resurrected(self):
        path = self.credential()
        pool = converter.CredentialPool([path])
        self.assertTrue(self.during_refresh(pool, lambda: pool.remove_file(path.name)))
        self.assertFalse(path.exists())
        self.assertIsNone(pool.first())

    def test_old_request_status_cannot_cooldown_reimported_credential(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path])
        converter.CONFIG["cred_pool"] = pool
        old_lease, _ = pool.headers_for("session", "glm-5.3-flash", with_generation=True)
        data = json.loads(path.read_text())
        data["auth"]["accessToken"] = "imported-synthetic-token"
        converter._store_credential(self.root, path.name, json.dumps(data).encode(), "synthetic")
        for status in (401, 403, 429):
            converter._note_cred_status(old_lease, status, model="glm-5.3-flash")
        self.assertIsNotNone(pool.pick("session", "glm-5.3-flash"))
        current, _ = pool.headers_for("session", "glm-5.3-flash", with_generation=True)
        converter._note_cred_status(current, 401)
        self.assertIsNone(pool.pick("session", "glm-5.3-flash"))

    def test_external_atomic_import_also_invalidates_old_request_generation(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path])
        converter.CONFIG["cred_pool"] = pool
        old_lease, _ = pool.headers_for("session", "glm-5.3-flash", with_generation=True)
        data = json.loads(path.read_text())
        data["auth"]["accessToken"] = "external-synthetic-token"
        with patch.dict(converter.CONFIG, {"cred_pool": None}):
            converter._store_credential(self.root, path.name, json.dumps(data).encode(), "synthetic")
        converter._note_cred_status(old_lease, 401)
        self.assertIsNotNone(pool.pick("session", "glm-5.3-flash"))

    def test_replacing_account_clears_previous_accounts_model_cooldown(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path])
        converter.CONFIG["cred_pool"] = pool
        pool.note_status(pool.first(), 429, model="glm-5.3-flash")
        data = json.loads(path.read_text())
        data["account"]["uid"] = "replacement-user"
        converter._store_credential(self.root, path.name, json.dumps(data).encode(), "replacement-user")
        self.assertIsNotNone(pool.pick("session", "glm-5.3-flash"))
        self.assertEqual(pool.first().summary()["uid"], "replacement-user")


    def test_unreadable_credential_does_not_starve_other_accounts(self):
        bad = self.root / "bad.info"
        bad.write_text("not json")
        good = self.credential("good.info", uid="good", age=0)
        pool = converter.CredentialPool([bad, good], scan=True)
        manager, _ = pool.headers_for(None, "glm-5.3-flash")
        self.assertEqual(manager.path, good)
        self.assertGreater(pool._entries[0]["fail_until"], time.time())


    def test_directory_changes_still_add_and_remove_accounts(self):
        first = self.credential(age=0)
        pool = converter.CredentialPool([first], scan=True)
        second = self.credential("second.info", uid="second", age=0)
        pool._rescan()
        self.assertEqual(len(pool.entries()), 2)
        first.unlink()
        pool._rescan()
        self.assertEqual([e["id"] for e in pool.entries()], [str(second)])

    def test_encrypted_envelope_tokens_fail_fast_and_stay_out_of_pool(self):
        valid = self.credential("valid.info", uid="valid", age=0)
        envelope = self.root / "envelope.info"
        envelope.write_text(json.dumps({"account": {"uid": "envelope"}, "auth": {
            "accessToken": {"$wbEncrypted": 1, "envelope": "c3lu"},
            "refreshToken": {"$wbEncrypted": 1, "envelope": "c3lu"},
            "domain": "www.workbuddy.ai", "expiresAt": (time.time() + 86400) * 1000}}),
            encoding="utf-8")
        pool = converter.CredentialPool([valid, envelope])
        self.assertEqual([Path(e["id"]).name for e in pool.entries()], ["valid.info"])
        rejected = [c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]]
        self.assertEqual(len(rejected), 1)
        self.assertIn("accessToken", rejected[0].args[0])
        pool.reload([valid, envelope], reset=False)  # Repeat reloads neither re-log nor admit.
        self.assertEqual([Path(e["id"]).name for e in pool.entries()], ["valid.info"])
        self.assertEqual(len([c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]]), 1)
        with self.assertRaises(converter.auth_oauth.AuthTokenTypeError):  # Direct use fails fast.
            converter.CredentialManager(envelope).get_headers()

    def test_envelope_refresh_token_alone_is_rejected(self):
        path = self.credential("rt.info", uid="rt", age=0)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auth"]["refreshToken"] = {"$wbEncrypted": 1, "envelope": "c3lu"}
        path.write_text(json.dumps(data), encoding="utf-8")
        pool = converter.CredentialPool([path])
        self.assertEqual(pool.entries(), [])
        rejected = [c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]]
        self.assertEqual(len(rejected), 1)
        self.assertIn("refreshToken", rejected[0].args[0])


    def test_replaced_envelope_file_at_same_path_logs_again(self):
        envelope = self.root / "churn.info"
        def write_envelope(uid):
            envelope.write_text(json.dumps({"account": {"uid": uid}, "auth": {
                "accessToken": {"$wbEncrypted": 1, "envelope": "c3lu"},
                "domain": "www.workbuddy.ai", "expiresAt": (time.time() + 86400) * 1000}}),
                encoding="utf-8")
        def rejections():
            return len([c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]])
        write_envelope("churn")
        pool = converter.CredentialPool([envelope], scan=True)
        self.assertEqual(rejections(), 1)
        envelope.unlink()
        write_envelope("other-account")  # A different credential, no scan ran while missing.
        pool._rescan()
        self.assertEqual(pool.entries(), [])
        self.assertEqual(rejections(), 2)
        pool._rescan()  # Identical content on the next scan stays silent.
        self.assertEqual(rejections(), 2)
    def test_recreated_envelope_file_at_same_path_logs_again(self):
        envelope = self.root / "reuse.info"
        def write_envelope():
            envelope.write_text(json.dumps({"account": {"uid": "reuse"}, "auth": {
                "accessToken": {"$wbEncrypted": 1, "envelope": "c3lu"},
                "domain": "www.workbuddy.ai", "expiresAt": (time.time() + 86400) * 1000}}),
                encoding="utf-8")
        def rejections():
            return len([c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]])
        write_envelope()
        pool = converter.CredentialPool([envelope], scan=True)
        self.assertEqual(pool.entries(), [])
        self.assertEqual(rejections(), 1)
        envelope.unlink()
        pool._rescan()  # Prune drops the suppressed path once the file is gone.
        write_envelope()
        pool._rescan()  # A new invalid file at the same path must warn again.
        self.assertEqual(pool.entries(), [])
        self.assertEqual(rejections(), 2)

    def test_swapped_envelope_file_fails_fast_on_next_use(self):
        path = self.credential(age=0)
        pool = converter.CredentialPool([path])
        self.assertEqual(len(pool.entries()), 1)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auth"]["accessToken"] = {"$wbEncrypted": 1, "envelope": "c3lu"}
        path.write_text(json.dumps(data), encoding="utf-8")
        pool.reload([path], reset=True)  # A damaged file must not block other credentials.
        self.assertEqual(pool.entries(), [])  # The stale identity is evicted, not retained.
        with self.assertRaises(converter.auth_oauth.AuthTokenTypeError):  # Direct use fails fast.
            converter.CredentialManager(path).get_headers()
        pool.reload([path], reset=False)  # The next scan rejects it once like any new file.
        self.assertEqual(pool.entries(), [])
        rejected = [c for c in self.logs.call_args_list if "拒绝入池" in c.args[0]]
        self.assertEqual(len(rejected), 1)

    def test_rejected_explicit_path_recovers_when_repaired(self):
        path = self.credential("explicit.info", uid="explicit", age=0)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auth"]["accessToken"] = {"$wbEncrypted": 1, "envelope": "c3lu"}
        path.write_text(json.dumps(data), encoding="utf-8")
        pool = converter.CredentialPool([path])  # Explicit mode (scan=False).
        self.assertEqual(pool.entries(), [])
        self.credential("explicit.info", uid="explicit", age=0)  # Repair the same file.
        pool._rescan()
        self.assertEqual([Path(e["id"]).name for e in pool.entries()], ["explicit.info"])

    def test_evicted_explicit_path_recovers_when_repaired(self):
        path = self.credential("evicted.info", uid="evicted", age=0)
        pool = converter.CredentialPool([path])
        self.assertEqual(len(pool.entries()), 1)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auth"]["refreshToken"] = {"$wbEncrypted": 1, "envelope": "c3lu"}
        path.write_text(json.dumps(data), encoding="utf-8")
        pool.reload([path], reset=True)  # The envelope replacement evicts the entry.
        self.assertEqual(pool.entries(), [])
        self.credential("evicted.info", uid="evicted", age=0)  # Repair the same file.
        pool._rescan()
        self.assertEqual([Path(e["id"]).name for e in pool.entries()], ["evicted.info"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
