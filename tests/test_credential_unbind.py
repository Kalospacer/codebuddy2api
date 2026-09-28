"""Deleting credentials that are still bound by model rules must auto-unbind on confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Allow direct execution.

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


from fastapi import HTTPException
from fastapi.testclient import TestClient

import converter
from app.control_store import ControlStore
from app.gateway_management import Management


class UnbindDeleteTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"CODEBUDDY_AUTH_DIR": str(root)}))
        self.enterContext(patch.dict(converter.CONFIG, {"log_path": None, "cred_pool": None, "cred": None}))
        self.path = root / "account.info"
        self.path.write_text(json.dumps({"account": {"uid": "synthetic"}, "auth": {
            "accessToken": "old-synthetic-token", "refreshToken": "synthetic-refresh",
            "domain": "www.codebuddy.cn", "expiresAt": (time.time() + 86400) * 1000,
            "lastRefreshTime": time.time() * 1000}}), encoding="utf-8")
        self.pool = converter.CredentialPool([self.path])
        self.store = ControlStore(root / "control.sqlite3")
        self.addCleanup(self.store.close)
        self.config = {"cred_pool": self.pool, "control_store": self.store,
                       "ledger": None, "trial_ledger": None}
        self.management = Management(SimpleNamespace(CONFIG=self.config))
        self.identity = self.pool.entries()[0]["account_key"]

    def bind(self):
        self.store.update_model("glm-4-flash", {"credential_ids": [self.identity]}, 0)

    def test_bound_delete_is_blocked_without_the_confirmation_flag(self):
        self.bind()
        with self.assertRaises(HTTPException) as ctx:
            self.management.admin_delete_guard("account.info")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail["models"], ["glm-4-flash"])

    def test_confirmed_delete_unbinds_only_after_removal_succeeds(self):
        self.store.update_model("glm-4-flash", {"public_id": "glm4", "credential_ids": [self.identity, "other"]}, 0)
        identity = self.management.admin_delete_guard("account.info", unbind=True)
        self.assertEqual(identity, self.identity)
        self.assertEqual(self.store.snapshot()["models"]["glm-4-flash"]["credential_ids"],
                         [self.identity, "other"])  # The guard itself must not unbind.
        self.assertTrue(self.pool.remove_file("account.info"))
        self.management.admin_unbind_credential(identity)
        rule = self.store.snapshot()["models"]["glm-4-flash"]
        self.assertEqual(rule["credential_ids"], ["other"])
        self.assertEqual(rule["public_id"], "glm4")
        self.assertFalse(self.path.exists())

    def test_route_keeps_bindings_when_removal_fails(self):
        self.bind()
        config = dict(self.config, api_key="secret", management=self.management)
        self.enterContext(patch.dict(converter.CONFIG, config))
        client = TestClient(converter.app)
        headers = {"X-Api-Key": "secret"}
        with patch.object(self.pool, "remove_file", return_value=False):
            response = client.delete("/admin/credentials/account.info?unbind=1", headers=headers)
            self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(self.store.snapshot()["models"]["glm-4-flash"]["credential_ids"],
                         [self.identity])  # A failed removal never widens routing to automatic.
        response = client.delete("/admin/credentials/account.info?unbind=1", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.snapshot()["models"]["glm-4-flash"]["credential_ids"], [])
        self.assertFalse(self.path.exists())

    def test_unbound_and_unknown_credentials_stay_on_the_fast_path(self):
        self.assertIsNone(self.management.admin_delete_guard("account.info"))  # No bindings.
        self.assertIsNone(self.management.admin_delete_guard("missing.info", unbind=True))  # Pool reports 404.
        self.assertEqual(self.store.snapshot()["revision"], 0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
