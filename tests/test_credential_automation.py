"""Travel preferences and removal of daily check-in use isolated accounts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from unittest.mock import patch
import converter
from app import credits
from app.control_store import ControlStore
from tests import test_credential_actions as fixtures


class AutomationTests(unittest.TestCase):
    add_account = fixtures.CredentialActionTests.add_account
    configure = fixtures.CredentialActionTests.configure
    handle_upstream = fixtures.CredentialActionTests.handle_upstream
    setUp = fixtures.CredentialActionTests.setUp

    def preference(self, entry, field, enabled):
        response = self.client.patch('/admin/credentials/' + entry['account_key'], json={field: enabled})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIs(response.json()[field], enabled)

    def periodic(self, entry):
        with patch.object(credits, 'fetch_credits', return_value={'credits': 10, 'intl': entry['profile'].startswith('intl')}):
            return converter._sync_credits(self.pool, self.ledger, entry, automatic=True, failed=set())

    def test_defaults_offer_domestic_travel_and_no_checkin(self):
        for row in self.client.get('/admin/credentials').json()['credentials']:
            domestic = row['profile'].startswith('cn-')
            self.assertIs(row['auto_travel'], domestic)
            self.assertIs(row['travel_supported'], domestic)
            self.assertNotIn('auto_checkin', row)
            self.assertNotIn('checkin', row)

    def test_preferences_persist_without_claiming_or_enabling_account(self):
        self.preference(self.entry, 'enabled', False)
        self.preference(self.entry, 'auto_travel', False)
        reopened = ControlStore(self.root / 'control.sqlite3')
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.snapshot()['credentials'][self.entry['account_key']],
                         {'enabled': False, 'auto_travel': False})
        self.travel_mock.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_removed_checkin_endpoints_and_settings_never_contact_upstream(self):
        for url in ('/admin/checkin', self.url + '/checkin'):
            self.assertEqual(self.client.post(url).status_code, 404)
        before = self.control.snapshot()['revision']
        self.assertEqual(self.client.patch(self.url, json={'auto_checkin': True}).status_code, 400)
        self.assertEqual(self.control.snapshot()['revision'], before)
        self.travel_mock.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_legacy_checkin_preference_stays_readable_but_inert(self):
        self.control._update(None, lambda state: state['credentials'].update({
            self.entry['account_key']: {'enabled': True, 'auto_checkin': True, 'auto_travel': False}}))
        reopened = ControlStore(self.root / 'control.sqlite3')
        self.addCleanup(reopened.close)
        with patch.dict(converter.CONFIG, control_store=reopened):
            self.assertIsNotNone(self.periodic(self.entry))
        self.travel_mock.assert_not_called()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.ledger.entry(self.entry['id'])['credits']['credits'], 10)

    def test_invalid_preferences_do_not_change_revision(self):
        before = self.control.snapshot()['revision']
        for data in ({}, {'auto_checkin': 1}, {'auto_travel': 'true'}, {'auto_checkin': None},
                     {'enabled': False, 'auto_travel': True}, {'bad': True}):
            with self.subTest(data=data):
                self.assertEqual(self.client.patch(self.url, json=data).status_code, 400)
        self.assertEqual(self.control.snapshot()['revision'], before)
        self.assertEqual(self.client.patch('/admin/credentials/unknown', json={'auto_travel': False}).status_code, 404)

    def test_periodic_travel_respects_switch_and_international_scope(self):
        self.assertIsNotNone(self.periodic(self.entry))
        self.travel_mock.assert_called_once()
        self.travel_mock.reset_mock()
        self.preference(self.entry, 'auto_travel', False)
        self.assertIsNotNone(self.periodic(self.entry))
        self.assertIsNotNone(self.periodic(self.entries['intl-work']))
        self.travel_mock.assert_not_called()
        response = self.client.patch('/admin/credentials/' + self.entries['intl-work']['account_key'], json={'auto_travel': True})
        self.assertEqual(response.status_code, 400)

    def test_sync_does_not_run_automations_and_travel_is_scoped(self):
        with patch.object(credits, 'fetch_credits', return_value={'credits': 10, 'intl': False}), \
             patch.object(credits, 'fetch_request_usage', return_value={'by_day': {}, 'total_credits': 0, 'requests': 0}):
            self.client.post(self.url + '/sync')
        self.travel_mock.assert_not_called()
        self.assertTrue(self.client.post(self.url + '/travel').json()['results'][0]['ok'])
        self.assertEqual(self.travel_mock.call_args.args[1], 'cn-cli')
        self.assertIn('travel', self.ledger.entry(self.entry['id']))
        self.travel_mock.reset_mock()
        for action in ('travel', 'travel-status'):
            response = self.client.post('/admin/credentials/' + self.entries['intl-work']['account_key'] + '/' + action)
            self.assertTrue(response.json()['results'][0]['skipped'])
        self.travel_mock.assert_not_called()

    def test_stale_automatic_travel_result_stops_balance_sync(self):
        def changed(*args, **kwargs):
            self.entry['cm'].invalidate()
            return {'ok': True}
        self.travel_mock.side_effect = changed
        with patch.object(credits, 'fetch_credits') as balance:
            failed = set()
            self.assertIsNone(converter._sync_credits(self.pool, self.ledger, self.entry, automatic=True, failed=failed))
            self.assertIn(self.entry['id'], failed)
            balance.assert_not_called()

    def test_cookie_preferences_require_csrf(self):
        self.client.headers.pop('authorization')
        self.assertEqual(self.client.patch(self.url, json={'auto_travel': False}).status_code, 401)
        response = self.client.post('/admin/session', json={'api_key': 'synthetic-management-key'}, headers={'Origin': 'https://testserver'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.patch(self.url, json={'auto_travel': False}).status_code, 403)
        self.client.headers.update({'Origin': 'https://testserver', 'X-CSRF-Token': response.json()['csrf_token']})
        self.preference(self.entry, 'auto_travel', False)


if __name__ == '__main__':
    unittest.main()
