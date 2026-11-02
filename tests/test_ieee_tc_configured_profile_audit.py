"""Offline profile provenance uses the selected config, not an old campaign."""
import copy
import hashlib
import json

import pytest

from scripts.ieee_tc.audit_d703_3b_full import configured_profile_sources, configured_release_reasons


def fixture(tmp_path):
    context = {'backend_environment_sha256': 'backend',
               'resource_envelope_sha256': 'resources', 'input_contract_sha256': 'input'}
    manifest = dict(passed=True, development_only=True, context=context)
    config = {'resource_coordination': {}}
    paths = []
    for kind in ('service', 'preparation'):
        path = tmp_path / (kind + '.json')
        path.write_text('{}\n')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        config['resource_coordination']['ieee_' + kind + '_profile'] = dict(
            path=str(path), sha256=sha, context=copy.deepcopy(context), ewma_beta=0.5)
        manifest[kind + '_profile'] = dict(profile_sha256=sha, beta=0.5)
        paths.append(path)
    return config, manifest, paths


def test_explicit_config_and_manifest_sources(tmp_path):
    config, manifest, paths = fixture(tmp_path)
    target = tmp_path / 'manifest.json'
    target.write_text(json.dumps(manifest))
    assert configured_profile_sources(config, target) == tuple(paths)


@pytest.mark.parametrize('fault', ['missing', 'bytes', 'manifest_sha', 'context', 'beta', 'passed'])
def test_refuses_stale_or_inconsistent_profile_binding(tmp_path, fault):
    config, manifest, paths = fixture(tmp_path)
    if fault == 'bytes': paths[0].write_text('{"changed":true}\n')
    elif fault == 'manifest_sha': manifest['preparation_profile']['profile_sha256'] = 'old'
    elif fault == 'context': manifest['context']['resource_envelope_sha256'] = 'old'
    elif fault == 'beta': manifest['service_profile']['beta'] = 0.8
    elif fault == 'passed': manifest['passed'] = False
    target = tmp_path / 'manifest.json'
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        configured_profile_sources(config, None if fault == 'missing' else target)


@pytest.mark.parametrize('enabled,allow,expected', [(True, True, True), (False, True, False),
                                                 (True, False, False), ('true', True, False)])
def test_forecast_release_requires_the_recorded_policy(enabled, allow, expected):
    config = {'resource_coordination': {'ieee_predictive_scaling':
              dict(enabled=enabled, allow_forecast_scale_down=allow)}}
    reasons = configured_release_reasons(config)
    assert ('ieee_forecast_capacity_release' in reasons) is expected
    assert 'ieee_all_low_through_cooldown' in reasons
    assert configured_release_reasons(None) == {'ieee_all_low_through_cooldown'}
