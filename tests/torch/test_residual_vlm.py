import copy
import json

import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def entrypoint_import_path(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / 'scripts'))


def specs(module):
    original = dict(config=dict(data=dict(nested_forecast_root='/original', img_tokens=4, data_seed=0),
                                 train=dict(learning_rate=.0002, num_train_epochs=20)),
                    source_sha256={'model.py': 'unchanged'}, nested_index_sha256='old',
                    backbone_revision='same', environment={'torch': 'same'})
    current = copy.deepcopy(original)
    current['config']['data']['nested_forecast_root'] = '/residual'
    current['nested_index_sha256'] = 'new'
    current['residual_forecast_protocol_sha256'] = 'frozen'
    current['source_sha256'].update({'scripts/' + name: 'frozen' for name in module.ADDED_SOURCES})
    return original, current


def test_only_forecast_routing_may_differ_from_matched_vlm(scripts):
    module = scripts('residual_vlm_protocol')
    original, current = specs(module)
    module.assert_matched(current, original)
    for section, key, value in [('train', 'learning_rate', .01), ('train', 'num_train_epochs', 100),
                                ('data', 'data_seed', 1), ('data', 'img_tokens', 1)]:
        altered = copy.deepcopy(current); altered['config'][section][key] = value
        with pytest.raises(ValueError, match='beyond'):
            module.assert_matched(altered, original)
    altered = copy.deepcopy(current); altered['source_sha256']['model.py'] = 'changed'
    with pytest.raises(ValueError, match='beyond'):
        module.assert_matched(altered, original)


def test_wrong_forecast_root_and_population_are_rejected_before_loading(scripts, tmp_path, monkeypatch):
    module = scripts('residual_vlm_protocol')
    monkeypatch.setattr(module, 'FORECAST_REPORT', tmp_path)
    (tmp_path / 'PROTOCOL.json').write_text(json.dumps(dict(forecast_output='/correct', population={'subjects': ['mouse']})))
    with pytest.raises(ValueError, match='different experiment or split'):
        module.verify_residual_inputs('/wrong', {}, {'subjects': ['mouse']}, [])
    with pytest.raises(ValueError, match='different experiment or split'):
        module.verify_residual_inputs('/correct', {}, {'subjects': ['different']}, [])
