import copy
import numpy as np
import pytest

pytestmark=pytest.mark.torch


def fixture_data(scripts):
    module=scripts('build_nested_forecasts')
    rng=np.random.default_rng(12)
    subjects=[f'NaF_KO_{i:02d}' for i in range(6)]
    lookup={(sid,f'Week {week}'):rng.normal(size=8).astype('float32') for sid in subjects for week in (12,15,18,20)}
    return module,subjects,lookup


def test_outer_future_and_labels_cannot_change_training_or_test_tokens(scripts):
    module,subjects,lookup=fixture_data(scripts)
    settings=dict(epochs=3,hidden=8,inner_folds=2)
    original,manifest=module.outer_forecasts(lookup,subjects,subjects[0],settings)
    changed=copy.deepcopy(lookup)
    for week in (15,18,20):changed[(subjects[0],f'Week {week}')]*=-100
    after,new_manifest=module.outer_forecasts(changed,subjects,subjects[0],settings)
    assert manifest==new_manifest
    for key in original:np.testing.assert_array_equal(original[key],after[key])
    # Positive control: permitted training data must actually affect the pipeline.
    changed=copy.deepcopy(lookup);changed[(subjects[1],'Week 15')]*=-100
    altered,_=module.outer_forecasts(changed,subjects,subjects[0],settings)
    assert not np.array_equal(original[f'{subjects[0]}_ts1.npy'],altered[f'{subjects[0]}_ts1.npy'])
    for query,fit in manifest['queries'].items():
        assert query not in manifest['fits'][fit]['fit_subjects']
        assert subjects[0] not in manifest['fits'][fit]['fit_subjects']


def test_manifest_cache_rejects_leakage_and_changed_tokens(scripts,tmp_path):
    from utils.research_io import verify_forecast_manifest,atomic_json
    module,subjects,lookup=fixture_data(scripts)
    settings=dict(module.DEFAULT_SETTINGS,epochs=2,hidden=8,inner_folds=2)
    manifest=module.export_outer(lookup,subjects,subjects[0],tmp_path,settings,'cpu')
    root=tmp_path/subjects[0]
    assert module.export_outer(lookup,subjects,subjects[0],tmp_path,settings,'cpu')==manifest
    damaged=copy.deepcopy(manifest)
    damaged['fits']['fit_0']['fit_subjects'].append(subjects[0]);atomic_json(root/'manifest.json',damaged)
    with pytest.raises(ValueError,match='leakage'):verify_forecast_manifest(root,subjects[0],subjects)
    atomic_json(root/'manifest.json',manifest)
    np.save(root/f'{subjects[0]}_ts1.npy',np.zeros(8))
    with pytest.raises(ValueError,match='changed'):verify_forecast_manifest(root,subjects[0],subjects)
