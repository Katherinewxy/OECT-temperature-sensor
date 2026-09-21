"""Reproduce the historical oracle-zone evaluation, not autonomous inference.

No network access. Experimental data must be supplied separately.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BANDS = ['low_[-26,-6]', 'mid_(-6,76]', 'high_(76,106]']
GRID = {'model__C': [10., 100., 1000.],
        'model__gamma': ['scale', .003, .01, .03, .1],
        'model__epsilon': [.25, .5, 1., 2.]}


def band(t):
    return BANDS[0] if t <= -6 else BANDS[1] if t <= 76 else BANDS[2]


def ranked(df, features):
    x = df[features].apply(pd.to_numeric, errors='coerce')
    x = x.fillna(x.median(axis=0))
    rows = []
    for f in features:
        v = x[f].to_numpy(float)
        corr = np.nan if len(v) < 3 or np.nanstd(v) < 1e-15 else np.corrcoef(v, df.temperature)[0, 1]
        rows.append({'feature': f, 'abs_corr': abs(corr)})
    return pd.DataFrame(rows).sort_values('abs_corr', ascending=False).feature.tolist()


def fit(df, features):
    model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler()), ('model', SVR(kernel='rbf'))])
    cv = KFold(n_splits=min(5, max(2, len(df)//12)), shuffle=True, random_state=42)
    search = GridSearchCV(model, GRID, scoring='neg_mean_absolute_error', cv=cv, n_jobs=1)
    search.fit(df[features].to_numpy(float), df.temperature.to_numpy(float))
    return search


def metrics(y, pred):
    err = np.abs(y - pred)
    return dict(MAE=float(mean_absolute_error(y, pred)),
                RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                R2=float(r2_score(y, pred)), within_2C=float(np.mean(err <= 2)),
                max_abs_error=float(err.max()))


def exact_three_feature_shap(model, train_x, test_x):
    """Enumerate all coalitions, using the historical empirical background.

    This verifies the 3-feature explanation without a SHAP dependency; it is
    not a replacement implementation for the historical 30-feature SHAP run.
    """
    if train_x.shape[1] != 3:
        raise ValueError('This verification is restricted to three features')
    bg = train_x
    if len(bg) > 20:
        bg = bg[np.random.default_rng(42).choice(len(bg), size=20, replace=False)]
    values = []
    for x in test_x:
        v = {}
        for mask in range(8):
            mixed = bg.copy()
            for j in range(3):
                if mask & (1 << j):
                    mixed[:, j] = x[j]
            v[mask] = float(np.mean(model.predict(mixed)))
        phi = np.zeros(3)
        for j in range(3):
            for mask in range(8):
                if not mask & (1 << j):
                    size = bin(mask).count('1')
                    weight = math.factorial(size)*math.factorial(2-size)/6
                    phi[j] += weight*(v[mask | (1 << j)]-v[mask])
        if not np.isclose(phi.sum(), v[7]-v[0], atol=1e-8):
            raise AssertionError('SHAP additivity verification failed')
        values.append(phi)
    return np.asarray(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--features', type=Path)
    source.add_argument('--raw-dir', type=Path)
    p.add_argument('--feature-names', type=Path, default=Path(__file__).with_name('feature_names.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--all-variants', action='store_true')
    p.add_argument('--reference-predictions', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.raw_dir:
        from extract_features import collect_real_environment_records
        df = collect_real_environment_records(a.raw_dir)
        df.to_csv(a.out/'extracted_features.csv', index=False)
    else:
        df = pd.read_csv(a.features)
    features = (json.loads(a.feature_names.read_text()) if a.feature_names.suffix == '.json'
                else pd.read_csv(a.feature_names, header=None).iloc[:, 0].tolist())
    if len(features) != 30 or len(set(features)) != 30:
        raise ValueError('Expected 30 distinct feature names')
    train = df[df.repeat_type.eq('temperature_stage')].copy()
    test = df[df.repeat_type.eq('real_environment')].copy().reset_index(drop=True)
    if train.empty or test.empty or df.sample_id.duplicated().any():
        raise ValueError('Empty split or duplicate sample IDs')
    train['band'] = train.temperature.map(band)
    test['band'] = test.temperature.map(band)
    ranks = {z: ranked(train[train.band.eq(z)], features) for z in BANDS}
    global_rank = ranked(train, features)
    ks = [1,3,5,8,10,15,20,30] if a.all_variants else [3,30]
    predictions, summaries, parameters, explanations = [], [], {}, []
    ref = pd.read_csv(a.reference_predictions) if a.reference_predictions else None
    for kind in ['three_zone', 'global']:
        for k in ks:
            variant = ('all30' if k == 30 else f'top{k}')
            reference_variant = variant if kind == 'three_zone' or k == 30 else f'global_top{k}'
            reference_model = 'Three-zone SVR-RBF' if kind == 'three_zone' else 'SVR_RBF'
            pred = np.full(len(test), np.nan)
            params = {}
            for z in (BANDS if kind == 'three_zone' else ['global']):
                tr = train[train.band.eq(z)] if kind == 'three_zone' else train
                idx = test.index[test.band.eq(z)] if kind == 'three_zone' else test.index
                fs = features if k == 30 else (ranks[z] if kind == 'three_zone' else global_rank)[:k]
                model = fit(tr, fs)
                if len(idx):
                    pred[idx] = model.predict(test.loc[idx, fs].to_numpy(float))
                    if k == 3 and kind == 'three_zone':
                        phi = exact_three_feature_shap(model, tr[fs].to_numpy(float), test.loc[idx, fs].to_numpy(float))
                        explanations.extend({'zone': z, 'feature': f,
                            'mean_abs_shap': float(np.abs(phi[:, j]).mean())}
                            for j, f in enumerate(fs))
                params[z] = {'features': fs, 'best_params': model.best_params_, 'n_train': len(tr)}
            if not np.isfinite(pred).all():
                raise ValueError('Missing predictions')
            row = {'model': reference_model, 'variant': reference_variant,
                   'routing': 'true-temperature oracle' if kind == 'three_zone' else 'none',
                   **metrics(test.temperature.to_numpy(float), pred)}
            result = pd.DataFrame({'sample_id': test.sample_id, 'model': reference_model,
                                   'variant': reference_variant, 'y_true': test.temperature,
                                   'y_pred': pred})
            if ref is not None:
                old = ref[ref.model.eq(reference_model) & ref.variant.eq(reference_variant)]
                merged = result.merge(old[['sample_id','y_true','y_pred']], on='sample_id', suffixes=('', '_old'), validate='one_to_one')
                if len(merged) != len(test) or not np.allclose(merged.y_true, merged.y_true_old):
                    raise ValueError('Reference sample mismatch')
                row['max_prediction_difference'] = float(np.max(np.abs(merged.y_pred - merged.y_pred_old)))
                row['matches_reference_at_1e-8_C'] = bool(row['max_prediction_difference'] <= 1e-8)
            predictions.append(result)
            summaries.append(row)
            parameters[kind + '_' + str(k)] = params
            print(json.dumps(row), flush=True)
    pd.concat(predictions).to_csv(a.out/'predictions.csv', index=False)
    pd.DataFrame(summaries).to_csv(a.out/'metrics.csv', index=False)
    pd.DataFrame(explanations).to_csv(a.out/'top3_exact_shap.csv', index=False)
    (a.out/'parameters.json').write_text(json.dumps(parameters, indent=2))
    (a.out/'environment.json').write_text(json.dumps({'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
        'n_train': len(train), 'n_test': len(test)}, indent=2))


if __name__ == '__main__':
    main()
