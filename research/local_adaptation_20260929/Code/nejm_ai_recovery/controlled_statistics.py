"""Paired development inference and probability calibration; no official test access."""
from pathlib import Path
import os
import hashlib, json, warnings
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score
from threshold_strategy import empirical_threshold, operating_counts, sha

ROOT = Path(os.environ.get('CXR_PROJECT_ROOT', str(Path(__file__).resolve().parents[1] / 'private_project')))
BASE = ROOT / 'results_nc_v1'
OUT = Path(os.environ.get('CXR_ANALYSIS_OUTPUT', str(Path(__file__).resolve().parents[1] / 'outputs/NEJM_AI_Recovery_20260926/Controlled_statistics')))
LAB = ['Cardiomegaly', 'Pleural_Effusion', 'Atelectasis', 'Consolidation']
VL = ['Cardiomegaly', 'Pleural effusion', 'Atelectasis', 'Consolidation']


def auc_structure(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert y.shape == p.shape and np.isin(y, [0, 1]).all()
    assert np.isfinite(p).all()
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order]) != 0) + 1]
    return order, y[order], starts


def weighted_auc_batch(struct, weights):
    order, y, starts = struct
    w = np.asarray(weights)[:, order]
    pos = np.add.reduceat(w*y, starts, axis=1)
    neg = np.add.reduceat(w*(1-y), starts, axis=1)
    numerator = np.sum(pos*(np.cumsum(neg, axis=1)-.5*neg), axis=1)
    denominator = pos.sum(axis=1)*neg.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(w), np.nan), where=denominator > 0)


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert y.shape == p.shape and np.isfinite(p).all()
    b = np.minimum((np.clip(p, 0, 1)*15).astype(int), 14)
    sy = np.bincount(b, weights=y, minlength=15)
    sp = np.bincount(b, weights=p, minlength=15)
    return {'Brier': float(np.mean((y-p)**2)), 'ECE': float(np.abs(sy-sp).sum()/len(y))}


def fit_map(y, p):
    y, p = np.asarray(y), np.asarray(p)
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
    details = {'fit_n': len(y), 'fit_positive': int(y.sum()), 'fit_negative': int(len(y)-y.sum())}
    if min(details['fit_positive'], details['fit_negative']) < 5:
        return None, details | {'fit_status': 'fewer_than_5_in_either_class'}
    fit = LogisticRegression(C=1., solver='lbfgs', max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit.fit(logit(np.clip(p, 1e-7, 1-1e-7)).reshape(-1, 1), y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return None, details | {'fit_status': 'optimizer_nonconvergence'}
    return fit, details | {'fit_status': 'fitted', 'map_intercept': float(fit.intercept_[0]),
                           'map_slope': float(fit.coef_[0, 0]), 'negative_map_slope': bool(fit.coef_[0, 0] < 0)}


def mapped(fit, p):
    return fit.predict_proba(logit(np.clip(p, 1e-7, 1-1e-7)).reshape(-1, 1))[:, 1]


def paired_gains(preds, ys, reg, repeats=2000):
    rows = []
    for (source, architecture, schedule), group in reg.groupby(['source', 'architecture', 'schedule']):
        for policy, y in ys.items():
            structures = {}
            for size in ['1000', 'all']:
                for seed in [1, 2, 3]:
                    run = group[group['size'].eq(size) & group.seed.eq(seed)].iloc[0].run_id
                    for j in range(4):
                        structures[size, seed, j] = auc_structure(y[:, j], preds[run][:, j])
            delta = np.zeros((repeats, 3))
            rng = np.random.default_rng(20260926)
            for start in range(0, repeats, 50):
                end = min(repeats, start+50)
                w = rng.multinomial(len(y), np.full(len(y), 1/len(y)), size=end-start)
                for si, seed in enumerate([1, 2, 3]):
                    pieces = [weighted_auc_batch(structures['all', seed, j], w)
                              - weighted_auc_batch(structures['1000', seed, j], w) for j in range(4)]
                    delta[start:end, si] = np.mean(pieces, axis=0)
            point = []
            for seed in [1, 2, 3]:
                values = [weighted_auc_batch(structures['all', seed, j], np.ones((1, len(y))))[0]
                          - weighted_auc_batch(structures['1000', seed, j], np.ones((1, len(y))))[0]
                          for j in range(4)]
                point.append(float(np.mean(values)))
            draws = delta.mean(axis=1)
            valid = draws[np.isfinite(draws)]
            assert len(valid) >= .99*repeats
            row = {'source': source, 'architecture': architecture, 'schedule': schedule,
                   'reference_policy': policy, 'images': len(y), 'gain': float(np.mean(point)),
                   'between_seed_SD': float(np.std(point, ddof=1)),
                   'valid_bootstrap_replicates': len(valid), 'requested_bootstrap_replicates': repeats,
                   **{f'paired_seed_{i+1}_gain': x for i, x in enumerate(point)}}
            for label, tail in [('95', .025), ('97_5', .0125)]:
                row.update({f'bootstrap_{label}_lower': float(np.quantile(valid, tail)),
                            f'bootstrap_{label}_upper': float(np.quantile(valid, 1-tail))})
            rows.append(row)
            print('Paired bootstrap', source, architecture, schedule, policy, flush=True)
            pd.DataFrame(rows).to_csv(OUT/'Paired_scale_gains.csv', index=False)
    return rows


def calibration(preds, ys, ids, reg):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f'ldh-2026|vindr|{ids[i]}'.encode()).hexdigest())
    cal, assess = np.array(order[:3000]), np.array(order[3000:])
    assert len(assess) == 12000 and not set(cal) & set(assess)
    subsets = {(rep, n): np.random.default_rng(71000+rep).permutation(cal)[:n]
               for rep in range(20) for n in [250, 500, 1000, 2000]}
    rows, threshold_rows, source_records, input_records = [], [], [], []
    for r in reg[reg.schedule.eq('matched')].to_dict('records'):
        path = ROOT/'results_nejm_upgrade_20260926/source_validation_private'/(r['run_id']+'.npz')
        source = pd.read_csv(r['validation'], dtype={'image_id': str})
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['image_id'], source.image_id.to_numpy())
            ps = z['probabilities']
        input_records.append({'path': str(path), 'sha256': sha(path),
                              'manifest': r['validation'], 'manifest_sha256': sha(Path(r['validation']))})
        p = preds[r['run_id']]
        for j, finding in enumerate(LAB):
            sf, sd = fit_map(source['y_'+finding].to_numpy(), ps[:, j])
            source_records.append({'run_id': r['run_id'], 'finding': finding, **sd})
            st = empirical_threshold(ps[:, j][source['y_'+finding].eq(1).to_numpy()])
            assert st is not None
            sprob = mapped(sf, p[assess, j]) if sf is not None else None
            for policy, y in ys.items():
                yt, pt = y[assess, j], p[assess, j]
                raw = probability_metrics(yt, pt)
                sm = probability_metrics(yt, sprob) if sf is not None else None
                common = {'run_id': r['run_id'], 'source': r['source'], 'training_size': r['size'],
                          'model_seed': r['seed'], 'finding': finding, 'reference_policy': policy,
                          'assessment_images': len(assess), 'assessment_positive': int(yt.sum())}
                threshold_rows.append(common | {'threshold_source': 'observed_source_validation_labels',
                                                 'threshold': st, **operating_counts(np.sort(pt[yt == 1]), np.sort(pt[yt == 0]), st)})
                for (rep, n), ix in subsets.items():
                    yc, pc = y[ix, j], p[ix, j]
                    pi = float(yc.mean())
                    base = common | {'sampling_seed': rep, 'calibration_images': n,
                                     'calibration_positive': int(yc.sum()), 'calibration_negative': int(n-yc.sum())}
                    constant = probability_metrics(yt, np.full(len(yt), pi))
                    # Repeated raw/source values preserve paired comparisons, not independent replication.
                    rows.append(base | {'method': 'raw', 'fit_status': 'fit_not_required', **raw,
                                        'Brier_skill_vs_constant': 1-raw['Brier']/constant['Brier']})
                    rows.append(base | {'method': 'target_constant_prevalence', 'fit_status': 'fitted',
                                        **constant, 'Brier_skill_vs_constant': 0.})
                    rows.append(base | {'method': 'source_logistic', **sd,
                                        **(sm | {'Brier_skill_vs_constant': 1-sm['Brier']/constant['Brier']} if sm else {})})
                    tf, td = fit_map(yc, pc)
                    tm = probability_metrics(yt, mapped(tf, pt)) if tf is not None else None
                    rows.append(base | {'method': 'target_logistic', **td,
                                        **(tm | {'Brier_skill_vs_constant': 1-tm['Brier']/constant['Brier']} if tm else {})})
        print('Probability calibration', r['run_id'], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'Probability_calibration_all_attempts.csv', index=False)
    groups = ['source', 'training_size', 'finding', 'reference_policy', 'calibration_images', 'method']
    summary = []
    for key, g in frame.groupby(groups, sort=False):
        ok = g[g.Brier.notna()]
        row = dict(zip(groups, key)) | {'requested_runs': len(g), 'successful_runs': len(ok),
                                       'failed_runs': len(g)-len(ok),
                                       'negative_map_slopes': int(g.negative_map_slope.eq(True).sum())}
        for metric in ['Brier', 'ECE', 'Brier_skill_vs_constant']:
            row[metric+'_mean_conditional_on_fit'] = float(ok[metric].mean()) if len(ok) else None
            row[metric+'_p10_conditional_on_fit'] = float(ok[metric].quantile(.1)) if len(ok) else None
            row[metric+'_p90_conditional_on_fit'] = float(ok[metric].quantile(.9)) if len(ok) else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'Probability_calibration_summary.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT/'Source_threshold_transfer.csv', index=False)
    pd.DataFrame(source_records).to_csv(OUT/'Source_calibration_fits.csv', index=False)
    return {'probability_attempts': len(frame), 'failed_probability_attempts': int(frame.Brier.isna().sum()),
            'source_prediction_inputs': input_records}


def main():
    status = json.loads((BASE/'controlled_development/status.json').read_text())
    assert status['stage'] == 'complete' and status['models_assessed'] == 66
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name('controlled_statistics_protocol.json')
    frozen = OUT/protocol_path.name
    if frozen.exists():
        assert frozen.read_bytes() == protocol_path.read_bytes()
    else:
        frozen.write_bytes(protocol_path.read_bytes())
    protocol = json.loads(protocol_path.read_text())
    regs = [pd.read_csv(ROOT/root/'protocol/main_run_registry.csv', dtype={'size': str})
            for root in ['results_nc_v1', 'results_nc_controls_v1']]
    reg = pd.concat(regs, ignore_index=True)
    assert len(reg) == 66 and reg.run_id.is_unique
    label_path = Path(os.environ.get('CXR_VINDR_LABELS', str(Path(__file__).resolve().parents[1] / 'private_inputs/image_labels_train.csv')))
    readers = pd.read_csv(label_path, dtype={'image_id': str, 'rad_id': str})
    assert len(readers) == 45000 and not readers.duplicated(['image_id', 'rad_id']).any()
    assert readers.groupby('image_id').size().eq(3).all()
    votes = readers.groupby('image_id')[VL].sum().sort_index()
    ids = votes.index.to_numpy()
    testids = {p.stem for p in Path(os.environ.get('CXR_VINDR_TEST_IMAGES', str(Path(__file__).resolve().parents[1] / 'private_inputs/test_images'))).glob('*.png')}
    assert len(ids) == 15000 and len(testids) == 3000 and not set(ids) & testids
    ys = {p: (votes.to_numpy() >= k).astype(int) for p, k in protocol['reference_policies'].items()}
    preds, inputs, metrics = {}, [], []
    for r in reg.to_dict('records'):
        path = BASE/'controlled_development/predictions_private'/(r['run_id']+'.npz')
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['image_id'], ids)
            p = z['probabilities']
        assert p.shape == (15000, 4) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        preds[r['run_id']] = p
        inputs.append({'path': str(path), 'sha256': sha(path)})
        for policy, y in ys.items():
            for j, finding in enumerate(LAB):
                metrics.append({k: r[k] for k in ['run_id', 'source', 'size', 'seed', 'architecture', 'schedule']} |
                               {'reference_policy': policy, 'finding': finding, 'images': len(y),
                                'positive_labels': int(y[:, j].sum()), 'AUROC': float(roc_auc_score(y[:, j], p[:, j])),
                                'average_precision': float(average_precision_score(y[:, j], p[:, j])),
                                **probability_metrics(y[:, j], p[:, j])})
    pd.DataFrame(metrics).to_csv(OUT/'Same_image_reference_metrics.csv', index=False)
    gains = paired_gains(preds, ys, reg, protocol['bootstrap']['replicates'])
    cal = calibration(preds, ys, ids, reg)
    summary = {'stage': 'complete_development_statistics', 'models': len(preds),
               'official_test_accessed': False, 'prediction_inputs': inputs,
               'protocol_sha256': sha(protocol_path), 'script_sha256': sha(Path(__file__)),
               'reader_labels_sha256': sha(label_path), 'scale_contrasts': len(gains), **cal,
               'completed_utc': pd.Timestamp.now(tz='UTC').isoformat(),
               'interval_scope': 'Image sampling conditional on three fitted seeds; seed SD separate. Development evidence is exploratory.',
               'calibration_scope': 'Map slopes are fitted transformation coefficients, not external calibration slopes; overlapping resample quantiles are descriptive, not confidence intervals.'}
    (OUT/'Analysis_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
