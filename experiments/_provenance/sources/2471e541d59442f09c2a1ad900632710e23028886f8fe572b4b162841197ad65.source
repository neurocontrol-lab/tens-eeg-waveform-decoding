"""Audited-loader adaptation of the supplied August28 AutoEOG33 pipeline.

Preserves its filtering, ICA, automatic EOG calibration and interpolation order.
Not a cross-validated preprocessing transform: intended for response diagnostics.
"""
import time
import numpy as np


SETTINGS = dict(highpass_hz=0.5, highpass_order=94, highpass_stopband_db=40,
                notch_hz=60., ica_highpass_hz=1., ica_components=33,
                ica_method='extended_infomax', ica_seed=42,
                eog_channel='Fp1', eog_z_threshold=3.,
                eog_target_blinks_per_minute=[10., 20.],
                blank_ms=[-2., 10.], lowpass_hz=190., muscle_rejection=False)


def interpolate_stimuli(data, samples, sfreq):
    """Match legacy half-open interpolation, including its endpoint convention."""
    for sample in samples:
        lo = max(1, int(sample) - int(round(.002 * sfreq)))
        hi = min(data.shape[1] - 1, int(sample) + int(round(.010 * sfreq)))
        if hi - lo >= 2:
            weights = np.linspace(0., 1., hi - lo)
            data[:, lo:hi] = (data[:, lo-1, None] * (1-weights)
                             + data[:, hi, None] * weights)


def supplied_preprocess(data, names, sfreq, samples, out):
    import mne
    mne.set_log_level('WARNING')
    started = time.perf_counter()
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, 'eeg'), verbose=False)
    raw.set_montage('standard_1020', on_missing='raise', verbose=False)
    raw.filter(.5, None, method='iir',
               iir_params=dict(order=94, ftype='cheby2', rs=40, output='sos'),
               phase='zero', verbose=False)
    raw.notch_filter(np.arange(60., sfreq/2, 60.), verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    hp = raw.copy().filter(1., None, fir_design='firwin', verbose=False)
    print('  Fitting supplied 33-component extended Infomax ICA', flush=True)
    ica = mne.preprocessing.ICA(n_components=33, method='infomax',
                              fit_params=dict(extended=True), random_state=42,
                              max_iter='auto')
    ica.fit(hp, verbose=False)
    ica.save(out/'decomposition-ica.fif', overwrite=False, verbose=False)
    std = hp.get_data(picks=['Fp1'])[0].std()
    duration_min = hp.times[-1]/60
    candidates = []
    for multiplier in np.geomspace(.02, 5., 60):
        n = len(mne.preprocessing.find_eog_events(
            hp, ch_name='Fp1', thresh=multiplier*std, verbose=False))
        rate = n/duration_min
        candidates.append((float(multiplier), float(rate)))
    in_band = [mr for mr in candidates if 10 <= mr[1] <= 20]
    multiplier, rate = min(in_band or candidates, key=lambda mr: abs(mr[1]-15))
    eog_epochs = mne.preprocessing.create_eog_epochs(
        hp, ch_name='Fp1', thresh=multiplier*std, verbose=False)
    if not len(eog_epochs):
        raise ValueError('No EOG calibration epochs; do not silently skip ICA rejection')
    excluded, _ = ica.find_bads_eog(eog_epochs, ch_name='Fp1', threshold=3., verbose=False)
    ica.exclude = sorted(int(c) for c in excluded)
    ica.save(out/'cleaning-ica.fif', overwrite=False, verbose=False)
    ica.apply(raw, verbose=False)
    del hp
    interpolate_stimuli(raw._data, samples, sfreq)
    raw.filter(None, 190., verbose=False)
    cleaned = raw.get_data()
    if not np.isfinite(cleaned).all():
        raise ValueError('Preprocessing produced nonfinite EEG')
    audit = dict(settings=SETTINGS, seconds=time.perf_counter()-started,
                 ica_iterations=int(ica.n_iter_), ica_n_components=int(ica.n_components_),
                 ica_excluded=ica.exclude, eog_multiplier=multiplier,
                 eog_rate_per_minute=rate, eog_calibration_grid=candidates,
                 adaptations=['strict original-onset event parser and CSV alignment',
                              'verified channel units; remove CREF/M1/M2 before filtering',
                              'no old ICA/calibration cache; fit and calibrate from scratch',
                              'explicit SOS high-pass representation',
                              'epoch/baseline/QC defined in diagnostics rather than legacy epochs'],
                 limitations=['zero-phase filtering before interpolation can spread stimulation artifacts',
                              'ICA fitted on full participant recording for descriptive analysis',
                              'no automatic bad-channel interpolation or muscle rejection'])
    print(f'  ICA removed {ica.exclude}; processing {audit["seconds"]:.1f}s', flush=True)
    return cleaned, audit
