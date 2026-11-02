"""Metadata-only support audit; unsupported contrasts fail explicitly."""
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from sklearn.model_selection import LeaveOneGroupOut

CODES=np.arange(239,244)


def filter_guard(tolerance=1e-4):
    """Conservative effective support, not a claim of compact IIR support."""
    sos=butter(4,[.5,30],btype='bandpass',fs=1000,output='sos')
    pulse=np.zeros(120001);pulse[60000]=1.
    response=sosfiltfilt(sos,pulse)
    seconds=abs(np.arange(len(pulse))-60000)/1000
    total=abs(response).sum()
    guard=next(g for g in range(1,31) if abs(response[seconds>g]).sum()/total<=tolerance)
    # Derivative features ending before 1050 ms use real samples through 1061 ms.
    return dict(one_sided_seconds=guard,l1_tail_tolerance=tolerance,
        measured_l1_tail_fraction=float(abs(response[seconds>guard]).sum()/total),
        impulse_simulation_seconds=120.,raw_feature_support_s=[-.25,1.062],
        minimum_trigger_separation_s=2*guard+1.312,
        note='Finite effective-support tolerance; the IIR response is not exactly zero outside this guard.')


def outer_splits(meta,separation):
    first=meta[meta.position==1].reset_index(drop=True)
    if not meta.groupby('pair_id').size().eq(2).all():raise ValueError('Require complete pairs')
    rows=[];seen=np.zeros(len(meta),int)
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(first,groups=first.group)):
        test_pairs=set(first.iloc[te].pair_id);train_pairs=set(first.iloc[tr].pair_id)
        test=np.flatnonzero(meta.pair_id.isin(test_pairs).to_numpy())
        train=meta[meta.pair_id.isin(train_pairs)]
        bad={p for p,g in train.groupby('pair_id') if np.min(abs(g.elapsed_s.to_numpy()[:,None]-meta.iloc[test].elapsed_s.to_numpy()[None,:]))<=separation}
        train_pairs-=bad;train=np.flatnonzero(meta.pair_id.isin(train_pairs).to_numpy())
        if set(meta.iloc[train].group)&set(meta.iloc[test].group):raise AssertionError('Comparison leakage')
        seen[test]+=1
        rows.append(dict(fold=fold,train=train,test=test,purged_pairs=sorted(bad),
                         test_groups=sorted(set(meta.iloc[test].group))))
    if not np.all(seen==1):raise AssertionError('Missing held-out events')
    return rows


def position_operator(meta):
    """Five adjusted responses for ONE position; no redundant position terms."""
    if meta.position.nunique()!=1 or set(meta.trigger)!=set(CODES):raise ValueError('Need one position and all five waveform classes')
    y=meta.trigger.to_numpy(int)-239
    wave=np.eye(5)[y,:4];wave[y==4]=-1.
    contexts=pd.get_dummies(meta.comparison,dtype=float).to_numpy()
    elapsed=(meta.elapsed_s-meta.groupby('comparison').elapsed_s.transform('mean')).to_numpy()
    elapsed=elapsed/max(float(elapsed.std()),1e-12)
    d=np.column_stack([contexts,elapsed,wave]);rank=int(np.linalg.matrix_rank(d))
    audit=dict(n=len(meta),rank=rank,columns=d.shape[1],condition_number=float(np.linalg.cond(d)))
    if rank!=d.shape[1] or len(meta)<=rank:raise ValueError(f'Unidentifiable position-specific comparison/time model: {audit}')
    contrast=np.vstack([np.eye(4),-np.ones((1,4))])
    return contrast@np.linalg.pinv(d)[-4:],audit


def calibration_and_halves(meta,seed,repeats=100,calibration_attempts=30,split_attempts=100):
    """One fixed independent basis; preferred 2/class, 1/class only if necessary.

    All choices use training metadata only. Every scoring half must independently
    identify all five adjusted waveform effects. No EEG values enter this audit.
    """
    rng=np.random.default_rng(seed);classes=[np.flatnonzero(meta.trigger.to_numpy()==c) for c in CODES]
    for per_class in (2,1):
        if any(len(ix)<per_class+2 for ix in classes):continue
        for attempt in range(calibration_attempts):
            cal=np.sort(np.concatenate([rng.choice(ix,per_class,replace=False) for ix in classes]))
            remaining=np.setdiff1d(np.arange(len(meta)),cal)
            groups=[g.index.to_numpy() for _,g in meta.iloc[remaining].groupby(['comparison','trigger'])]
            pairs=[];audits=[]
            for repeat in range(repeats):
                found=False
                for _ in range(split_attempts):
                    a=[];b=[]
                    for group in groups:
                        order=rng.permutation(group);cut=len(order)//2
                        if len(order)%2 and rng.integers(2):cut+=1
                        a.extend(order[:cut]);b.extend(order[cut:])
                    a=np.sort(a);b=np.sort(b)
                    try:
                        oa,aa=position_operator(meta.iloc[a].reset_index(drop=True))
                        ob,ab=position_operator(meta.iloc[b].reset_index(drop=True))
                    except ValueError:continue
                    pa=np.zeros((5,len(meta)));pb=pa.copy();pa[:,a]=oa;pb[:,b]=ob
                    pairs.append((pa,pb));audits.extend([dict(repeat=repeat,half='A',**aa),dict(repeat=repeat,half='B',**ab)])
                    found=True;break
                if not found:break
            if len(pairs)==repeats:
                return cal,np.stack(pairs),audits,dict(calibration_per_class=per_class,
                    calibration_attempt=attempt,ranking_events=len(remaining),seed=seed)
    raise ValueError('No independent five-class calibration/A/B allocation with identifiable adjusted effects')
