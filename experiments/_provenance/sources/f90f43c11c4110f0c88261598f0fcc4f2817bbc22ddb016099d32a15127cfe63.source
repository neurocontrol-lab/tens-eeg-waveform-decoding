"""Descriptive intervals resampling independent participants, not repeated splits."""
import numpy as np
import pandas as pd


def median_intervals(values,repeats=10000,seed=20261023):
    values = np.asarray(values,float)
    if values.ndim!=2 or len(values)<3 or not np.isfinite(values).all():
        raise ValueError('Finite participant-by-summary values required')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values),size=(repeats,len(values)))
    boots = np.median(values[indices],axis=1)
    return np.median(values,axis=0),np.quantile(boots,[.025,.975],axis=0)


def summarize(args):
    rows,contrasts = [],[]
    for subject in args.subjects:
        data = pd.read_csv(args.prior/f'sub-{subject}/repeatability.csv')
        # Splits are summarized within participant before any bootstrap draw.
        d = data.groupby(['representation','position'])[['common_response_r','waveform_deviation_r']].median()
        for (representation,position),values in d.iterrows():
            for metric,value in values.items():
                rows.append(dict(subject=subject,representation=representation,position=position,metric=metric,value=value))
        for (representation,position),values in d.iterrows():
            if representation in ('voltage','slope25'):
                continue
            reference = 'slope25' if representation.startswith('deriv') else 'voltage'
            for metric,value in values.items():
                contrasts.append(dict(subject=subject,representation=representation,reference=reference,
                    position=position,metric=metric,difference=value-d.loc[(reference,position),metric]))
    frame = pd.DataFrame(rows)
    differences = pd.DataFrame(contrasts)
    frame.to_csv(args.output/'multiscale_participant_summaries.csv',index=False)
    differences.to_csv(args.output/'multiscale_participant_differences.csv',index=False)
    def intervals(frame,keys,value,filename):
        wide = frame.pivot(index='subject',columns=keys,values=value)
        if list(wide.index)!=sorted(args.subjects) or wide.isna().any().any():
            raise ValueError('Missing or duplicate participant summaries')
        centre,ci = median_intervals(wide.to_numpy())
        result = pd.DataFrame([dict(zip(keys,key if isinstance(key,tuple) else (key,)),
            n_subjects=len(wide),median=float(centre[i]),lower=float(ci[0,i]),upper=float(ci[1,i]),
            interpretation='Descriptive pointwise participant-bootstrap interval; not multiplicity adjusted')
            for i,key in enumerate(wide.columns)])
        result.to_csv(args.output/filename,index=False)
        return result
    intervals(frame,['representation','position','metric'],'value','multiscale_intervals.csv')
    intervals(differences,['representation','reference','position','metric'],'difference','multiscale_difference_intervals.csv')
    comparison = pd.read_csv(args.comparison/'paired_participant_contrasts.csv')
    comparison = comparison[comparison.subject.isin(args.subjects)]
    intervals(comparison,['contrast','representation','position'],'difference_r','preprocessing_difference_intervals.csv')
