"""Fixed-feature one-versus-rest LDA; no held-out labels in the fitting API."""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
from experiments.waveform_decoding.e015_position_repeatability_decoding.features import WIDTHS,contract_rows

CODES=np.arange(239,244)
WAVES={239:'Sine',240:'Square',241:'Triangle',242:'LinDec',243:'LinInc'}


def cached_spatial_features(sensor,contract,filters,classes):
    """Reapply the saved spatial basis to cached voltage means, without refit."""
    table=pd.DataFrame(contract);blocks=[];spatial_contract=[]
    names=[f'{int(c)}:component-{i+1}' for c in classes for i in range(2)]
    if filters.shape[0]!=len(names):raise ValueError('Saved spatial basis and class slots differ')
    for width in WIDTHS:
        ix=np.flatnonzero(((table.family=='voltage_mean')&(table.width_ms==width)).to_numpy())
        part=table.iloc[ix];channels=part.channel.drop_duplicates().tolist()
        edges=part[part.channel==channels[0]][['start_ms','end_ms']].to_numpy()
        if len(channels)!=filters.shape[1] or len(ix)!=len(channels)*len(edges):raise ValueError('Spatial sensor axis mismatch')
        expected=[(ch,float(a),float(b)) for ch in channels for a,b in edges]
        actual=list(part[['channel','start_ms','end_ms']].itertuples(index=False,name=None))
        if actual!=expected:raise ValueError('Cached voltage ordering differs from the saved representation')
        bins=np.asarray(sensor[:,ix]).reshape(len(sensor),len(channels),len(edges))
        projected=np.einsum('kc,nct->nkt',filters,bins,optimize=True)
        blocks.append(projected.reshape(len(sensor),-1))
        spatial_contract+=contract_rows('xdawn',names,edges[:,0],edges[:,1],width,'calibration RMS units')
    return np.concatenate(blocks,axis=1),spatial_contract


def selected_matrix(sensor,contract,filters,classes,selected,selected_contract):
    spatial,spatial_contract=cached_spatial_features(sensor,contract,filters,classes)
    full_contract=contract+spatial_contract;selected=np.asarray(selected,int)
    if len(selected)!=len(selected_contract) or (selected<0).any() or (selected>=len(full_contract)).any():raise ValueError('Invalid saved selection indices')
    if [full_contract[i]['feature'] for i in selected]!=[r['feature'] for r in selected_contract]:raise ValueError('Saved feature IDs do not match reconstructed columns')
    full=np.column_stack([sensor,spatial]);result=full[:,selected]
    if not np.isfinite(result).all():raise ValueError('Nonfinite reconstructed features')
    return result


def fit_binary_targets(train,test,training_waveforms,targets=CODES):
    """Five balanced-prior detectors share one training-only scaler.

    Missing target/rest classes yield explicit constant predictions, never a
    fabricated LDA fit. Prediction columns follow `targets`, not learned labels.
    """
    train=np.asarray(train);test=np.asarray(test);training_waveforms=np.asarray(training_waveforms)
    if len(train)!=len(training_waveforms) or train.shape[1]!=test.shape[1]:raise ValueError('Training/test axes differ')
    if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('Nonfinite model values')
    scaler=StandardScaler().fit(train);a=scaler.transform(train);b=scaler.transform(test)
    fits={}
    for code in targets:
        code=int(code);y=(training_waveforms==code).astype(int);positive=int(y.sum());negative=len(y)-positive
        flags=[];estimator=None;messages=[]
        if positive==0:
            probability=np.zeros(len(test));flags.append('target_absent_from_training');kind='constant_negative'
        elif negative==0:
            probability=np.ones(len(test));flags.append('rest_absent_from_training');kind='constant_positive'
        else:
            estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.array([.5,.5]))
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always');estimator.fit(a,y)
            messages=sorted({str(w.message) for w in recorded})
            probability=estimator.predict_proba(b)[:,int(np.flatnonzero(estimator.classes_==1)[0])]
            kind='shrinkage_lda'
        if positive<5:flags.append('limited_positive_training_support')
        if negative<5:flags.append('limited_negative_training_support')
        if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('Invalid detector probabilities')
        fits[code]=dict(probability=probability,predicted=(probability>=.5).astype(int),
            estimator=estimator,training_positive=positive,training_negative=negative,kind=kind,flags=flags,warnings=messages)
    return scaler,fits


def binary_metrics(y,probability,predicted=None):
    y=np.asarray(y,int);probability=np.asarray(probability,float)
    if len(y)!=len(probability) or not len(y) or not set(y)<=set([0,1]):raise ValueError('Invalid binary target')
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('Invalid binary scores')
    pred=(probability>=.5).astype(int) if predicted is None else np.asarray(predicted,int)
    if pred.shape!=y.shape or not set(pred)<=set([0,1]):raise ValueError('Invalid binary predictions')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    positives=tp+fn;negatives=tn+fp;sensitivity=tp/positives if positives else None;specificity=tn/negatives if negatives else None
    both=bool(positives and negatives)
    return dict(n_events=len(y),n_positive=positives,n_negative=negatives,prevalence=positives/len(y),
        balanced_accuracy=.5*(sensitivity+specificity) if both else None,
        roc_auc=float(roc_auc_score(y,probability)) if both else None,
        average_precision=float(average_precision_score(y,probability)) if both else None,
        sensitivity=sensitivity,specificity=specificity,precision=tp/(tp+fp) if tp+fp else 0.,
        accuracy=(tp+tn)/len(y),always_rest_accuracy=negatives/len(y),
        confusion_matrix=[[tn,fp],[fn,tp]],threshold=.5,
        note='Binary balanced accuracy and AUC undefined in a single-class test fold' if not both else '')
