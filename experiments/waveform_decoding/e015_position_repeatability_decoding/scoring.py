"""Training-only common-feature selection and position-specific LDA fitting."""
import warnings
import numpy as np
from sklearn.feature_selection import f_classif
from .design import calibration_and_halves,CODES
from .features import xdawn_features
from experiments.response_characterization.e014_all_electrode_repeatability.features import fit_spatial
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import model


def repeatability_scores(x,operators,ranking_indices,batch=5):
    """One signed median correlation per scalar feature across five classes.

    Only ranking-half data determine numerical tolerances. Calibration data do
    not enter the contrast estimates or their validity thresholds.
    """
    x=np.asarray(x);operators=np.asarray(operators)
    if operators.shape[1:3]!=(2,5) or operators.shape[-1]!=len(x):raise ValueError('Wrong contrast axes')
    scale=np.sqrt(np.mean(x[ranking_indices]**2,axis=0))
    tolerance=(100*np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny))**2
    correlations=[]
    for start in range(0,len(operators),batch):
        op=operators[start:start+batch]
        profiles=(op.reshape(-1,len(x))@x).reshape(len(op),2,5,x.shape[1])
        a,b=profiles[:,0],profiles[:,1]
        a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        aa=np.mean(a*a,axis=1);bb=np.mean(b*b,axis=1)
        valid=(aa>tolerance)&(bb>tolerance)
        correlations.append(np.divide(np.mean(a*b,axis=1),np.sqrt(aa*bb),
            out=np.full(aa.shape,np.nan),where=valid))
    values=np.concatenate(correlations)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(values,axis=0)
    valid_count=np.isfinite(values).sum(axis=0)
    scores=np.where((valid_count==len(operators))&np.isfinite(median),np.maximum(median,0)**2,np.nan)
    return median,scores,valid_count


def select_columns(scores,k=100):
    if k<1:raise ValueError('Feature budget must be positive')
    candidates=np.flatnonzero(np.isfinite(scores)&(scores>0))
    if not len(candidates):raise ValueError('No valid positive-scoring features')
    order=np.lexsort((candidates,-scores[candidates]))
    return candidates[order[:k]]


def fit_fold(sensor_train,sensor_test,epochs_train,epochs_test,times,meta_train,
             sensor_contract,seed,k=100,repeats=100):
    """Test labels are deliberately absent from the fitting API.

    A single position-specific xDAWN basis fits independent calibration data.
    Both feature selectors use only the remaining training events. Their LDA
    classifiers then fit all training events, including calibration, in that
    fixed basis. Held-out values enter transformation/prediction only.
    """
    if meta_train.position.nunique()!=1:raise ValueError('Do not pool presentation positions')
    cal,operators,designs,allocation=calibration_and_halves(meta_train,seed,repeats=repeats)
    y=meta_train.trigger.to_numpy(int)
    mask=(times>=.05)&(times<1.05)
    filters,_,classes=fit_spatial(epochs_train[cal][...,mask],y[cal])
    if not np.array_equal(classes,CODES):raise ValueError('Unexpected spatial class order')
    components=[f'{code}:component-{i+1}' for code in classes for i in range(2)]
    train_spatial,spatial_contract=xdawn_features(epochs_train,times,filters,components)
    test_spatial,_=xdawn_features(epochs_test,times,filters,components)
    train=np.column_stack([sensor_train,train_spatial]);test=np.column_stack([sensor_test,test_spatial])
    contract=sensor_contract+spatial_contract
    if train.shape[1]!=len(contract) or test.shape[1]!=len(contract):raise ValueError('Feature contract differs')
    ranking=np.setdiff1d(np.arange(len(train)),cal)
    median,r_scores,valid_count=repeatability_scores(train,operators,ranking)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);f_scores,_=f_classif(train[ranking],y[ranking])
    result={}
    for name,score in [('repeatability',r_scores),('anova',f_scores)]:
        selected=select_columns(score,k)
        estimator=model().fit(train[:,selected],y)
        if not np.array_equal(estimator.classes_,CODES):raise ValueError('Classifier omitted a class')
        result[name]=dict(predicted=estimator.predict(test[:,selected]),probabilities=estimator.predict_proba(test[:,selected]),
            selected=selected,selected_contract=[contract[i] for i in selected],estimator=estimator,
            selected_scores=score[selected],n_selected=len(selected),requested_features=k)
    return dict(models=result,median_r=median,repeatability_score=r_scores,valid_r_splits=valid_count,
                calibration_indices=cal,ranking_indices=ranking,spatial_filters=filters,
                allocation=allocation,ranking_design_audits=designs)
