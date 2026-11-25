"""Leakage-aware CKD candidate comparison, calibration, evaluation, and artifacts."""
from __future__ import annotations
import json, time
from hashlib import sha256
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .ckd_data import DATA_PATH, EXPECTED_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET, load_dataset, make_features, profile_dataset
from .drift import compare as compare_drift

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS=ROOT/"artifacts"/"ckd"
SEED=42


class CKDModel:
    def __init__(self, estimator, threshold:float, model_name:str, model_version:str):
        self.estimator=estimator; self.threshold=float(threshold); self.model_name=model_name; self.model_version=model_version
        self.input_columns=FEATURE_COLUMNS
    def predict_proba(self, frame:pd.DataFrame)->np.ndarray:
        missing=sorted(set(self.input_columns)-set(frame.columns)); extra=sorted(set(frame.columns)-set(self.input_columns))
        if missing or extra: raise ValueError(f"CKD input schema mismatch; missing={missing}; extra={extra}")
        return self.estimator.predict_proba(make_features(frame))
    def predict(self,frame:pd.DataFrame)->np.ndarray:
        return (self.predict_proba(frame)[:,1]>=self.threshold).astype("int8")


def _preprocessor(X, scaler="standard", min_frequency=None, cat_impute="most_frequent", dense=False):
    numeric=X.select_dtypes(include=np.number).columns.tolist(); categorical=[c for c in X if c not in numeric]
    scale={"standard":StandardScaler(),"robust":RobustScaler(),"none":"passthrough"}[scaler]
    num=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",scale)])
    cat=Pipeline([("impute",SimpleImputer(strategy=cat_impute,fill_value="Unknown" if cat_impute=="constant" else None)),
        ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=min_frequency,sparse_output=not dense))])
    return ColumnTransformer([("numeric",num,numeric),("categorical",cat,categorical)],remainder="drop",sparse_threshold=0.0 if dense else 1.0)


def _pipeline(X,model,*,scaler="standard",freq=None,cat_impute="most_frequent",dense=False):
    return Pipeline([("preprocess",_preprocessor(X,scaler,freq,cat_impute,dense)),("model",model)])


def _candidates(X):
    specs=[]
    for c,scaler,weight in [(.05,"standard","balanced"),(.2,"standard",None),(.5,"robust","balanced"),(1.,"robust",None),(5.,"standard","balanced"),(10.,"none",None)]:
        specs.append((f"logistic_C{c}_{scaler}_{weight}",_pipeline(X,LogisticRegression(C=c,class_weight=weight,max_iter=2000,solver="liblinear",random_state=SEED),scaler=scaler,freq=2)))
    for kernel,c,gamma,weight in [("rbf",.5,"scale","balanced"),("rbf",2.,"scale",None),("rbf",5.,.03,"balanced"),("linear",.5,"scale",None),("linear",2.,"scale","balanced")]:
        specs.append((f"svc_{kernel}_C{c}_g{gamma}_{weight}",_pipeline(X,SVC(C=c,kernel=kernel,gamma=gamma,class_weight=weight,probability=False,random_state=SEED),scaler="standard",freq=2)))
    for k,weights,p in [(3,"uniform",2),(5,"distance",2),(9,"uniform",None),(15,"distance",None)]:
        specs.append((f"knn_k{k}_{weights}_p{p}",_pipeline(X,KNeighborsClassifier(n_neighbors=k,weights=weights,p=2 if p is None else p),scaler="robust",freq=2)))
    for depth,leaf,weight in [(4,2,"balanced"),(8,2,"balanced"),(None,2,"balanced"),(8,5,None),(None,5,None)]:
        specs.append((f"random_forest_d{depth}_leaf{leaf}_{weight}",_pipeline(X,RandomForestClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,class_weight=weight,max_features="sqrt",n_jobs=1,random_state=SEED),scaler="none",freq=2)))
    for depth,leaf,weight in [(4,2,"balanced"),(8,2,"balanced"),(None,2,"balanced"),(None,5,None)]:
        specs.append((f"extra_trees_d{depth}_leaf{leaf}_{weight}",_pipeline(X,ExtraTreesClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,class_weight=weight,max_features="sqrt",n_jobs=1,random_state=SEED),scaler="none",freq=2)))
    for leaves,rate,reg,iters in [(7,.05,1.,100),(15,.05,2.,150),(7,.1,5.,150),(15,.1,10.,200),(7,.08,0.,200),(15,.08,1.,100)]:
        specs.append((f"hist_gb_leaf{leaves}_lr{rate}_l2{reg}_n{iters}",_pipeline(X,HistGradientBoostingClassifier(max_leaf_nodes=leaves,learning_rate=rate,l2_regularization=reg,max_iter=iters,min_samples_leaf=8,random_state=SEED),scaler="none",freq=2,cat_impute="constant",dense=True)))
    if len(specs)!=30: raise AssertionError(f"Expected 30 CKD candidates, got {len(specs)}")
    return specs


def _metrics(y,prob,threshold):
    pred=(prob>=threshold).astype("int8")
    return {"average_precision":float(average_precision_score(y,prob)),"roc_auc":float(roc_auc_score(y,prob)),
        "accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),"brier_score":float(brier_score_loss(y,prob)),
        "alert_rate":float(pred.mean()),"confusion_matrix_tn_fp_fn_tp":confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()}


def _thresholds(y,prob,target_recall=.80):
    p,r,t=precision_recall_curve(y,prob)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    i=np.flatnonzero(f==np.nanmax(f))[-1]
    maxf={"threshold":float(t[i]),"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i]),"selection":"max validation F1; ties favor higher threshold"}
    eligible=np.flatnonzero(r[:-1]>=target_recall)
    j=eligible[-1] if len(eligible) else 0
    high={"threshold":float(t[j]),"precision":float(p[j]),"recall":float(r[j]),"f1":float(f[j]),"target_recall":target_recall,
        "selection":"highest validation threshold meeting recall target" if len(eligible) else "target unavailable; lowest threshold returned"}
    return maxf,high


def _plot_outputs(y,prob,threshold,out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));PrecisionRecallDisplay.from_predictions(y,prob,ax=ax);ax.set_title("CKD test precision-recall curve");fig.tight_layout();fig.savefig(out/"precision_recall.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));RocCurveDisplay.from_predictions(y,prob,ax=ax);ax.set_title("CKD test ROC curve");fig.tight_layout();fig.savefig(out/"roc_curve.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(y,prob>=threshold,display_labels=["notckd","ckd"],cmap="Blues",ax=ax,colorbar=False);ax.set_title("CKD test confusion matrix");fig.tight_layout();fig.savefig(out/"confusion_matrix.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));CalibrationDisplay.from_predictions(y,prob,n_bins=5,strategy="quantile",ax=ax);ax.set_title("CKD test calibration");fig.tight_layout();fig.savefig(out/"calibration.png",dpi=150);plt.close(fig)


def run_training(data_path:str|Path=DATA_PATH,artifact_dir:str|Path=ARTIFACTS,*,cv_folds:int=5)->dict[str,Any]:
    if cv_folds<3: raise ValueError("Use at least 3 stratified folds for this small dataset.")
    started=time.perf_counter();out=Path(artifact_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_dataset(data_path);profile=profile_dataset(df);X=make_features(df);y=df["target"].to_numpy()
    # Group identical predictor records so exact duplicate rows never cross a split or CV fold.
    dup_group=pd.util.hash_pandas_object(df[FEATURE_COLUMNS].astype("string"),index=False).astype(str).to_numpy()
    all_idx=np.arange(len(df));holdout=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    dev_local,test_local=next(holdout.split(X,y,dup_group));dev=all_idx[dev_local];test=all_idx[test_local]
    val_split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
    train_local,val_local=next(val_split.split(X.iloc[dev],y[dev],dup_group[dev]));train=dev[train_local];val=dev[val_local]
    # Assert patient-record groups remain partition-isolated.
    group_sets={k:set(dup_group[v]) for k,v in {"train":train,"validation":val,"test":test}.items()}
    overlaps={f"{a}__{b}":len(group_sets[a]&group_sets[b]) for i,a in enumerate(group_sets) for b in list(group_sets)[i+1:]}
    if any(overlaps.values()): raise RuntimeError(f"duplicate-record leakage: {overlaps}")
    Xtrain=X.iloc[train].reset_index(drop=True);ytrain=y[train];gtrain=dup_group[train]
    Xval=X.iloc[val];yval=y[val];Xtest=X.iloc[test];ytest=y[test]
    cv=StratifiedGroupKFold(n_splits=cv_folds,shuffle=True,random_state=SEED+2)
    candidates=_candidates(Xtrain);scores=[]
    for index,(name,estimator) in enumerate(candidates,1):
        print(f"[{index:02d}/30] {name}",flush=True)
        with threadpool_limits(limits=1):
            cvscore=cross_validate(estimator,Xtrain,ytrain,groups=gtrain,cv=cv,
                scoring={"ap":"average_precision","roc_auc":"roc_auc"},n_jobs=1,error_score="raise")
        scores.append({"candidate":name,"mean_cv_average_precision":float(cvscore["test_ap"].mean()),
            "std_cv_average_precision":float(cvscore["test_ap"].std()),"mean_cv_roc_auc":float(cvscore["test_roc_auc"].mean()),
            "std_cv_roc_auc":float(cvscore["test_roc_auc"].std()),
            "fold_average_precision":cvscore["test_ap"].tolist(),"fold_roc_auc":cvscore["test_roc_auc"].tolist()})
    scores.sort(key=lambda row:(row["mean_cv_roc_auc"],row["mean_cv_average_precision"]),reverse=True)
    chosen=scores[0]["candidate"];base=dict(candidates)[chosen]
    calibrated=CalibratedClassifierCV(estimator=base,method="sigmoid",cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+3))
    with threadpool_limits(limits=1): calibrated.fit(Xtrain,ytrain)
    pval=calibrated.predict_proba(Xval)[:,1];maxf,high=_thresholds(yval,pval,.80)
    ptest=calibrated.predict_proba(Xtest)[:,1]
    metrics=_metrics(ytest,ptest,maxf["threshold"]);high_metrics=_metrics(ytest,ptest,high["threshold"])
    model_version=sha256(Path(data_path).read_bytes()+chosen.encode()).hexdigest()[:20]
    bundle=CKDModel(calibrated,maxf["threshold"],chosen,model_version)
    model_path=out/"ckd_model.joblib";joblib.dump(bundle,model_path)
    predictions=pd.DataFrame({"actual":ytest,"probability":ptest,"prediction_max_f1":(ptest>=maxf["threshold"]).astype(int),"prediction_recall_80_policy":(ptest>=high["threshold"]).astype(int)})
    predictions.to_csv(out/"holdout_predictions.csv",index=False);_plot_outputs(ytest,ptest,maxf["threshold"],out)
    pd.DataFrame(scores).to_csv(out/"model_comparison.csv",index=False)
    with threadpool_limits(limits=1):
        importance=permutation_importance(calibrated,Xval,yval,scoring="average_precision",n_repeats=3,random_state=SEED,n_jobs=1)
    pd.DataFrame({"feature":Xval.columns,"importance":importance.importances_mean,"importance_std":importance.importances_std}).sort_values("importance",ascending=False).to_csv(out/"feature_importance.csv",index=False)
    pd.DataFrame([{"feature":c,"missing_count":int(df[c].isna().sum()),"missing_rate":float(df[c].isna().mean())} for c in EXPECTED_COLUMNS]).to_csv(out/"data_quality.csv",index=False)
    numeric=df[NUMERIC_COLUMNS];outliers=[]
    for col in NUMERIC_COLUMNS:
        q1,q3=numeric[col].quantile([.25,.75]);iqr=q3-q1;flag=(numeric[col]<q1-1.5*iqr)|(numeric[col]>q3+1.5*iqr)
        outliers.append({"feature":col,"iqr_flagged":int(flag.sum()),"rate":float(flag.mean())})
    pd.DataFrame(outliers).to_csv(out/"outlier_review.csv",index=False)
    compare_drift(X.iloc[train],X.iloc[test]).to_csv(out/"feature_drift.csv",index=False)
    latency=[]
    for i in range(len(Xtest)):
        row=df.iloc[[test[i]]][FEATURE_COLUMNS];tick=time.perf_counter();bundle.predict_proba(row);latency.append((time.perf_counter()-tick)*1000)
    latency_stats={f"p{q}_ms":float(np.percentile(latency,q)) for q in (50,95,99)}
    partitions={name:{"rows":int(len(ix)),"positive_count":int(y[ix].sum()),"positive_rate":float(y[ix].mean())} for name,ix in {"train":train,"validation":val,"test":test}.items()}
    report={"project":"Sentinel Health CKD classification","target":"class == ckd","positive_label":"ckd","negative_label":"notckd",
        "dataset_profile":profile,"split_strategy":"stratified group holdout and validation; exact duplicate feature rows isolated",
        "split":partitions,"duplicate_group_overlap":overlaps,"cv_folds":cv_folds,"cv_results":scores,"selected_model":chosen,
        "validation_policy":maxf,"validation_recall_80_policy":high,"test_metrics":metrics,"test_metrics_at_validation_recall_80_threshold":high_metrics,
        "dummy_baseline":{"positive_rate":float(ytest.mean()),"constant_score_average_precision":float(average_precision_score(ytest,np.full(len(ytest),ytest.mean()))),"always_majority_accuracy":float(np.mean(ytest==(1 if ytest.mean()>=.5 else 0)))},
        "offline_inference_latency":latency_stats,"duration_seconds":float(time.perf_counter()-started),"model_version":model_version,
        "model_artifact":model_path.name,"clinical_use":"Educational benchmark only; not a diagnostic or treatment tool."}
    (out/"training_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    (out/"model_registry.json").write_text(json.dumps({"active_version":model_version,"versions":[{"version":model_version,"model_name":chosen,"artifact":str(model_path.resolve().relative_to(ROOT)).replace("\\","/"),"threshold":maxf["threshold"],"status":"active"}]},indent=2),encoding="utf-8")
    (out/"model_card.md").write_text(f"# CKD model card\n\n- Task: binary classification of the supplied CKD dataset (`ckd` vs `notckd`).\n- Selected by {cv_folds}-fold duplicate-grouped stratified CV: {chosen}.\n- Holdout ROC-AUC: {metrics['roc_auc']:.4f}; average precision: {metrics['average_precision']:.4f}.\n- Small, single-source retrospective benchmark (n={len(df)}); no external validation.\n- Educational use only; not for diagnosis, screening, or care decisions.\n",encoding="utf-8")
    return {"model":bundle,"report":report,"comparison":pd.DataFrame(scores),"artifact_dir":out}
