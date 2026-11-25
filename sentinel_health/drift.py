"""Univariate train-to-holdout drift metrics for numeric and categorical features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

def compare(reference:pd.DataFrame,current:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for col in reference.columns:
        if col not in current:continue
        a=reference[col].dropna();b=current[col].dropna()
        if pd.api.types.is_numeric_dtype(a):
            av=a.to_numpy(float);bv=b.to_numpy(float)
            edges=np.unique(np.quantile(av,np.linspace(0,1,11)))
            if len(edges)<3:psi=js=0.
            else:
                edges[0],edges[-1]=-np.inf,np.inf
                pa=np.histogram(av,bins=edges)[0].astype(float);pb=np.histogram(bv,bins=edges)[0].astype(float)
                pa=np.clip(pa/max(pa.sum(),1),1e-8,None);pb=np.clip(pb/max(pb.sum(),1),1e-8,None);pa/=pa.sum();pb/=pb.sum()
                psi=float(np.sum((pb-pa)*np.log(pb/pa)));js=float(jensenshannon(pa,pb,base=2))
            ks=ks_2samp(av,bv,method="auto");ks_stat=float(ks.statistic);ks_p=float(ks.pvalue)
            kind="numeric"
        else:
            ac=a.astype(str).value_counts(normalize=True);bc=b.astype(str).value_counts(normalize=True)
            keys=ac.index.union(bc.index);pa=ac.reindex(keys,fill_value=0).to_numpy(float);pb=bc.reindex(keys,fill_value=0).to_numpy(float)
            pa=np.clip(pa,1e-8,None);pb=np.clip(pb,1e-8,None);pa/=pa.sum();pb/=pb.sum()
            psi=float(np.sum((pb-pa)*np.log(pb/pa)));js=float(jensenshannon(pa,pb,base=2));ks_stat=ks_p=float("nan");kind="categorical"
        rows.append({"feature":col,"type":kind,"psi":psi,"ks_statistic":ks_stat,"ks_p_value":ks_p,"js_distance":js,"reference_rows":int(len(a)),"current_rows":int(len(b))})
    return pd.DataFrame(rows).sort_values("psi",ascending=False)
