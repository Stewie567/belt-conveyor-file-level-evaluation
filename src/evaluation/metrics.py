import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
def metrics(y,p):
    pred=np.asarray(p).argmax(1);pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_precision=float(pr.mean()),macro_recall=float(re.mean()),macro_f1=float(f.mean()),damage_recall=float(re[5]),damage_f1=float(f[5]),per_class=[dict(label=i,precision=float(pr[i]),recall=float(re[i]),f1=float(f[i]),support=int(s[i])) for i in range(6)],confusion_matrix=confusion_matrix(y,pred,labels=range(6)).tolist())
def aggregate(m,p):
    t=m[['recording_id','condition','speed_rpm']].copy()
    for i in range(6):t[f'p{i}']=p[:,i]
    assert t.groupby('recording_id').condition.nunique().max()==1
    return t.groupby('recording_id',as_index=False).agg(dict(condition='first',speed_rpm='first',**{f'p{i}':'mean' for i in range(6)}))
