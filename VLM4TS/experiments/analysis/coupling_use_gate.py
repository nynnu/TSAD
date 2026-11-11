"""Use existing coupling only when train-backed comparison is trustworthy.

This module supplies diagnostics; pipeline_final can opt in to Stage2 context.
It has no Stage1/aggregation vote. Uncertain is never silently rejected.
"""
import numpy as np
from replicate_channel_coupling import window_features,lower_p

STATUS={0:'unavailable',1:'uncertain_train',2:'uncertain_state',3:'uncertain_reference',4:'no_material_coupling_evidence',5:'coupling_change_evidence'}


def upper_support(reference,values):
    r=np.sort(np.asarray(reference)[np.isfinite(reference)]);v=np.asarray(values)
    out=np.full(v.shape,np.nan)
    if len(r)<20:return out
    valid=np.isfinite(v);out[valid]=(len(r)-np.searchsorted(r,v[valid],side='left')+1)/(len(r)+1)
    return out


class CouplingUseGate:
    def __init__(self,train,peers,sign,eligible,audit_passed,width):
        self.train=train;self.peers=peers;self.sign=sign;self.eligible=np.asarray(eligible,bool)
        self.audit_passed=np.asarray(audit_passed,bool);self.width=width
        self.fit_end=3*len(train)//10;self.bank_end=len(train)//2;self.cal_end=3*len(train)//4
        self.scale=np.maximum(train[:self.fit_end].std(axis=0),1e-8)
        self.bank_starts=np.arange(width,self.bank_end-width+1,width)
        self.bank=window_features(train,self.bank_starts,width,peers,sign)
        self.cal_starts=np.arange(self.bank_end,self.cal_end-width+1,width)
        self.cal=window_features(train,self.cal_starts,width,peers,sign)
        self.audit_starts=np.arange(self.cal_end,len(train)-width+1,width)
        self.audit=window_features(train,self.audit_starts,width,peers,sign)
        self.parameters=[]
        for c in range(train.shape[1]):
            if not self.eligible[c]:self.parameters.append(None);continue
            bd=self.context_features(train,self.bank_starts,c)
            valid=np.isfinite(self.bank['coupling'][:,c])&np.isfinite(bd).all(axis=1)
            if valid.sum()<4:self.parameters.append(None);continue
            param=dict(bank_context=bd[valid],bank_rho=self.bank['coupling'][valid,c],bank_starts=self.bank_starts[valid])
            dist,pred=self.reference(train,self.cal_starts,c,param)
            ref=self.cal['coupling'][:,c];ok=np.isfinite(dist)&np.isfinite(pred)&np.isfinite(ref)
            if ok.sum()<20:self.parameters.append(None);continue
            param.update(context_distances=dist[ok],margin=float(np.quantile(np.maximum(0.,pred[ok]-ref[ok]),.95)),
                         coupling_reference=ref[np.isfinite(ref)],cal_queries=int(ok.sum()))
            ad,ap=self.reference(train,self.audit_starts,c,param)
            target=self.audit['coupling'][:,c];valid_a=np.isfinite(target)&np.isfinite(ap)
            if valid_a.sum()>=10:
                mse=float(np.mean((target[valid_a]-ap[valid_a])**2))
                base=float(np.mean((target[valid_a]-np.median(param['coupling_reference']))**2))
                skill=1-mse/base if base>1e-15 else None
            else:mse=base=skill=None
            param.update(audit_prediction_mse=mse,audit_constant_mse=base,audit_prediction_skill=skill,
                         reference_useful=skill is not None and skill>0,audit_queries=int(valid_a.sum()))
            self.parameters.append(param)

    def context_features(self,values,starts,c):
        ids=[c,*self.peers[c].tolist()];out=np.full((len(starts),6),np.nan)
        for i,s in enumerate(starts):
            if s<self.width:continue
            x=values[s-self.width:s,ids]
            out[i]=np.r_[x.mean(axis=0)/self.scale[ids],x.std(axis=0)/self.scale[ids]]
        return out

    def reference(self,values,starts,c,param):
        contexts=self.context_features(values,starts,c);dist=np.full(len(starts),np.nan);pred=dist.copy()
        for i,context in enumerate(contexts):
            if not np.isfinite(context).all():continue
            distance=np.sqrt(np.mean((param['bank_context']-context)**2,axis=1))
            chosen=np.argsort(distance,kind='stable')[:4]
            dist[i]=float(distance[chosen[0]]);pred[i]=float(np.median(param['bank_rho'][chosen]))
        return dist,pred

    def decide(self,values,starts):
        f=window_features(values,starts,self.width,self.peers,self.sign);shape=f['coupling'].shape
        status=np.zeros(shape,dtype=np.int8);tail=np.full(shape,np.nan);support=tail.copy();deficit=tail.copy()
        arms={name:np.zeros(shape,bool) for name in ('tail_audit','normal_margin','context_supported','full_gate')}
        for c,param in enumerate(self.parameters):
            # Preserve baseline scorable coverage even if the new reference is unavailable.
            if self.eligible[c]:tail[:,c]=lower_p(self.cal['coupling'][:,c],f['coupling'][:,c])
            finite=np.isfinite(tail[:,c]);status[finite,c]=1
            if not self.audit_passed[c]:continue
            raw=finite&(tail[:,c]<=.05);arms['tail_audit'][:,c]=raw
            if param is None:
                status[finite,c]=3;continue
            distance,pred=self.reference(values,starts,c,param)
            support[:,c]=upper_support(param['context_distances'],distance)
            deficit[:,c]=pred-f['coupling'][:,c]
            material=np.isfinite(deficit[:,c])&(deficit[:,c]>param['margin'])
            state=np.isfinite(support[:,c])&(support[:,c]>.05)
            arms['normal_margin'][:,c]=raw&material
            arms['context_supported'][:,c]=raw&material&state
            status[finite,c]=2
            if not param['reference_useful']:
                status[finite&state,c]=3;continue
            supported=finite&state
            status[supported,c]=4;status[supported&raw&material,c]=5
            arms['full_gate'][:,c]=status[:,c]==5
        return dict(features=f,status=status,tail_p=tail,context_support_p=support,deficit=deficit,arms=arms)

    def metadata(self):
        rows=[]
        for c,p in enumerate(self.parameters):
            if p is None:rows.append(dict(channel=c,available=False));continue
            rows.append(dict(channel=c,available=True,cal_queries=p['cal_queries'],margin=p['margin'],
                audit_queries=p['audit_queries'],audit_prediction_skill=p['audit_prediction_skill'],reference_useful=p['reference_useful'],
                audit_prediction_mse=p['audit_prediction_mse'],audit_constant_mse=p['audit_constant_mse']))
        return rows

    def explain(self,result,index,channel):
        """Explicit evidence contract: a rare coupling change has no event veto.

        A normal-train gate passing is not a proof of physical abnormality.
        Consumers must not interpret the descriptive status as event validity.
        """
        c=channel;param=self.parameters[c]
        def number(value):return float(value) if np.isfinite(value) else None
        return dict(source='existing_signed_coupling',role='diagnostic_context',
            observed_coupling=number(result['features']['coupling'][index,c]),
            statistical_tail_p=number(result['tail_p'][index,c]),
            context_support_p=number(result['context_support_p'][index,c]),
            decrease_vs_reference=number(result['deficit'][index,c]),
            normal_train_variation_margin=param['margin'] if param else None,
            status=STATUS[int(result['status'][index,c])],
            passes_train_use_conditions=bool(result['arms']['full_gate'][index,c]),
            event_decision='undetermined',can_confirm_anomaly=False,can_reject_event=False,
            limitation='Train rarity and normal-variation exceedance do not establish a practically abnormal relationship.')
