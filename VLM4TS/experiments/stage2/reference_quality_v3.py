"""Train-only reference support diagnostics with disjoint normal validation.

Bank: first 50%; support calibration: next 25%; untouched audit: last 25%.
Context-support percentiles are descriptive under serial dependence. They do
not turn a retrieved 10-90% envelope into an anomaly decision boundary.
"""
import numpy as np


class ReferenceQuality:
    def __init__(self, train, period_estimator, calibration_queries=64, *, bank_fraction=.5, calibration_end_fraction=.75):
        self.train=np.asarray(train,float)
        if self.train.ndim!=2 or not self.train.shape[1] or not np.isfinite(self.train).all():
            raise ValueError('Expected a finite 2D normal-train matrix')
        if type(calibration_queries) is not int or calibration_queries<1:
            raise ValueError('calibration_queries must be a positive integer')
        if not 0<bank_fraction<calibration_end_fraction<1:
            raise ValueError('Expected 0 < bank_fraction < calibration_end_fraction < 1')
        self.bank_end=int(len(self.train)*bank_fraction);self.cal_end=int(len(self.train)*calibration_end_fraction)
        # A full 60-tick focus must fit inside *each* disjoint evaluation
        # segment. Otherwise linspace would silently draw calibration queries
        # backwards into the reference bank or audit queries into calibration.
        if self.cal_end-self.bank_end<60 or len(self.train)-self.cal_end<60:
            raise ValueError('Normal train is too short for disjoint 60-tick validation and audit')
        self.bank=self.train[:self.bank_end]
        self.lo,self.hi=np.quantile(self.bank,[.01,.99],axis=0)
        self.hi=np.where(self.hi-self.lo<1e-8,self.lo+1.,self.hi)
        self.period_estimator=period_estimator;self.n_queries=calibration_queries;self.cache={}

    def _retrieve(self, channel, values, focus_start, lo, hi):
        c=channel;info=self._bank(c);context=min(info['context'],focus_start)
        starts=info['starts']
        if context!=info['context']:
            starts=np.arange(context,len(self.bank)-60+1,info['stride'])
            normalized=(self.bank[:,c]-self.lo[c])/(self.hi[c]-self.lo[c])
            contexts=np.stack([normalized[s-context:s] for s in starts])
        else:contexts=info['contexts']
        target=(values[focus_start-context:focus_start,c]-self.lo[c])/(self.hi[c]-self.lo[c])
        if context:
            level_mse=np.mean((contexts-target)**2,axis=1)
            change_mse=np.mean((np.diff(contexts,axis=1)-np.diff(target))**2,axis=1) if context>1 else np.zeros(len(starts))
            distance=level_mse+.5*change_mse
        else:
            level_mse=np.zeros(len(starts));change_mse=np.zeros(len(starts));distance=np.zeros(len(starts))
        chosen=[]
        for j in np.argsort(distance,kind='stable'):
            if all(abs(int(starts[j])-int(starts[k]))>=max(60,info['period'] or 0) for k in chosen):chosen.append(int(j))
            if len(chosen)==8:break
        refs=np.stack([(self.bank[s:s+60,c]-lo[c])/(hi[c]-lo[c]) for s in starts[chosen]])
        return np.quantile(refs,[.1,.5,.9],axis=0),dict(context=context,n_refs=len(chosen),
            context_rmse=float(np.sqrt(distance[chosen[0]])),reference_starts=starts[chosen].tolist(),
            context_level_rmse=float(np.sqrt(level_mse[chosen[0]])),
            context_change_rmse=float(np.sqrt(change_mse[chosen[0]])),
            context_distance_definition='sqrt(level_mse + 0.5 * first_difference_mse)',
            period=info['period'],period_strength=info['strength'])

    def _bank(self,c):
        if c not in self.cache:
            norm=(self.bank[:,c]-self.lo[c])/(self.hi[c]-self.lo[c])
            period,strength=self.period_estimator(norm)
            context=min(max(48,2*period if period else 48),512)
            stride=max(1,min(14,period//12 if period else 14))
            starts=np.arange(context,len(norm)-60+1,stride)
            if not len(starts):raise ValueError('Reference bank too short')
            self.cache[c]=dict(period=period,strength=strength,context=context,stride=stride,
                starts=starts,contexts=np.stack([norm[s-context:s] for s in starts]))
        return self.cache[c]

    def _validate(self,c):
        info=self._bank(c)
        if 'validation' not in info:
            # Same calibration query locations for every channel, no GT.
            locations=np.unique(np.linspace(self.bank_end,self.cal_end-60,self.n_queries).astype(int))
            context_errors=[];focus_errors=[];outside=[]
            for t in locations:
                band,meta=self._retrieve(c,self.train,int(t),self.lo,self.hi)
                target=(self.train[t:t+60,c]-self.lo[c])/(self.hi[c]-self.lo[c])
                context_errors.append(meta['context_rmse'])
                focus_errors.append(float(np.sqrt(np.mean((target-band[1])**2))))
                outside.append(float(np.mean((target<band[0])|(target>band[2]))))
            info['validation']=dict(context_errors=np.array(context_errors),
                median_focus_rmse_q90=float(np.quantile(focus_errors,.9)),
                mean_fraction_outside_descriptive_band=float(np.mean(outside)),n=len(locations))
        return info['validation']

    def band(self,train,tile,channel,focus,lo,hi,test=None,tile_start=0,return_metadata=False):
        if test is None:raise ValueError('Explicit full context series required for quality assessment')
        fs,fe=focus
        if fe-fs!=60:raise ValueError('Reference quality expects a 60-tick focus')
        result,meta=self._retrieve(channel,test,tile_start+fs,lo,hi)
        validation=self._validate(channel)
        errors=validation['context_errors']
        p=float((1+np.sum(errors>=meta['context_rmse']))/(1+len(errors)))
        info=self._bank(channel)
        insufficient=meta['context']<info['context'] or meta['n_refs']<4 or len(errors)<20
        status='insufficient_support' if insufficient else 'out_of_support' if p<=.05 else 'context_supported'
        audit=self._audit_support(channel)
        if status=='context_supported' and audit['warning']:status='validation_unstable'
        meta.update(limited=insufficient,reference_status=status,context_support_p=p,
            validation_queries=validation['n'],validation_focus_rmse_q90=validation['median_focus_rmse_q90'],
            validation_outside_fraction=validation['mean_fraction_outside_descriptive_band'],
            bank_interval=[0,self.bank_end],validation_interval=[self.bank_end,self.cal_end],
            support_audit_failure_fraction=audit['failure_fraction'],support_audit_warning=audit['warning'],
            reliability_scope='pre-focus context support only; does not establish focus normality')
        return (result,meta) if return_metadata else result

    def _audit_support(self,c):
        info=self._bank(c)
        if 'support_audit' not in info:
            # Audit only warns; it never refits a reference or a cutoff.
            validation=self._validate(c);errors=validation['context_errors'];failed=[]
            locations=np.unique(np.linspace(self.cal_end,len(self.train)-60,32).astype(int))
            for t in locations:
                _,meta=self._retrieve(c,self.train,int(t),self.lo,self.hi)
                p=(1+np.sum(errors>=meta['context_rmse']))/(1+len(errors))
                failed.append(p<=.05 or meta['n_refs']<4)
            rate=float(np.mean(failed))
            info['support_audit']=dict(failure_fraction=rate,warning=rate>.10,queries=len(locations))
        return info['support_audit']

    def audit_channel(self,c):
        report=self._audit_support(c)
        return dict(channel=c,queries=report['queries'],out_of_support_fraction=report['failure_fraction'],
                    transfer_warning=report['warning'],validation=dict(self._validate(c)))
