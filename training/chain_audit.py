"""Read-only checkpoint supervision and fixed-continuation decision diagnostics."""
from collections import defaultdict
import json
from pathlib import Path
import torch
from actions import ACTION_IDS
from cfr_solver import regret_matching
from ml.model import StreetNetworks, STREETS, encode_batch
from ml.train import weight_quality
from ml.street_training import grouped_partition
from training.checkpoint import unpack_memory
from training.metrics import decision_context, opening_hand_class
from training.preflop_diagnostics import opening_matrix, paired_opening_values


class AuditPolicy:
    def __init__(self, model: StreetNetworks) -> None:
        self.model = model

    def probabilities(self, observation) -> tuple[float, ...]:
        if self.model.kind == "strategy":
            return self.model.probabilities(observation)
        with torch.no_grad():
            output = self.model.specialists[STREETS[observation.street]](encode_batch([observation]))[0].tolist()
        return regret_matching(output, observation.legal_mask) if self.model.kind == 'advantage' else tuple(output)


def replay_metrics(model: StreetNetworks, samples: list, seed: int) -> dict:
    predictions=[]
    with torch.no_grad():
        for offset in range(0,len(samples),128):
            predictions.extend(model.specialists['PREFLOP'](encode_batch([s.state for s in samples[offset:offset+128]])).double().tolist())
    groups=defaultdict(list)
    for s,p in zip(samples,predictions):
        split=grouped_partition(s,seed);context=decision_context(s.state)
        groups[f'{split}/all'].append((s,p))
        groups[f'{split}/{context}'].append((s,p))
    result={}
    for name,rows in groups.items():
        ss=[s for s,p in rows];w=torch.tensor([s.weight*s.iteration for s in ss],dtype=torch.float64);w/=w.sum()
        target=torch.tensor([s.target for s in ss],dtype=torch.float64);pred=torch.tensor([p for s,p in rows],dtype=torch.float64)
        mask=torch.tensor([s.state.legal_mask for s in ss]);n=mask.sum(1)
        if model.kind=='advantage':
            loss=((pred-target).square()*mask).sum(1)/n
            base=(target.square()*mask).sum(1)/n
            extra={'nonpositive_fraction':float((pred.masked_fill(~mask,-torch.inf).max(1).values<=0).double().mean())}
        else:
            loss=(target*(target.clamp_min(1e-12).log()-pred.clamp_min(1e-12).log())).sum(1)
            base=(target*(target.clamp_min(1e-12)*n[:,None]).log()).sum(1)
            extra={'mean_l1':float((w*(pred-target).abs().sum(1)).sum())}
        result[name]={'samples':len(ss),'loss':float((w*loss).sum()),'simple_baseline_loss':float((w*base).sum()),**weight_quality(ss),**extra}
    openings={}
    for label in ('AA','KK','AKs','72o','32o'):
        rows=[(s,p) for s,p in zip(samples,predictions) if opening_hand_class(s.state)==label]
        if not rows:openings[label]={'samples':0};continue
        weights=[s.weight*s.iteration for s,p in rows];total=sum(weights)
        openings[label]={'samples':len(rows),'train_samples':sum(grouped_partition(s,seed)=='train' for s,p in rows),
            'targets':[sum(s.target[i]*w for (s,p),w in zip(rows,weights))/total for i in range(len(ACTION_IDS))],
            'predictions':[sum(p[i]*w for (s,p),w in zip(rows,weights))/total for i in range(len(ACTION_IDS))],**weight_quality([s for s,p in rows])}
    return {'partitions':result,'openings_across_contexts':openings}


def audit(source: Path, destination: Path, samples: int = 128) -> dict:
    torch.set_num_threads(1)
    raw=torch.load(source,map_location='cpu',weights_only=True)
    count=2 if raw['track']=='hu' else 3
    models={}
    report={'iteration':raw['iteration'],'track':raw['track'],'run_id':raw['metadata']['run_id'],'actions':ACTION_IDS,
            'scope':'Historical supervised targets and fixed-policy rollout EVs; no full-game equilibrium claim',
            'config':raw['config'],'replay':{},'opening':{},'rollouts':{},'states':raw['states']}
    for kind,key in [('advantage','advantage_weights'),('strategy','average_weights')]:
        model=StreetNetworks(kind);model.load_state_dict(raw[key]);model.eval();models[kind]=model
        if not bool(model.trained.all()): raise ValueError('Audit requires all street specialists trained')
        memory=unpack_memory(raw['preflop'][kind]);report['replay'][kind]=replay_metrics(model,memory.samples,raw['seed'])
        report['replay'][kind]['allocation']=memory.diagnostics()
        report['opening'][kind]=opening_matrix(AuditPolicy(model),count,37.5 if count==2 else 25.)
        print(raw['track'],kind,'replay and 1326-combo probes complete',flush=True)
    # Independent seeds/deals; compare both root policies under each fixed continuation.
    for kind,model in models.items():
        policy=AuditPolicy(model)
        for label,holding in [('AA',(48,49)),('KK',(44,45)),('72o',(20,1))]:
            result=paired_opening_values(policy,count,37.5 if count==2 else 25.,holding,samples=samples,seed=810713)
            report['rollouts'][kind+'/'+label]=result
            print(raw['track'],kind,label,'rollout complete',flush=True)
    from training.street_experiments import river_reference
    report['river']={kind:river_reference(AuditPolicy(model),count) for kind,model in models.items()}
    # Validation-bank samples contain binary encodings; retain the numerical metrics only in JSON.
    for states in report['states'].values():
        for state in states.values():state.pop('validation_bank',None)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
