"""Read-only W&B metadata snapshot; uses existing SDK authentication."""
import itertools,json
from pathlib import Path
import wandb
root=Path(__file__).resolve().parent
api=wandb.Api(timeout=30,overrides={"base_url":"https://api.wandb.ai"})
entity="437snowcap-sungkyunkwan-university"
project="adds-sac-ued6-261011ENV"
result=[]
for r in itertools.islice(api.runs(f"{entity}/{project}",order="-created_at",per_page=15),12):
 c={k:v for k,v in r.config.items() if not any(w in k.lower() for w in ("api_key","token","password","secret"))}
 s={k:v for k,v in r.summary._json_dict.items() if isinstance(v,(int,float,str,bool)) or v is None}
 row=dict(id=r.id,name=r.name,state=r.state,created_at=r.created_at,url=r.url,project=project,config=c,summary=s)
 result.append(row)
 print(json.dumps(dict(id=r.id,name=r.name,state=r.state,created_at=r.created_at,episode=s.get("global_episode"),approach=c.get("REWARD_W_SHAPE_APPROACH"),progress=c.get("REWARD_W_SHAPE_PROGRESS"),resumed=c.get("run/resumed_episode"),hps=s.get("eval/validation/200m/robots_3/policy_deterministic/hazard_person_steps")),ensure_ascii=False),flush=True)
(root/"inventory.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
