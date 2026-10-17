import json, os
import numpy as np
import geo_match as gm
idx=json.load(open("data/index.json")); geo=json.load(open("data/geo_pass.json"))
G=gm.Graph(); out={}
for trip in sorted(idx):
    obs=gm.observations(idx[trip],geo,G)
    row={}
    for label,kw in (("turns only",dict(use_names=False,use_tram=False)),("turns + tram tracks",dict(use_names=False))):
        ps=[]
        for seed in (1,2,3):
            r=gm.run(G,obs,n=40000,seed=seed,keep_hist=False,**kw); ps.append(r["perplexity"])
        row[label]=[round(float(np.median([p[i] for p in ps])),1) for i in range(len(obs))]
    out[trip]={"turns":sum(1 for o in obs if o["maneuver"] in ("left_turn","right_turn","u_turn")),"curves":row}
    print(trip,{k:(v[0],v[10],v[30],v[60],v[120],v[-1]) for k,v in row.items()},"turns",out[trip]["turns"])
json.dump(out,open("data/pf_curves.json","w"))
