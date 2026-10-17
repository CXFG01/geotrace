"""Particle filter v2: continuous flow-yaw turn evidence + only the solver-accepted sign anchors."""
import json, math, os, sys
import numpy as np
import geo_match as gm
from build_graph import LAT0, LON0, KX, KY
D=gm.D
def ll(x,y): return [round(LAT0+y/KY,5), round(LON0+x/KX,5)]
idx=json.load(open(f"{D}/index.json")); geo=json.load(open(f"{D}/geo_pass.json")); chain=json.load(open(f"{D}/chain_results.json"))
res0=json.load(open(f"{D}/results.json"))
G=gm.Graph()
def run_trip(trip,n=60000,seed=1,use_yaw=True,use_names=True,turn_vlm=False):
    obs=gm.observations(idx[trip],geo,G)
    for o in obs:
        o["names"]=[]; o["strength"]={}
        if not turn_vlm: o["maneuver"]="straight"
    for a in chain[trip]["accepted"]:
        i=int(round(a["idx"]))
        for j in range(max(0,i-1),min(len(obs),i+2)):
            if a["name"] not in obs[j]["names"]:
                obs[j]["names"].append(a["name"]); obs[j]["strength"][a["name"]]=(1,1)
    fl=json.load(open(f"{D}/flow_{trip}.json")); yaw=np.array([f["yaw_deg"] for f in fl]) if use_yaw else None
    r=gm.run(G,obs,n=n,seed=seed,use_turns=turn_vlm,use_names=use_names,yaw=yaw,sig_yaw=30.0,name_pen=0.08,speed_mult=1.2,name_tol=150.0)
    return obs,r
def path_info(r):
    pos=[]; pts=[]; last=None
    for e,s in zip(r["path_e"],r["path_s"]):
        p=G.position(np.array([e]),np.array([s]))[0]; pos.append(ll(*p))
        if e!=last:
            pts+= [ll(*q) for q in G.E[e]["pts"]]; last=e
    return pos,pts
if __name__=="__main__":
    for trip in sys.argv[1:]:
        for label,kw in (("flow yaw + anchors",{}),("anchors only",dict(use_yaw=False))):
            obs,r=run_trip(trip,**kw)
            pos,pts=path_info(r)
            names=[]
            for e in r["path_e"]:
                nm=G.E[e]["name"]
                if nm and (not names or names[-1]!=nm): names.append(nm)
            print(trip,label,"final mass",round(r["final_cell_mass"],2),"perp end",round(r["perplexity"][-1],1),"| roads:",names[:12])
