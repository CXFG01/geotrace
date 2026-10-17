"""PF3: pinned to the solver's accepted anchor junctions (within 140 m, +-1 segment); flow yaw + straight prior choose the roads between."""
import json, math, os, pickle
import numpy as np
import geo_match as gm
import run_pf2 as P
from build_graph import xy
def run_trip3(trip,n=60000,seed=1,use_yaw=True):
    obs=gm.observations(P.idx[trip],P.geo,P.G)
    for o in obs:
        o["names"]=[]; o["strength"]={}; o["maneuver"]="straight"
    pins={}
    for a in P.chain[trip]["accepted"]:
        i=int(round(a["idx"])); p=np.array(xy(*a["latlon"]))
        for j in range(max(0,i-1),min(len(obs),i+2)):
            pins.setdefault(j,[]).append(p)
    # a pin set per step: the car is near at least one of the anchors near this time
    pins={k:np.array(v) for k,v in pins.items()}
    fl=json.load(open(f"{P.D}/flow_{trip}.json")); yaw=np.array([f["yaw_deg"] for f in fl]) if use_yaw else None
    r=gm.run(P.G,obs,n=n,seed=seed,use_turns=False,use_names=False,yaw=yaw,sig_yaw=30.0,speed_mult=1.2,pins=pins,pin_tol=140.0)
    return obs,r
def hav(a,b):
    R=6371000;p1,p2=math.radians(a[0]),math.radians(b[0]);h=math.sin((p2-p1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(b[1]-a[1])/2)**2
    return 2*R*math.asin(math.sqrt(h))
if __name__=="__main__":
    tot=[0,0,0]; store={}
    for t in P.res0["trips"]:
        trip=t["id"]; obs,r=run_trip3(trip); pos,pts=P.path_info(r); store[trip]=(pos,pts,r["perplexity"])
        da=[min(hav(pos[min(len(pos)-1,max(0,int(round(a["t"]/5))+o))],a["latlon"]) for o in (-1,0,1)) for a in t["anchors"]]
        pois=[p for p in t.get("pois",[]) if p["matches"]<=6]
        dp=[min(hav(pos[min(len(pos)-1,max(0,p["t"]//5+o))],p["latlon"]) for o in (-1,0,1)) for p in pois]
        tot[0]+=len(dp); tot[1]+=sum(x<=300 for x in dp); tot[2]+=sum(x<=600 for x in dp)
        print(trip,"anchors within 300m:",sum(x<=300 for x in da),"/",len(da),"| POIs within 300m:",sum(x<=300 for x in dp),"/",len(dp),"| route pts",len(pts),flush=True)
    print("POI total",tot)
    pickle.dump(store,open(f"{P.D}/pf3_store.pkl","wb"))
