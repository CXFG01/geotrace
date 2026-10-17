import json, math, os
import numpy as np
D="data"
res=json.load(open(f"{D}/results.json")); geo=json.load(open(f"{D}/geo_pass.json")); idx=json.load(open(f"{D}/index.json"))
K=math.cos(math.radians(43.67))
def xy(p): return np.array([(p[1]+79.4)*111320*K,(p[0]-43.67)*110540])
def arclen(R):
    return np.r_[0,np.cumsum(np.linalg.norm(np.diff(R,axis=0),axis=1))]
def point_at(R,s,L):
    k=int(np.searchsorted(s,min(max(L,0),s[-1])))
    k=min(max(k,1),len(R)-1); r=(L-s[k-1])/max(s[k]-s[k-1],1e-6); return R[k-1]+(R[k]-R[k-1])*min(max(r,0),1)
def bearing(a,b): return math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))%360
def dh(a,b): return (b-a+180)%360-180
def net_change(R,s,s0,s1,span=35.0):
    """net heading change between the direction just before s0 and just after s1 (arclength positions on route)"""
    a0=point_at(R,s,s0-span); a1=point_at(R,s,s0); b0=point_at(R,s,s1); b1=point_at(R,s,s1+span)
    if np.linalg.norm(a1-a0)<5 or np.linalg.norm(b1-b0)<5: return 0.0
    return dh(bearing(a0,a1),bearing(b0,b1))
tot=dict(big=0,agree=0,opp=0,flow_big=0,flow_ok=0,flow_opp=0); xs=[];ys=[]
for t in res["trips"]:
    R=np.array([xy(p) for p in t["route"]]); s=arclen(R)
    fl=json.load(open(f"{D}/flow_{t['id']}.json")); yaw=np.array([f["yaw_deg"] for f in fl]); pos=t["pos"]
    # arclength of each pos on route (nearest vertex)
    P=np.array([xy(p[:2]) for p in pos])
    ar=np.array([s[int(np.argmin(np.linalg.norm(R-q,axis=1)))] for q in P])
    # keep monotone
    ar=np.maximum.accumulate(ar)
    big=agree=opp=0; fb=fo=fopp=0
    for i in range(1,len(pos)-2):
        if pos[i-1][2] or pos[i+2][2]: continue
        rc=net_change(R,s,ar[i],ar[i+1])
        fy=yaw[i-1:i+2].sum()
        xs.append(rc); ys.append(fy)
        if abs(rc)>=40:
            big+=1
            if abs(fy)>=25 and fy*rc>0: agree+=1
            elif abs(fy)>=25: opp+=1
        if abs(fy)>=40:
            fb+=1
            if abs(rc)>=25 and fy*rc>0: fo+=1
            elif abs(rc)>=25: fopp+=1
    print(t["id"],"route turns>=40:",big,"flow agrees:",agree,"opposite:",opp,"| flow turns>=40 (3-seg):",fb,"route reproduces:",fo,"opposite:",fopp)
    for k,v in zip(("big","agree","opp","flow_big","flow_ok","flow_opp"),(big,agree,opp,fb,fo,fopp)): tot[k]+=v
xs=np.array(xs);ys=np.array(ys)
print("TOTAL",tot)
print("r = %.2f over %d windows"%(np.corrcoef(xs,ys)[0,1],len(xs)))
