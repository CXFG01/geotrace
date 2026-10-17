"""How unique is a turn sequence on the real Toronto road graph? Random drives, then count start edges that can realise the same sequence."""
import numpy as np, random, math
import geo_match as gm
G=gm.Graph(); n=G.n
def sym(e,c):
    a=gm.angdiff(G.b_out[c],G.b_in[e])
    return 'S' if abs(a)<30 else ('L' if -150<a<=-30 else ('R' if 30<=a<150 else 'U'))
NEXT={}   # (edge,symbol)->list of next edges
for e in range(n):
    for k in range(G.cnt[e]):
        c=int(G.OUT[e,k]); NEXT.setdefault((e,sym(e,c)),[]).append(c)
inter=np.array([len({G.E[c]['v'] for c in G.OUT[e,:G.cnt[e]] if c>=0})>=2 for e in range(n)])  # real choice at end of edge
rng=random.Random(5)
def drive(L):
    e=rng.randrange(n); seq=[]
    while len(seq)<L:
        if G.cnt[e]==0: return None
        c=int(G.OUT[e,rng.randrange(G.cnt[e])] if rng.random()>0.7 else min(G.OUT[e,:G.cnt[e]],key=lambda c:abs(gm.angdiff(G.b_out[c],G.b_in[e]))))
        if inter[e]: seq.append(sym(e,c))
        e=c
    return seq
def count_exact(seq,starts):
    ok=0
    for e in starts:
        cur=[e]; good=True; i=0
        # follow decisions only at real intersections
        for s in seq:
            nxt=[]
            for x in cur:
                y=x
                while not inter[y] and G.cnt[y]>0: y=int(G.OUT[y,0])
                nxt+=NEXT.get((y,s),[])
            cur=list(set(nxt))
            if not cur: good=False;break
        ok+=good
    return ok
starts=random.Random(1).sample(range(n),1500)
scale=n/len(starts)
print("directed edges",n)
for L in (4,6,8,10,12,15,20,25):
    res=[]
    for _ in range(40):
        s=drive(L)
        if s: res.append(count_exact(s,starts)*scale)
    print(f"exact L/R/S at each intersection, {L:2d} decisions: median matching start edges {np.median(res):8.0f}  (p10 {np.percentile(res,10):.0f}, p90 {np.percentile(res,90):.0f})",flush=True)
# turns only (straights unknown): collapse S, allow any number of straight intersections between turns
def count_turns_only(turns,starts):
    ok=0
    for e in starts:
        cur={e}; good=True
        for t in turns:
            nxt=set()
            for x in cur:
                y=x
                for _ in range(30):
                    while not inter[y] and G.cnt[y]>0: y=int(G.OUT[y,0])
                    nxt.update(NEXT.get((y,t),[]))
                    st=NEXT.get((y,'S'),[])
                    if not st: break
                    y=st[0]
            cur=nxt
            if not cur: good=False;break
        ok+=good
    return ok
print("--- turns only (number of straight blocks unknown)")
for k in (2,4,6,8,10,12):
    res=[]
    for _ in range(30):
        s=drive(60); 
        if not s: continue
        t=[x for x in s if x in 'LR'][:k]
        if len(t)<k: continue
        res.append(count_turns_only(t,starts[:600])*(n/600))
    print(f"{k:2d} turns: median matching start edges {np.median(res):8.0f} of {n}",flush=True)
