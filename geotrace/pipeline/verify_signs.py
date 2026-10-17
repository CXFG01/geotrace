"""Second-opinion sign verification with Nemotron-3.5-Super-VL-120B (W&B): is the claimed street sign visible in the clip's frames?"""
import base64, json, os, re, subprocess, sys, tempfile, urllib.parse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/vast-builders-challenge/tools/trip-risk"))
import vsslib
DATA=os.path.expanduser("~/trip-risk-data")
MODEL="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
HDR={"Authorization":"Bearer "+os.environ["WANDB_API_KEY"],"Content-Type":"application/json"}
if os.environ.get("WANDB_TEAM") and os.environ.get("WANDB_PROJECT"): HDR["OpenAI-Project"]=os.environ["WANDB_TEAM"]+"/"+os.environ["WANDB_PROJECT"]
idx=json.load(open(f"{DATA}/index.json")); tasks=json.load(open(f"{DATA}/verify_tasks.json"))
OUT=f"{DATA}/verify_signs.json"; done=json.load(open(OUT)) if os.path.exists(OUT) else {}
def key(t): return f'{t["trip"]}|{t["seg"]}|{t["name"]}'
def sheet(src):
    url=vsslib.BASE+"/api/v1/videos/stream?source="+urllib.parse.quote(src,safe="")+"&token="+vsslib.token()
    with tempfile.TemporaryDirectory() as d:
        mp4=os.path.join(d,"s.mp4"); open(mp4,"wb").write(urllib.request.urlopen(url,timeout=120).read())
        out=os.path.join(d,"sheet.jpg")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",mp4,"-vf","fps=1.2,scale=960:-1,tile=3x2","-frames:v","1","-q:v","3",out],check=True)
        return base64.b64encode(open(out,"rb").read()).decode()
def ask(t,tokens=3500):
    src=idx[t["trip"]][min(t["seg"],len(idx[t["trip"]])-1)]["source"]
    prompt=("These are 6 frames in time order (3 per row) from a forward-facing dashcam. List every STREET-NAME sign (the green or blue sign on a pole naming a street) "
            "you can clearly read, and say whether a sign reading '%s' is clearly visible. Do not guess. Reply ONLY with JSON: "
            '{"street_names_read":[...],"target_visible":true|false}')%t["name"]
    body={"model":MODEL,"max_tokens":tokens,"temperature":0,"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+sheet(src)}}]}]}
    r=json.load(urllib.request.urlopen(urllib.request.Request("https://api.inference.wandb.ai/v1/chat/completions",data=json.dumps(body).encode(),headers=HDR),timeout=300))
    txt=(r["choices"][0]["message"].get("content") or "")
    m=re.search(r"\{[^{}]*\}",txt,re.S)
    try: j=json.loads(m.group(0)) if m else {}
    except Exception: j={}
    if "target_visible" not in j and tokens<10000: return ask(t,10000)
    return j
def work(t):
    try: return key(t),ask(t)
    except Exception as e: return key(t),{"error":str(e)[:120]}
todo=[t for t in tasks if key(t) not in done or "error" in done[key(t)]]
print("tasks",len(tasks),"todo",len(todo),flush=True)
n=0
with ThreadPoolExecutor(4) as ex:
    for k,r in ex.map(work,todo):
        done[k]=r; n+=1
        if n%10==0: json.dump(done,open(OUT,"w")); print("done",n,flush=True)
json.dump(done,open(OUT,"w")); print("finished",len(done),flush=True)
