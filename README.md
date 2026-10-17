# GeoTrace

**Where was this dashcam clip recorded? Let the video tell you.**

GeoTrace reconstructs where a dashcam trip happened from the video alone, with no GPS and no metadata. Built for the VAST Builders Challenge (Video Agents) on the VAST video search and summarization stack.

- NVIDIA **Cosmos3-Reason** reads street-name signs, business names, tram tracks, motion and turns from every 5-second segment.
- A numpy **optical-flow** pass measures yaw (turns) independently of the model.
- An **anchor-chain solver** matches the sign reads to the **OpenStreetMap** road network. It picks the largest set of reads that are mutually consistent in network distance versus elapsed time, and rejects the rest as misreads.
- **Nemotron-3.5-Super-VL-120B** (Weights & Biases serverless inference) re-checks each sign from the clip's frames as a second opinion.
- A web app shows the route, anchors, a video-synced marker, and a claim-verification box ("was this recorded at ...?").

## Results (6 Toronto dashcam trips, 15 minutes each)

| Check | Result |
|---|---|
| Hold-one-sign-out (hide a sign, predict the car's position at that moment) | median error 10 to 102 m per trip (Trip 02: 16 m) |
| Independent business-name check (never used for matching, geocoded with OSM) | 38 of 73 within 300 m of the matched position |
| Second model confirms accepted vs rejected sign reads | 47 of 93 vs 6 of 50 |
| Claim check | "Times Square, New York" is 557 km off; a real in-trip place is about 36 m off |

## Honest limits

- City-scale (downtown Toronto road network), not world-scale.
- No GPS ground truth exists in this dataset. The numbers measure self-consistency, not true accuracy.
- Turn sequences alone leave hundreds of candidate places; the sign reads are what makes a trip unique. Turn directions are not used for matching (the optical-flow check confirmed 48 of the model's 87 labelled turns and contradicted 7, but the route between signs is too coarse to validate turn by turn).
- About half of the accepted sign reads are not confirmed by the second model (frames can miss a sign).
- Video geolocation can reveal where someone was. This project runs only on the provided challenge footage and is framed as claim verification.

## Layout

```
geotrace/
  app/            stdlib web app (Leaflet map, video proxy), results.json bundled
  pipeline/       sign/motion pass (geo_pass.py), OSM fetch + graph, solver, evaluation, deploy helpers
  analysis/       optical flow, turn checks, ambiguity experiments
  deploy.sh       Kubernetes deploy used for the challenge (ConfigMap + Secret + Ingress /app)
tripscore/        earlier idea: explainable per-trip driving-risk score from the same footage
```

## Pipeline

1. `pipeline/geo_pass.py` (on the VM): one Cosmos3-Reason video call per segment -> signs, businesses, tram tracks, motion, maneuver.
2. `pipeline/fetch_osm.py`, `build_graph.py`: Overpass fetch and a junction-to-junction directed road graph.
3. `pipeline/anchor_chain.py`: robust anchor-chain solver with leave-one-out evaluation.
4. `pipeline/verify_signs.py`: second-model confirmation. `poi_check.py`: independent business-name check.
5. `pipeline/build_geo_results.py`: assembles `app/results.json`.
6. `app/main.py`: serves the UI and a seekable video proxy.

## Configuration

No credentials are stored in this repo. Scripts read `VSS_URL` / `VSS_USERNAME` / `VSS_PASSWORD` (or `INGRESS_URL` / `USERNAME` / `PASSWORD`), `GPU_BEARER_TOKEN` and `WANDB_API_KEY` from the environment. The challenge VM exposes them from `/config/*.config`. Bundled `results.json` references the challenge team's S3 segment URIs, so the video proxy only works inside that environment; the map, anchors and claim box work from the data alone.

## Credits

VAST Data, NVIDIA Cosmos and Nemotron, Weights & Biases / CoreWeave, OpenStreetMap contributors (ODbL).
