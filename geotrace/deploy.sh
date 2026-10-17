#!/usr/bin/env bash
# Deploy TripScore to this team's namespace per deployment/deploy-app-no-registry (no image build, code in ConfigMap).
set -euo pipefail
CFG=$(ls /config/*.config | head -1)
USERNAME=$(grep '^USERNAME=' "$CFG" | cut -d= -f2-)
INGRESS_URL=$(grep '^INGRESS_URL=' "$CFG" | cut -d= -f2-)
PASSWORD=$(grep '^PASSWORD=' "$CFG" | cut -d= -f2-)
NS="$USERNAME"
export KUBECONFIG=/config/${NS}-k8s.yaml
TEAM_N="${USERNAME#team-}"
APP_HOST="video-lab-team-${TEAM_N}.cosmos.vastdata.com"
APP_NAME=tripscore
APP_PORT=8080
APP_DIR=$HOME/vast-builders-challenge/tools/geotrace/app

kubectl -n "$NS" create configmap "${APP_NAME}-code" --from-file="$APP_DIR" --dry-run=client -o yaml | kubectl apply --server-side --force-conflicts -f - >/dev/null
kubectl -n "$NS" create secret generic "${APP_NAME}-vss-creds" \
  --from-literal=VSS_URL="http://video-backend-service.${NS}.svc.cluster.local:8000" --from-literal=VSS_USERNAME="$USERNAME" --from-literal=VSS_PASSWORD="$PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl -n "$NS" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  replicas: 1
  selector:
    matchLabels: {app: ${APP_NAME}}
  template:
    metadata:
      labels: {app: ${APP_NAME}}
    spec:
      containers:
      - name: app
        image: python:3.12-slim
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: ${APP_PORT}
        env:
        - {name: PORT, value: "${APP_PORT}"}
        - name: VSS_URL
          valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_URL}}
        - name: VSS_USERNAME
          valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_USERNAME}}
        - name: VSS_PASSWORD
          valueFrom: {secretKeyRef: {name: ${APP_NAME}-vss-creds, key: VSS_PASSWORD}}
        volumeMounts:
        - {name: code, mountPath: /code}
        workingDir: /code
        command: ["python", "main.py"]
        readinessProbe:
          httpGet: {path: /health, port: ${APP_PORT}}
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: code
        configMap: {name: ${APP_NAME}-code}
---
apiVersion: v1
kind: Service
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
spec:
  selector: {app: ${APP_NAME}}
  ports:
  - {name: http, port: 80, targetPort: ${APP_PORT}}
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${APP_NAME}
  labels: {app: ${APP_NAME}}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
spec:
  ingressClassName: nginx
  rules:
  - host: ${APP_HOST}
    http:
      paths:
      - path: /app(/|\$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: ${APP_NAME}
            port: {number: 80}
EOF

kubectl -n "$NS" rollout restart deploy/"$APP_NAME" >/dev/null
kubectl -n "$NS" rollout status deploy/"$APP_NAME" --timeout=120s
kubectl -n "$NS" get pods,svc,ingress -l app="$APP_NAME"
sleep 3
echo "host: $APP_HOST"
curl -sS -m 15 -o /dev/null -w "app http %{http_code}\n" "http://${APP_HOST}/app" || true
curl -sS -m 15 "http://${APP_HOST}/app/health" || true; echo
