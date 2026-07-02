import subprocess, json

values = """replicaCount: 1

image:
  repository: sabya610/rag-app
  tag: llama3.1-8b-q4-v13
  pullPolicy: IfNotPresent

initImage:
  repository: sabya610/busybox
  tag: "1.36"

service:
  type: ClusterIP
  port: 80

env:
  DB_HOST: "postgres-headless.rag-app.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "ragdb"
  DB_USER: "postgres"
  DB_PASS: "postgres"
  MODEL_PATH: "/app/rag_app/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  EMBEDDING_MODEL: "/app/rag_app/models/embedding/all-MiniLM-L6-v2"
  SFDC_ENABLED: "true"
  SF_URL: "https://hp.my.salesforce.com"
  SFDC_PRODUCT_QUEUE: "HPE Ezmeral"
  SFDC_PRODUCT_LINE: "CONT PLT SW (RM)"
  SFDC_SEARCH_LIMIT: "5"
  MAX_CONTEXT_CHARS: "12000"

sfdc:
  sessionId: "00Dd0000000bUlK!ARAAQOK8_uF259fTsrTEOqnIwYwOlpORuawjCyjrt1P8IILRB05UVw_DbBxgq5XQ23dRPt1bBFuxGNSey.9hFbN44evL2QeL"
  sessionIdFile: "/etc/sfdc/sid.txt"
  clientId: ""
  clientSecret: ""
  username: ""
  password: ""
  securityToken: ""
  loginUrl: "https://login.salesforce.com"

resources:
  requests:
    memory: "24Gi"
    cpu: "2"
  limits:
    memory: "32Gi"
    cpu: "4"

proxy:
  httpProxy: "http://hpeproxy.its.hpecorp.net:80"
  httpsProxy: "http://hpeproxy.its.hpecorp.net:80"
  noProxy: "10.0.0.0/8,192.168.0.0/16,localhost,127.0.0.1,.hpecorp.net,.storage.hpecorp.net"

ingress:
  enabled: false
  host: rag.local

postgresql:
  enabled: true
  useStatefulSet: false
  image:
    repository: sabya610/pgvector
    tag: latest
  storage: 5Gi
  username: postgres
  password: postgres
  database: ragdb

ezua:
  virtualService:
    endpoint: "rag-app.${DOMAIN_NAME}"
    istioGateway: "istio-system/ezaf-gateway"
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"
    matchLabels:
      istio: "ingressgateway"
"""

patch = json.dumps({"spec": {"chartVersion": "0.9.12", "values": values}})
with open("/tmp/p2.json", "w") as f:
    f.write(patch)

r = subprocess.run(
    ["kubectl", "patch", "ezappconfig", "rag-app-0.8.9-1782725530822",
     "-n", "rag-app", "--type=merge", "--patch-file=/tmp/p2.json"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
print(r.stdout.decode())
print(r.stderr.decode())
