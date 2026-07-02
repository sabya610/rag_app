import subprocess, json

patch = json.dumps({"spec": {"chartVersion": "0.9.12"}})
with open("/tmp/p.json", "w") as f:
    f.write(patch)

r = subprocess.run(
    ["kubectl", "patch", "ezappconfig", "rag-app-0.8.9-1782725530822",
     "-n", "rag-app", "--type=merge", "--patch-file=/tmp/p.json"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
print(r.stdout.decode())
print(r.stderr.decode())
