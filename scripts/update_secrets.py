import json
from pathlib import Path

env = {}
for line in Path(".env").read_text().strip().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k, v = k.strip(), v.strip().strip("'\"")
    if not k.startswith("AWS_"):
        env[k] = v

Path("/tmp/cvee-secrets.json").write_text(json.dumps(env, ensure_ascii=False))
print(f"✓ {len(env)} keys written to /tmp/cvee-secrets.json")
