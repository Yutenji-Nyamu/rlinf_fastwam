"""Print only sanitized subscription quota fields from the root-only Mihomo config."""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

import yaml


with open("/etc/mihomo/config.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

providers = config.get("proxy-providers") or {}
urls = [entry.get("url") for entry in providers.values() if isinstance(entry, dict) and entry.get("url")]
if not urls:
    raise SystemExit("no provider URL found")

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for index, url in enumerate(urls, start=1):
    request = urllib.request.Request(url, headers={"User-Agent": "ClashforWindows/0.20.39"})
    with opener.open(request, timeout=20) as response:
        response.read(1)
        raw = response.headers.get("subscription-userinfo")
        status = response.status
    if not raw:
        print(json.dumps({"provider_index": index, "http_status": status, "quota_header": "absent"}))
        continue
    fields = {}
    for item in raw.split(";"):
        key, _, value = item.strip().partition("=")
        if key in {"upload", "download", "total", "expire"} and value.isdigit():
            fields[key] = int(value)
    required = {"upload", "download", "total", "expire"}
    if set(fields) != required:
        raise SystemExit("subscription header missing required numeric fields")
    used = fields["upload"] + fields["download"]
    payload = {
        "provider_index": index,
        "http_status": status,
        **fields,
        "used_bytes": used,
        "remaining_bytes": fields["total"] - used,
        "used_gib": round(used / 2**30, 3),
        "remaining_gib": round((fields["total"] - used) / 2**30, 3),
        "expire_utc": dt.datetime.fromtimestamp(fields["expire"], dt.timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True))
