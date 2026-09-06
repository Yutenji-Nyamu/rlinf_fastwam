#!/usr/bin/env bash
set -u

ROOT_SCRIPT=$(cat <<'ROOT'
set -u

printf '=== TIME_SERVICE_CONFIG ===\n'
date --iso-8601=seconds
timedatectl show -p Timezone -p NTPSynchronized --value 2>/dev/null || true
printf 'mihomo_service='; systemctl is-active mihomo 2>/dev/null || true
systemctl show mihomo -p ActiveEnterTimestamp -p ExecMainStartTimestamp -p MainPID -p NRestarts --no-pager 2>/dev/null || true
/usr/local/bin/mihomo -v 2>/dev/null || mihomo -v 2>/dev/null || true
printf '%s\n' '-- config/provider metadata only'
find /etc/mihomo -maxdepth 3 -type f -printf '%M %U:%G %s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort

printf '=== SUBSCRIPTION_AND_CONTROLLER_SANITIZED ===\n'
/usr/bin/python3 - <<'PY'
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request

import yaml

with open('/etc/mihomo/config.yaml', encoding='utf-8') as handle:
    cfg = yaml.safe_load(handle) or {}

providers_cfg = cfg.get('proxy-providers') or {}
urls = [x.get('url') for x in providers_cfg.values() if isinstance(x, dict) and x.get('url')]
for index, url in enumerate(urls, start=1):
    payload = {'provider_index': index}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers={'User-Agent': 'ClashforWindows/0.20.39'})
        with opener.open(req, timeout=20) as resp:
            resp.read(1)
            raw = resp.headers.get('subscription-userinfo')
            payload['http_status'] = resp.status
        if raw:
            fields = {}
            for item in raw.split(';'):
                key, _, value = item.strip().partition('=')
                if key in {'upload', 'download', 'total', 'expire'} and value.isdigit():
                    fields[key] = int(value)
            payload.update(fields)
            if {'upload', 'download', 'total', 'expire'} <= set(fields):
                used = fields['upload'] + fields['download']
                payload.update({
                    'used_bytes': used,
                    'remaining_bytes': fields['total'] - used,
                    'used_gib': round(used / 2**30, 3),
                    'remaining_gib': round((fields['total'] - used) / 2**30, 3),
                    'expire_utc': dt.datetime.fromtimestamp(fields['expire'], dt.timezone.utc).isoformat(),
                    'seconds_until_expire': fields['expire'] - int(time.time()),
                    'expired': fields['expire'] <= int(time.time()),
                })
        else:
            payload['quota_header'] = 'absent'
    except Exception as exc:
        payload['error_type'] = type(exc).__name__
        payload['error_text_class'] = str(exc).split(':', 1)[0][:100]
    print(json.dumps(payload, sort_keys=True))

controller = cfg.get('external-controller') or '127.0.0.1:9090'
if controller.startswith(':'):
    controller = '127.0.0.1' + controller
if controller.startswith('0.0.0.0:'):
    controller = '127.0.0.1:' + controller.split(':', 1)[1]
secret = cfg.get('secret') or ''
base = 'http://' + controller.rstrip('/')
headers = {'Authorization': f'Bearer {secret}'} if secret else {}

def fetch(path):
    req = urllib.request.Request(base + path, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=8) as resp:
            return resp.status, json.load(resp)
    except Exception as exc:
        return None, {'error_type': type(exc).__name__}

status, providers = fetch('/providers/proxies')
summary = {'http_status': status, 'providers': []}
provider_payload = providers.get('providers', {}) if isinstance(providers, dict) else {}
for idx, (_, provider) in enumerate(sorted(provider_payload.items()), start=1):
    proxies = provider.get('proxies') or [] if isinstance(provider, dict) else []
    alive = sum(1 for p in proxies if isinstance(p, dict) and p.get('alive') is True)
    dead = sum(1 for p in proxies if isinstance(p, dict) and p.get('alive') is False)
    unknown = len(proxies) - alive - dead
    summary['providers'].append({
        'provider_index': idx,
        'vehicle_type': provider.get('vehicleType') if isinstance(provider, dict) else None,
        'proxy_count': len(proxies),
        'alive': alive,
        'dead': dead,
        'unknown': unknown,
        'updated_at': provider.get('updatedAt') if isinstance(provider, dict) else None,
    })
if isinstance(providers, dict) and 'error_type' in providers:
    summary['error_type'] = providers['error_type']
print(json.dumps(summary, sort_keys=True))

status, version = fetch('/version')
print(json.dumps({'controller_version_http': status, 'version_present': bool(isinstance(version, dict) and version.get('version'))}, sort_keys=True))
PY

printf '=== MIHOMO_JOURNAL_COUNTS ===\n'
for pattern in 'subscription|provider' 'TLS|SSL|EOF|handshake' 'timeout|deadline' 'health.?check' 'error|failed'; do
  printf 'pattern=%s count=' "$pattern"
  journalctl -u mihomo --since '2026-08-23 12:00:00' --no-pager -o cat 2>/dev/null | grep -Eic "$pattern" || true
done
printf '%s\n' '-- sanitized recent error classes'
journalctl -u mihomo --since '2026-08-23 18:00:00' --no-pager -o cat 2>/dev/null \
  | sed -E 's#https?://[^ ]+#URL_REDACTED#g; s#([?&](token|key|auth)=)[^& ]+#\1REDACTED#g' \
  | grep -Ei 'error|failed|timeout|TLS|SSL|EOF|handshake|provider|subscription|health' \
  | tail -n 160 || true

printf '=== SYSTEM_AROUND_GRPO_EXIT ===\n'
printf '%s\n' '-- kernel/system anomaly classes 19:50-20:10 CST'
journalctl --since '2026-08-23 19:50:00' --until '2026-08-23 20:10:00' --no-pager -o short-iso 2>/dev/null \
  | grep -Ei 'oom|out of memory|killed process|segfault|NVRM|Xid|nvme|I/O error|ray|gcs|python|chenyiteng' \
  | tail -n 240 || true
printf '%s\n' '-- exact systemd/session events in exit window'
journalctl --since '2026-08-23 19:58:00' --until '2026-08-23 20:04:00' --no-pager -o short-iso 2>/dev/null \
  | grep -Ei 'session|scope|user@1003|systemd-logind|sshd|oom|kill|python|ray' \
  | tail -n 240 || true

printf 'SZ_PROXY_ADMIN_DIAGNOSIS_OK\n'
ROOT
)

sudo -S -k -p '' /bin/bash -c "$ROOT_SCRIPT"
rc=$?
sudo -k 2>/dev/null || true
exit "$rc"
