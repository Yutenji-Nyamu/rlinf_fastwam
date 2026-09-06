set -euo pipefail

RUNTIME_ROOT=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime
POSTFLIGHT="$RUNTIME_ROOT/remote_ogpo_20260807_smoke_postflight_v1.sh"
EXPECTED_POSTFLIGHT_SHA=c1015c40bf406b0bece422837b1011a1180518b22f1a66b1f23a0a9434fa155c

test -s "$POSTFLIGHT"
test "$(sha256sum "$POSTFLIGHT" | awk '{print $1}')" = "$EXPECTED_POSTFLIGHT_SHA"
exec bash "$POSTFLIGHT"
