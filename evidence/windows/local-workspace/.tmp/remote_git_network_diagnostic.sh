set -u

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
cd "$REPO" || exit 1

echo "REMOTES"
git remote -v
echo "GIT_PROXY_CONFIG"
git config --show-origin --get-regexp '^(http|https)\..*proxy$|^http\.proxy$|^https\.proxy$' || true
echo "ENV_PROXY_NAMES"
env | sed -n 's/^\([^=]*[Pp][Rr][Oo][Xx][Yy][^=]*\)=.*/\1=<set>/p'
echo "DNS"
getent ahostsv4 github.com | head -n 4 || true
echo "TCP443"
timeout 10s bash -c '</dev/tcp/github.com/443' && echo "TCP443=PASS" || echo "TCP443=FAIL"
