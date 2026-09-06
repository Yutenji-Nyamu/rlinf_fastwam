set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin

cd "$target"
remote_url="$(git config --get remote.personal.url)"
case "$remote_url" in
  https://*) printf 'REMOTE_PROTOCOL=https\n' ;;
  http://*) printf 'REMOTE_PROTOCOL=http\n' ;;
  ssh://*) printf 'REMOTE_PROTOCOL=ssh\n' ;;
  git@*:*) printf 'REMOTE_PROTOCOL=scp-ssh\n' ;;
  *) printf 'REMOTE_PROTOCOL=other\n' ;;
esac
if git config --get http.proxy >/dev/null; then
  printf 'GIT_HTTP_PROXY_CONFIGURED=1\n'
else
  printf 'GIT_HTTP_PROXY_CONFIGURED=0\n'
fi
if test -n "${HTTPS_PROXY:-}${https_proxy:-}"; then
  printf 'ENV_HTTPS_PROXY_CONFIGURED=1\n'
else
  printf 'ENV_HTTPS_PROXY_CONFIGURED=0\n'
fi
if test -n "${ALL_PROXY:-}${all_proxy:-}"; then
  printf 'ENV_ALL_PROXY_CONFIGURED=1\n'
else
  printf 'ENV_ALL_PROXY_CONFIGURED=0\n'
fi
