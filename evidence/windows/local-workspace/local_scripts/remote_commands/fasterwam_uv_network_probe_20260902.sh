set -euo pipefail

torch_url='https://download-r2.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl'
embreex_url='https://files.pythonhosted.org/packages/11/f5/460b7f79689ac5e6ceb3ec2a1194176a0a66d6c4e010dae68ba899a1c927/embreex-4.4.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

probe() {
  label="$1"
  url="$2"
  proxy="$3"
  printf '%s %s\n' "$label" "$url"
  if test "$proxy" = direct; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl -L --fail --silent --show-error --max-time 25 --range 0-4194303 \
      --output /dev/null --write-out 'status=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' "$url" || true
  else
    HTTPS_PROXY=http://127.0.0.1:7890 \
      curl -L --fail --silent --show-error --max-time 25 --range 0-4194303 \
      --output /dev/null --write-out 'status=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' "$url" || true
  fi
}

probe TORCH_DIRECT "$torch_url" direct
probe TORCH_PROXY "$torch_url" proxy
probe PYPI_DIRECT "$embreex_url" direct
probe PYPI_PROXY "$embreex_url" proxy
