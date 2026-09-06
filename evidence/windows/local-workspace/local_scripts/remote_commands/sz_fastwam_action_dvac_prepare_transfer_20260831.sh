set -euo pipefail
install -d -m 700 /data/chenyiteng/tmp
test -d /data/chenyiteng/tmp
stat -c 'transfer_dir=%n mode=%a owner=%U' /data/chenyiteng/tmp
