set -eu
getconf PTHREAD_KEYS_MAX || true
getconf _POSIX_THREAD_KEYS_MAX || true
ulimit -u || true
cat /proc/sys/kernel/threads-max || true
cat /proc/sys/fs/file-max || true
