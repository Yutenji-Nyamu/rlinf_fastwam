set -eu

RAY=/data/chenyiteng/ray/rlt-dsrl-v3/session_2026-08-23_16-27-53_911161_321906/logs
R0=$RAY/old/worker-d734f6621f783c0b0749e732d56215132e018144a6c7e667c94dad18-62010000-3590591.err
R1=$RAY/old/worker-8c350d1c84d85e825575e227bf3008847ae55d7d11dc74436295a92c-62010000-3590594.err
RO=$RAY/old/worker-9b658068e8c34ac66b2ba752203aba5b227f3e42c3a4abc3aade5d11-62010000-3590534.err

echo '=== RANK0 ERR HEAD/TAIL ==='
head -n 25 "$R0"
tail -n 25 "$R0"

echo '=== RANK1 ERR HEAD/TAIL ==='
head -n 25 "$R1"
tail -n 30 "$R1"

echo '=== ROLLOUT ERR TAIL ==='
tail -n 80 "$RO"

echo '=== RAYLET 06:52 FAILURE LINES ==='
grep -nE '2026-09-02 06:52|d734f6621f783c0b0749e732d56215132e018144a6c7e667c94dad18|8c350d1c84d85e825575e227bf3008847ae55d7d11dc74436295a92c' "$RAY/raylet.out" 2>/dev/null | grep -v 'state-dump' | tail -n 120 || true

echo '=== ERROR COUNTS ==='
printf 'rank0 invalid='; grep -c 'OIDN Error: invalid handle' "$R0" || true
printf 'rank0 pthread='; grep -c 'OIDN Error: pthread_key_create failed' "$R0" || true
printf 'rank1 invalid='; grep -c 'OIDN Error: invalid handle' "$R1" || true
printf 'rank1 pthread='; grep -c 'OIDN Error: pthread_key_create failed' "$R1" || true
printf 'rank0 python thread headers='; grep -c '^Thread 0x' "$R0" || true
printf 'rank0 close_env stacks='; grep -c ' in close_env$' "$R0" || true
printf 'rank0 vector reset worker stacks='; grep -c 'vector_env.py.*, line 147 in reset' "$R0" || true
