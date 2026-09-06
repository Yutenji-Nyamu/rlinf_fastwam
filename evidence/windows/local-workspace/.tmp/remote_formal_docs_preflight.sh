set -euo pipefail

cd /root/autodl-tmp/RLinf_fastwam_rlinf
printf 'BRANCH='
git branch --show-current
printf 'HEAD='
git rev-parse HEAD
printf 'UPSTREAM='
git rev-parse '@{upstream}'
printf 'STATUS_BEGIN\n'
git status --short
printf 'STATUS_END\n'

kill -0 70062
printf 'DRIVER_ALIVE=1\n'
pgrep -c -f '^ray::EmbodiedSACFSDPPolicy' | awk '{print "ACTOR_WORKERS=" $1}'
pgrep -c -f '^ray::MultiStepRolloutWorker' | awk '{print "ROLLOUT_WORKERS=" $1}'
pgrep -c -f '^ray::EnvWorker' | awk '{print "ENV_WORKERS=" $1}'
