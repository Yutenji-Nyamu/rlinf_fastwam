set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

repo=/root/autodl-tmp/RLinf_fastwam_rlinf
expected_head_prefix=8138d670
branch=codex/dsrl-pi0-robotwin

cd "$repo"
case "$(git rev-parse HEAD)" in
  "$expected_head_prefix"*) ;;
  *) echo "unexpected HEAD" >&2; exit 1 ;;
esac
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git switch -c "$branch"
git status --short --branch --untracked-files=all
gh --version | sed -n '1p'
gh auth status --hostname github.com
