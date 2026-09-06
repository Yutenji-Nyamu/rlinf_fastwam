set -u

cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
echo "ROOT_DOCS"
for file in PROJECT_CONTEXT.md HANDOFF.md; do
  if test -e "$file"; then
    ls -l "$file"
  else
    echo "ABSENT:$file"
  fi
done
echo "QAM_DOCS"
if test -d docs/rlinf-robotwin-pi0-qam; then
  find docs/rlinf-robotwin-pi0-qam -maxdepth 2 -type f -print | sort
else
  echo "ABSENT:docs/rlinf-robotwin-pi0-qam"
fi

