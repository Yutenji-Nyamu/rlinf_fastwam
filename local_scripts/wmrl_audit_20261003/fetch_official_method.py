import concurrent.futures,json,urllib.request
from pathlib import Path
OUT=Path(r'E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/official-sources')
URLS={
'issue831-comments.json':'https://api.github.com/repos/RLinf/RLinf/issues/831/comments?per_page=100',
'wmpo-coffee128.sh':'https://raw.githubusercontent.com/WM-PO/WMPO/c836d74ec6f4525c93fe980d54d0ca870118615a/examples/mimicgen/coffee/train_wmpo_128.sh',
'wmpo-core-algos.py':'https://raw.githubusercontent.com/WM-PO/WMPO/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/trainer/ppo/core_algos.py',
'wmpo-reward-inference.py':'https://raw.githubusercontent.com/WM-PO/WMPO/c836d74ec6f4525c93fe980d54d0ca870118615a/reward_model/inference_videomae.py',
'issue831-comment.json':'https://api.github.com/repos/RLinf/RLinf/issues/comments/4102817745',
'wmpo-tree.json':'https://api.github.com/repos/WM-PO/WMPO/git/trees/c836d74ec6f4525c93fe980d54d0ca870118615a?recursive=1',
'pi05-model-card.md':'https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/raw/45ccfcc4e28634f1576ebf78cab0fbe2fd82432d/README.md',
'wan-goal-model-card.md':'https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal/raw/bd395971c3467de3dd19e7e6c7562af48a2894a6/README.md',
}
def fetch(kv):
    name,url=kv
    try:
        data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'WMRL-method-audit'}),timeout=40).read()
        (OUT/name).write_bytes(data)
        return {'file':name,'url':url,'bytes':len(data)}
    except Exception as exc:return {'file':name,'url':url,'error':repr(exc)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=list(pool.map(fetch,URLS.items()))
(OUT/'fetch-manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print(json.dumps(rows,indent=2))
