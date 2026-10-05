"""CPU client: small semantic controls and true B4/B8/B16 generation benchmarks."""
import argparse,datetime,hashlib,io,json,time,urllib.request
from pathlib import Path
import numpy as np

GIB=1024**3
FIELDS=('frames','instructions','episode_uids','end_action_indices','frame_action_indices')

def record(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')

def call(url,path,payload=None,timeout=900):
    request=urllib.request.Request(url+path,data=payload,method='GET' if payload is None else 'POST',
        headers={'Content-Type':'application/octet-stream'})
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.load(response)

def infer(url,data,indices,batch,key,out):
    request={k:data[k][indices].copy() for k in FIELDS}
    request['episode_uids']=np.asarray([f'{key}/{i}/{uid}' for i,uid in enumerate(request['episode_uids'])])
    request.update(request_id=np.asarray(key),rm_batch_size=np.asarray(batch,dtype=np.int64))
    buffer=io.BytesIO();np.savez(buffer,**request)
    start=time.monotonic();result=call(url,'/infer',buffer.getvalue());elapsed=time.monotonic()-start
    assert result['ok'] is True and result['request_id']==key
    assert len(result['items'])==len(indices)
    for item,uid,end in zip(result['items'],request['episode_uids'],request['end_action_indices']):
        assert item['episode_uid']==str(uid) and item['end_action_idx']==int(end)
    record(out/(key+'.json'),result)
    batches=result['timings']['batches']
    summary={'key':key,'samples':len(indices),'requested_batch':batch,'wall_s':elapsed,
        'clips_per_s':len(indices)/elapsed,'actual_batch_sizes':[b['size'] for b in batches],
        'peak_allocated_gib':max(b.get('peak_allocated_bytes',0) for b in batches)/GIB,
        'peak_reserved_gib':max(b.get('peak_reserved_bytes',0) for b in batches)/GIB,
        'unknown':sum(row['success'] is None for row in result['items'])}
    print(json.dumps(summary),flush=True)
    return result,summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls',nargs=2,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.parent;out.mkdir(parents=True,exist_ok=True)
    assert not args.output.exists()
    with np.load(args.samples,allow_pickle=False) as source:data={k:source[k].copy() for k in source.files}
    assert data['frames'].shape==(32,8,256,320,3)
    health=[call(url,'/health') for url in args.urls]
    assert [row['physical_gpu'] for row in health]==[4,5]
    assert all(row['ok'] and row['is_offloaded'] for row in health)
    result={'time':datetime.datetime.now().astimezone().isoformat(),'passed':False,
        'sample_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'quality_scope':'32 easy controls: 16 initial resets and 16 expert demonstrations; not native policy/WM accuracy',
        'benchmarks_use_repeated_inputs':True,'selected_rm_batch':None,'measurements':[]}
    error=None
    try:
        reference,summary=infer(args.urls[0],data,np.asarray([0,1]),1,'reference-b1',out)
        result['measurements'].append(summary)
        quality,summary=infer(args.urls[0],data,np.arange(32),4,'sanity-b4',out)
        result['measurements'].append(summary)
        labels=data['expected_success'];pred=[r['success'] for r in quality['items']]
        negatives=[p for p,label in zip(pred,labels) if label==0]
        positives=[p for p,label in zip(pred,labels) if label==1]
        result['sanity']={'negative_count':len(negatives),'false_positive':sum(x is True for x in negatives),
            'positive_count':len(positives),'true_positive':sum(x is True for x in positives),
            'unknown':sum(x is None for x in pred),'predictions':pred}
        # This only rejects gross semantic failures; it is not an accuracy certificate.
        assert result['sanity']['unknown']==0,'Unknown judgments in simple control clips'
        assert result['sanity']['false_positive']==0,'Rynn reports success on unmoved initial controls'
        assert result['sanity']['true_positive']>=12,'Rynn misses most/too many full expert demonstrations'
        assert [r['success'] for r in reference['items']]==pred[:2],'B1/B4 basic verdict mismatch'
        candidates=[]
        attempted=[]
        benchmark_indices=np.asarray([0,1]*8)
        for batch in (4,8,16):
            if batch==16:
                prior=next(row for row in attempted if row['requested_batch']==8)
                projected=17.84+2*max(0,prior['peak_allocated_gib']-17.84)+4
                if projected>50:
                    result['b16_skipped']={'reason':'predicted Rynn peak leaves too little rollout coexistence margin','predicted_gib':projected}
                    break
            response,summary=infer(args.urls[0],data,benchmark_indices,batch,f'throughput-b{batch}',out)
            result['measurements'].append(summary)
            attempted.append(summary)
            assert batch in summary['actual_batch_sizes'],'Requested batch not exercised inside generation'
            assert [r['success'] for r in response['items']]==[pred[i] for i in benchmark_indices],'Verdict differs across batch size'
            if summary['unknown']==0 and summary['peak_reserved_gib']<=50:
                candidates.append(summary)
        assert candidates,'No suitable RM batch within the reserved-memory budget'
        selected=max(candidates,key=lambda row:row['clips_per_s'])['requested_batch']
        result['selected_rm_batch']=selected
        # Actual other-card service and a short tail; no extra model process.
        response,summary=infer(args.urls[1],data,np.asarray([0,1]*8),selected,'second-service',out)
        result['measurements'].append(summary)
        assert [r['success'] for r in response['items']]==[pred[i] for i in [0,1]*8]
        assert summary['peak_reserved_gib']<=50
        response,summary=infer(args.urls[0],data,np.asarray([1,0,1,0,1,0,1]),selected,'tail-order',out)
        result['measurements'].append(summary)
        assert [r['success'] for r in response['items']]==[pred[i] for i in [1,0,1,0,1,0,1]]
        result['passed']=True
    except BaseException as exc:
        result['error']={'type':type(exc).__name__,'message':str(exc)};error=exc
    finally:
        result['offload']=[]
        for url in args.urls:
            try:
                released=call(url,'/offload',b'');result['offload'].append(released)
                assert released['ok'] and released['is_offloaded']
            except BaseException as exc:
                result['passed']=False
                result['offload'].append({'error':str(exc)})
                error=error or exc
        record(args.output,result)
    print(json.dumps({'passed':result['passed'],'selected_rm_batch':result['selected_rm_batch'],'output':str(args.output),'sanity':result.get('sanity'),'error':result.get('error')}),flush=True)
    if error:raise error

if __name__=='__main__':main()
