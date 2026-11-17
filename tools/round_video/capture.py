"""Record native action frames without advancing physics or adding render calls."""
import csv
from pathlib import Path


def install(cls):
    original_obs, original_step, original_serve = cls.observations, cls.step, cls.serve

    def observations(self):
        result = original_obs(self)
        if getattr(self, '_record_query', None) is not None:
            import cv2
            for slot, writer in self._video_writers.items():
                step = int(self.env.take_action_cnt[slot])
                if step <= self._video_last[slot]:
                    continue
                frame = result['main_images'][slot]
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self._video_rows[slot].writerow([self._video_counts[slot], step, self._record_query])
                self._video_streams[slot].flush()
                self._video_counts[slot] += 1
                self._video_last[slot] = step
        return result

    def step(self, request):
        import cv2
        if not hasattr(self, '_video_writers'):
            out = Path(self.cfg['action_video_output']); out.mkdir(parents=True, exist_ok=True)
            self._video_writers={}; self._video_streams={}; self._video_rows={}
            self._video_counts={}; self._video_last={}; self._video_query=0
            for slot in self.cfg.get('action_video_slots', [0,11,9]):
                h,w = self.last_obs['main_images'][slot].shape[:2]
                writer=cv2.VideoWriter(str(out/f'D-B{slot:02d}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(w,h))
                if not writer.isOpened(): raise RuntimeError('Video encoder failed')
                stream=(out/f'D-B{slot:02d}-frames.csv').open('w',newline='')
                rows=csv.writer(stream); rows.writerow(['frame','action_step','query'])
                self._video_writers[slot]=writer;self._video_streams[slot]=stream;self._video_rows[slot]=rows
                self._video_counts[slot]=0; self._video_last[slot]=-1
        self._record_query=self._video_query
        request=dict(request,capture_every_action=True)
        try:return original_step(self, request)
        finally:self._record_query=None;self._video_query+=1

    def serve(self):
        try:return original_serve(self)
        finally:
            for writer in getattr(self,'_video_writers',{}).values():writer.release()
            for stream in getattr(self,'_video_streams',{}).values():stream.close()

    cls.observations=observations;cls.step=step;cls.serve=serve
