import time
import numpy as np
import cv2 as cv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import imgproc, helper
from altusi.logger import Logger
from altusi.videos import WebcamVideoStream
from altusi.misc import read_py_config
from altusi.visualization import visualize_multicam_detections

from mc_tracker.mct import MultiCameraTracker
from mc_tracker.sct import SingleCameraTracker

from altusi.objectdetector import ObjectDetector
from altusi.personembedder import PersonEmbedder


LOG = Logger('app-face-detector')

class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)
                
def app(video_link, video_name, show, record, flip_hor, flip_ver):
    # initialize Face Detection net
    config = read_py_config('config.py')
    object_detector = ObjectDetector()
    reid =  PersonEmbedder()

    # initialize Video Capturer
    cap = MulticamCapture(args.i)
    #cap = WebcamVideoStream(src=video_link).start()
    # (W, H), FPS = imgproc.cameraCalibrate(cap)
    # LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, FPS))
    tracker = MultiCameraTracker(cap.get_num_sources(), reid, **config)
    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()
    
    if record:
        time_str = time.strftime(cfg.TIME_FM)
        writer = cv.VideoWriter(video_name+time_str+'.avi',
                cv.VideoWriter_fourcc(*'XVID'), 20, (1280, 720))

    cnt_frm = 0
    counter=0
    while thread_body.process:
        try:
            frm = thread_body.frames_queue.get_nowait()
        except queue.Empty:
            frm = None
        if frm is None:
            continue
        cnt_frm += 1

        # if flip_ver: frm = cv.flip(frm, 0)
        # if flip_hor: frm = cv.flip(frm, 1)
        _start_t = time.time()
        all_detections=[]
        for f in frm:
            frm = imgproc.resizeByHeight(f, 640)
            scores, bboxes = object_detector.getObjects(f, def_score=0.5)
            all_detections.append(bboxes)

        tracker.process(frm, all_detections, [[]])
        tracked_objects = tracker.get_tracked_objects()
        _prx_t = time.time() - _start_t
        fps = round(1 / _prx_t, 1)
        if len(bboxes):
            frm = visualize_multicam_detections(frm,tracked_objects, fps)
        frm = vis.plotInfo(frm, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
        frm = cv.cvtColor(np.asarray(frm), cv.COLOR_BGR2RGB)

        if record:
            writer.write(frm)
       
        if show:
            cv.imshow(video_name, frm)
            key = cv.waitKey(1)
            if key in [27, ord('q')]:
                LOG.info('Interrupted by Users')
                break
        else:
            if(counter%10==0):
                print(f"IN : {SingleCameraTracker.COUNT_IN}, OUT: {SingleCameraTracker.COUNT_OUT}")
        counter+=1

    if record:
        writer.release()
    cap.release()
    cv.destroyAllWindows()


def main(args):
    video_link = args.video if args.video else 0
    app(video_link, args.name, args.show, args.record, args.flip_hor, args.flip_ver)


if __name__ == '__main__':
    LOG.info('Raspberry Pi: Object Detection')

    args = helper.getArgs()
    main(args)

    LOG.info('Process done')
