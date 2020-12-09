#!/usr/bin/env python3

import os
import time
from io_utils import load_seqmap, load_detections_for_seq
from config import DATASET, RESULT_PATH, seqmap_filename, det_path, split_str
from pycocotools.mask import iou

out_path = os.path.join(RESULT_PATH, DATASET, "tracking", split_str)
seqmap = load_seqmap(seqmap_filename)
os.makedirs(out_path, exist_ok=True)

# values taken from paper


def track_iou(detections, sigma_l=0.0, sigma_h=0.9, sigma_iou=0.3, t_min=3):
    """
    Simple IOU tracker based on the Paper "High-Speed Tracking-by-Detection Without Using Image Information  by E. Bochinski, V. Eiselein, T. Sikora"
    using masked detections

    :param detections: all detections per frame
    :param sigma_l: float low detection threshold
    :param sigma_h: float, high detection threshold
    :param sigma_iou: float, IOU threshold
    :param t_min: float, minimum track length in frames
    :return: list: list of tracks
    """

    tracks_active = []
    tracks_finished = []
    for frame_num, detections_frame in detections.items():
        # apply low threshold to detections
        dets = [detection for detection in detections_frame if detection.confidence >= sigma_l and detection.class_id == 1]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                # iou_matches = [iou([track['detections'][-1].rle], [detection.rle], [0]) for detection in dets]
                # best_match = dets[iou_matches.index(max(iou_matches))]

                best_match = max(dets, key=lambda x: iou([track['detections'][-1].rle], [x.rle], [0]))
                if iou([track['detections'][-1].rle], [best_match.rle], [0]) >= sigma_iou:
                    track['detections'].append(best_match)
                    track['score'] += best_match.confidence
                    updated_tracks.append(track)
                    # remove detection from detections
                    del dets[dets.index(best_match)]
            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when condition are met
                if track['score']/len(track['detections']) >= sigma_h and len(track['detections']) >= t_min:
                    tracks_finished.append(track)
        # create new tracks
        new_tracks = [{'detections': [detection], 'score': detection.confidence, 'start': frame_num} for detection in dets]
        tracks_active = updated_tracks + new_tracks
    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if (track['score']/len(track['detections'])) >= sigma_h
                        and len(track['detections']) >= t_min]
    return tracks_finished


def foo():
    for seq_id in seqmap:
        print(seq_id)
        det_fn = os.path.join(det_path, seq_id + ".txt")
        seq_dets, _ = load_detections_for_seq(det_fn, decode_masks=False)
        out_file = os.path.join(out_path, seq_id + ".txt")
        tracks = track_iou(seq_dets)
        with open(out_file, "w") as f:
            track_id = 0
            for track in tracks:
                # avg of all confidence in that track (for now just a placeholder)
                mask_merge_confidence = track['score']/len(track['detections'])
                for det in track['detections']:
                    det_id = det.det_id
                    print(det_id, track_id, mask_merge_confidence, file=f)
                track_id += 1
