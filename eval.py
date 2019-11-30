import motmetrics as mm
import numpy as np


def bbox_iou(boxA, boxB):
    # format: [top left (x, y), bottom right (x, y)]
    # assuming top left is (0, 0), bottom right is (width, height)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class Target:
    def __init__(self, bbox, id, frame_num):
        self.bbox = bbox
        self.id = id
        self.frame_num = frame_num


class BboxTrackEvaluator:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
        self.accs = []
        self.names = []
        self.seq_id = 0
    

    def gen_dt_target_sequence(self, dt_result_txt_path):
        dt_sequence = []
        frame = []
        current_frame_num = 2
        with open(dt_result_txt_path) as f:
            content = f.readlines()
        for line in content:
            data_str = line.split(',')
            data = [float(x) for x in data_str[:-1]]
            frame_num = data[0]
            # txt output format: {frame},{id},{x1},{y1},{w},{h}
            id = data[1]
            bbox = data[2:6]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            # target bbox format: bottom left, top right
            new_bbox = [x1, y1, x2, y2]
            new_bbox = [x / 256 for x in new_bbox]
            if frame_num != current_frame_num:
                dt_sequence.append(frame)
                frame = []
                current_frame_num = frame_num
            target = Target(new_bbox, id, frame_num)
            frame.append(target)
        dt_sequence.append(frame)
        return dt_sequence


    def gen_gt_target_sequence(self, sequence):
        gt_sequence = []
        for i in range(len(sequence)):
            frame = []
            image_info = sequence[i]
            bboxes = image_info['bboxes']
            id = 0
            for bbox in bboxes:
                x_center, y_center, width, height = bbox
                x1 = x_center - 0.5 * width
                y1 = y_center - 0.5 * height
                x2 = x_center + 0.5 * width
                y2 = y_center + 0.5 * height
                new_bbox = [x1, y1, x2, y2]
                frame.append(Target(new_bbox, id, i+1))
                id += 1
            gt_sequence.append(frame)
        return gt_sequence


    def eval_on_sequence(self, dt_sequence, gt_sequence):
        gt_sequence = self.gen_gt_target_sequence(gt_sequence)
        acc = mm.MOTAccumulator(auto_id=True)
        dt_idx = 0
        gt_idx = 0
        while(dt_idx < len(dt_sequence) - 1):
            dt_frame = dt_sequence[dt_idx]
            gt_frame = gt_sequence[gt_idx]
            gt_target = gt_frame[0]
            dt_target = dt_frame[0]
            if (gt_target.frame_num < dt_target.frame_num):
                gt_idx += 1
            elif (gt_target.frame_num > dt_target.frame_num):
                dt_idx += 1
            num_dt = len(dt_frame)
            num_gt = len(gt_frame)
            dt_ids = [x.id for x in dt_frame]
            gt_ids = [x.id for x in gt_frame]
            dist_matrix = np.zeros((num_gt, num_dt))
            for j in range(num_gt):
                for k in range(num_dt):
                    gt_target = gt_frame[j]
                    dt_target = dt_frame[k]
                    iou = bbox_iou(gt_target.bbox, dt_target.bbox)
                    if iou < self.iou_threshold:
                        dist = np.nan
                    else:
                        dist = 1 - iou
                    dist_matrix[j, k] = dist
            acc.update(gt_ids, dt_ids, dist_matrix)
            dt_idx += 1
            gt_idx += 1
        self.accs.append(acc)
        self.names.append(f'seq_{self.seq_id}')
        self.seq_id += 1
    

    def summarize(self):
        metrics_names = ['num_frames', 'idf1', 'mota']
        print('Matching hypothesis with ground truth')
        print(f'Metrics: {metrics_names}')
        mh = mm.metrics.create()
        summary = mh.compute_many(
            self.accs, 
            metrics=metrics_names, 
            names=self.names,
            generate_overall=False
            )
        strsummary = mm.io.render_summary(
            summary, 
            formatters=mh.formatters, 
            namemap=mm.io.motchallenge_metric_names
        )
        return strsummary