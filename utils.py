import os
import numpy as np
import glob
import re
import torch.nn as nn
import torch



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class EarlyStop:
    def __init__(self, patience: int, verbose: bool = True):
        self.patience = patience
        self.init_patience = patience
        self.verbose = verbose
        self.lowest_loss = 9999999.999
        self.highest_acc = -0.1

    def state_dict(self):
        return {
            'patience': self.patience,
            'init_patience': self.init_patience,
            'verbose': self.verbose,
            'lowest_loss': self.lowest_loss,
            'highest_acc': self.highest_acc,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.init_patience = state_dict['init_patience']
        self.verbose = state_dict['verbose']
        self.lowest_loss = state_dict['lowest_loss']
        self.highest_acc = state_dict['highest_acc']

    def step(self, loss=None, acc=None, criterion=lambda x1, x2: x1 or x2):
        if loss is None:
            loss = self.lowest_loss
            better_loss = True
        else:
            better_loss = (loss < self.lowest_loss) and ((self.lowest_loss - loss) / self.lowest_loss > 0.01)
        if acc is None:
            acc = self.highest_acc
            better_acc = True
        else:
            better_acc = acc > self.highest_acc

        if better_loss:
            self.lowest_loss = loss
        if better_acc:
            self.highest_acc = acc

        if criterion(better_loss, better_acc):
            self.patience = self.init_patience
            if self.verbose:
                print('Remaining patience: {}'.format(self.patience))
            return False
        else:
            self.patience -= 1
            if self.verbose:
                print('Remaining patience: {}'.format(self.patience))
            if self.patience < 0:
                if self.verbose:
                    print('Ran out of patience.')
                return True


################## Metrics
def get_labels_start_end_time(frame_wise_labels, bg_class):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], 'float')
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, bg_class):
    norm = True
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def recog_file(filename, ground_truth_path, overlap, background_class_list):
    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/', '/', filename)
    with open(gt_file, 'r') as f:
        gt_content = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  
        f.close()

    n_frame_correct = 0
    for i in range(len(recog_content)):
        if recog_content[i] == gt_content[i]:
            n_frame_correct += 1

    edit_score_value = edit_score(recog_content, gt_content, background_class_list)

    tp_arr = []
    fp_arr = []
    fn_arr = []
    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], background_class_list)
        tp_arr.append(tp1)
        fp_arr.append(fp1)
        fn_arr.append(fn1)
    return n_frame_correct, len(recog_content), tp_arr, fp_arr, fn_arr, edit_score_value


def calculate_mof(ground_truth_path_name, prediction_path, background_class):
    overlap = [.1, .25, .5]
    overlap_scores = np.zeros(3)
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit = 0
    n_frames = 0
    n_correct = 0

    filelist = glob.glob(prediction_path + '/*txt')

    if len(filelist) == 0:
        return 0, 0, overlap_scores
    # loop over all recognition files and evaluate the frame error
    for filename in filelist:
        correct, frames, tp_arr, fp_arr, fn_arr, edit_score_value = recog_file(filename, ground_truth_path_name,
                                                                               overlap, background_class)
        n_correct += correct
        n_frames += frames
        edit += edit_score_value

        for i in range(len(overlap)):
            tp[i] += tp_arr[i]
            fp[i] += fp_arr[i]
            fn[i] += fn_arr[i]

    if n_correct == 0 or n_frames == 0:
        acc = 0
    else:
        acc = float(n_correct) * 100.0 / n_frames

    final_edit_score = ((1.0 * edit) / len(filelist))

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        overlap_scores[s] = f1

    return final_edit_score, acc, overlap_scores


################### post_process
class PostProcess(nn.Module):
    def __init__(self, args, label_dict, actions_dict, gt_path):
        super().__init__()
        self.labels_dict_id2name = label_dict
        self.labels_dict_name2id = actions_dict

        self.results_dict = dict()
        self.chunk_size = args.chunk_size
        self.gd_path = gt_path
        self.count = 0

    def start(self):
        self.results_dict = dict()
        self.count = 0

    def dump_to_directory(self, path, suffix='_gt'):
        print("Number of cats =", self.count, ", Number of videos = ", len(self.results_dict))
        if len(self.results_dict.items()) == 0:
            return
        for video_id, video_value in self.results_dict.items():
            pred_value = video_value[0].detach().cpu().numpy()
            label_count = video_value[1].detach().cpu().numpy()
            label_send = video_value[2].detach().cpu().numpy()
            #print('video_id:', video_id)
            name = video_id.split('%')[0]
            video_path = os.path.join(self.gd_path, name + ".txt")
            recog_content = []
            with open(video_path.replace('assembly_', ''), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    tmp = l.split('\t')
                    start_l, end_l, label_l = int(tmp[0]), int(tmp[1]), tmp[2]
                    recog_content.extend([label_l] * (end_l - start_l))

            label_name_send = [self.labels_dict_id2name[i.item()] for i in label_send]
            #
            # print('label_name_send:', label_name_send)

            recog_content = label_name_send
            
            label_name_arr = [self.labels_dict_id2name[i.item()] for i in pred_value[:label_count.item()]]
            new_label_name_expanded = []  
            for i, ele in enumerate(label_name_arr):
                st = i * self.chunk_size
                end = st + self.chunk_size
                if end > len(recog_content):
                    end = len(recog_content)
                for j in range(st, end):
                    new_label_name_expanded.append(ele)
                if len(new_label_name_expanded) >= len(recog_content):
                    break

            out_path = os.path.join(path, video_id.replace("%", '_') + ".txt")
            with open(out_path, "w") as fp:
                fp.write("\n".join(new_label_name_expanded))
                fp.write("\n")
            out_path1 = os.path.join(path + suffix, video_id.replace("%", '_') + ".txt")
            with open(out_path1, "w") as fp:
                fp.write("\n".join(recog_content[:len(new_label_name_expanded)]))
                fp.write("\n")

    @torch.no_grad()
    def forward(self, outputs, video_names, framewise_labels, counts):
        """ Perform the computation
        Parameters:
            :param outputs: raw outputs of the model
            :param start_frame:
            :param video_names:
            :param clip_length:
        """
        for output, vn, framewise_label, count in zip(outputs, video_names, framewise_labels, counts):
            output_video = torch.argmax(output, 0)
            if vn in self.results_dict:
                self.count += 1

                prev_tensor, prev_count, prev_gt_labels = self.results_dict[vn]
                output_video = torch.cat([prev_tensor, output_video])
                framewise_label = torch.cat([prev_gt_labels, framewise_label])
                count = count + prev_count

            self.results_dict[vn] = [output_video, count, framewise_label]


class PostProcess_test(nn.Module):
    def __init__(self, weights, label_dict, actions_dict, gt_path):
        super().__init__()
        self.labels_dict_id2name = label_dict
        self.labels_dict_name2id = actions_dict
        self.results_dict = dict()
        self.results_json = None
        self.count = 0
        self.gd_path = gt_path
        self.acc_dict = dict()
        self.weights = weights

    def start(self):
        self.results_dict = dict()
        self.count = 0

    def get_acc_dict(self):
        return self.acc_dict

    def upsample_video_value(self, predictions, video_len, chunk_size):
        new_label_name_expanded = []
        prediction_swap = predictions.permute(1, 0)
        for i, ele in enumerate(prediction_swap):
            st = i * chunk_size
            end = st + chunk_size
            for j in range(st, end):
                new_label_name_expanded.append(ele)
        out_p = torch.stack(new_label_name_expanded).permute(1, 0)[:, :video_len]
        return out_p

    def accumulate_result(self, all_pred_value):
        sum_ac = 0
        for wt, pred_v in zip(self.weights, all_pred_value):
            sum_ac = sum_ac + (wt * pred_v)

        return torch.argmax(sum_ac / sum(self.weights), dim=0)

    def dump_to_directory(self, path, suffix='_gt'):
        print("Number of cats =", self.count)
        if len(self.results_dict.items()) == 0:
            return
        prev_vid_id = None
        all_pred_value = None
        ne_dict = {}
        video_id = None

        for video_chunk_id, video_value in self.results_dict.items():
            video_id, chunk_id = video_chunk_id.split("@")[0], video_chunk_id.split("@")[1]
            #print('video_value:', video_value)
            assert video_value[1] * video_value[3] >= video_value[4]
            upped_pred_logit = self.upsample_video_value(video_value[0][:, :video_value[1]],
                                                         video_value[4], video_value[3]).unsqueeze(0)
            if video_id == prev_vid_id:
                all_pred_value = torch.cat([all_pred_value, upped_pred_logit], dim=0)
            else:
                if all_pred_value is not None:
                    ne_dict[prev_vid_id] = self.accumulate_result(all_pred_value)
                    all_pred_value = None
                prev_vid_id = video_id
                all_pred_value = upped_pred_logit  # With refinement softmax has to be added

        if all_pred_value is not None:
            ne_dict[video_id] = self.accumulate_result(all_pred_value)

        label_send = []
        for video_id, video_value in ne_dict.items():
            pred_value = video_value.detach().cpu().numpy()
            label_name_arr = [self.labels_dict_id2name[i.item()] for i in pred_value]

            label_send.append(video_value[2]) #

            name = video_id.split('%')[0]
            video_path = os.path.join(self.gd_path, name + ".txt")
            recog_content = []
            with open(video_path.replace('assembly_', ''), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    tmp = l.split('\t')
                    start_l, end_l, label_l = int(tmp[0]), int(tmp[1]), tmp[2]
                    recog_content.extend([label_l] * (end_l - start_l))
            
            recog_content = [x for x in recog_content if x != 'Background']

            out_path = os.path.join(path, video_id.replace("%", '_') + ".txt")
            with open(out_path, "w") as fp:
                fp.write("\n".join(label_name_arr))
                fp.write("\n")

            out_path1 = os.path.join(path + suffix, video_id.replace("%", '_') + ".txt")
            if not os.path.exists(path + suffix):
                os.makedirs(path + suffix)
            with open(out_path1, "w") as fp:
                fp.write("\n".join(recog_content[:len(label_name_arr)]))
                fp.write("\n")

    @torch.no_grad()
    def forward(self, outputs, video_names, framewise_labels, counts, chunk_size_arr, chunk_id_arr, vid_len_arr):
        

        for output, vn, framewise_label, count, chunk_size, chunk_id, vid_len in zip(outputs, video_names,
                                                                                     framewise_labels,
                                                                                     counts, chunk_size_arr,
                                                                                     chunk_id_arr, vid_len_arr):
            key = '{}@{}'.format(vn, chunk_id)

            if key in self.results_dict:
                self.count += 1

                prev_tensor, prev_count, prev_gt_labels, chunk_size, pre_vid_len = self.results_dict[key]
                output = torch.cat([prev_tensor, output], dim=1)
                framewise_label = torch.cat([prev_gt_labels, framewise_label])
                count = count + prev_count
                vid_len = vid_len + pre_vid_len

            self.results_dict[key] = [output, count, framewise_label, chunk_size, vid_len]
