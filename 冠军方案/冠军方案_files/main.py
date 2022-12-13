import sys
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import shutil
SRC_PATH = r'../input/nfl2solution'
sys.path.insert(1, SRC_PATH)

# Read in data files
BASE_DIR = '../input/nfl-health-and-safety-helmet-assignment'

# Labels and sample submission
labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
ss = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')

# Player tracking data
tr_tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
te_tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')

# Baseline helmet detection labels
tr_helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
te_helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')

# Extra image labels
img_labels = pd.read_csv(f'{BASE_DIR}/image_labels.csv')


def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
        "timedelta64[ms]"
    ) / 1_000
    # Estimated video frame
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks

tr_tracking = add_track_features(tr_tracking)
te_tracking = add_track_features(te_tracking)


# 根据视频帧在est_frame与next_est_frame中的相对位置, 计算inter_x和inter_y
# I_____________________________I___________________________________________________I
# est_frame                     frame                                               next_est_frame
# x, y                          inter_x, inter_y                                    next_x, next_y
# inter_x = x * (next_est_frame - frame) / (next_est_frame - est_frame) + \
#           next_x * (frame - est_frame) / (next_est_frame - est_frame)
# inter_y = y * (next_est_frame - frame) / (next_est_frame - est_frame) + \
#           next_y * (frame - est_frame) / (next_est_frame - est_frame)
def make_interpolated_tracking(df_tracking, df_helmet):
    df_ref_play_frame = pd.DataFrame(df_helmet["video_frame"].unique())[0].str.rsplit(
        '_', n=2, expand=True).rename(columns={0: 'game_play', 1: 'view', 2: "frame"}).drop("view",
                                                                                            axis=1).drop_duplicates()
    df_ref_play_frame["frame"] = df_ref_play_frame["frame"].astype('int')
    df_ref_play_frame = df_ref_play_frame.sort_values(['game_play', "frame"])

    df_list = []

    for keys, _df_tracking in tqdm(df_tracking.groupby(["player", "game_play"])):
        # skip because there are sideline player
        if keys[0] == "H00" or keys[0] == "V00":
            continue
        _df_ref_play_frame = df_ref_play_frame[df_ref_play_frame["game_play"] == keys[1]].copy()
        _df_ref_play_frame = _df_ref_play_frame.drop("game_play", axis=1)

        _df_tracking = _df_tracking.sort_values("est_frame")
        _df_tracking_copy = _df_tracking[["est_frame", "x", "y"]].copy().rename(
            columns={"est_frame": "next_est_frame", "x": "next_x", "y": "next_y"}).shift(-1).interpolate()
        _df_tracking_copy.iloc[-1, 0] += 1
        _df_tracking = pd.concat([_df_tracking, _df_tracking_copy], axis=1)

        # merge with frame and est_frame
        merged_df = pd.merge_asof(
            _df_ref_play_frame.copy(),
            _df_tracking,
            left_on="frame",
            right_on="est_frame",
            direction="backward",  # 'nearest',
        )
        df_list.append(merged_df)

    all_merged_df = pd.concat(df_list)
    w_1 = all_merged_df[["x", "y"]].values * ((all_merged_df["next_est_frame"].values - all_merged_df[
        "frame"].values) / (all_merged_df["next_est_frame"].values - all_merged_df["est_frame"].values))[:, np.newaxis]
    w_2 = all_merged_df[["next_x", "next_y"]].values * ((all_merged_df["frame"].values - all_merged_df[
        "est_frame"].values) / (all_merged_df["next_est_frame"].values - all_merged_df["est_frame"].values))[:,
                                                       np.newaxis]
    all_merged_df["x_interp"] = w_1[:, 0] + w_2[:, 0]
    all_merged_df["y_interp"] = w_1[:, 1] + w_2[:, 1]
    all_merged_df = all_merged_df.drop(["next_est_frame", "next_x", "next_y"], axis=1)
    return all_merged_df


print("preparing interpolated dataset")
te_tracking = make_interpolated_tracking(te_tracking, te_helmets)
tr_tracking = make_interpolated_tracking(tr_tracking, tr_helmets)

import os
import glob
import json
import warnings
import argparse
import random
import time
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from input.nfl2solution.model.model import build_model_team, build_model_map
from input.nfl2solution.model.model_detection import build_detection_model, soft_nms_layer
from input.nfl2solution.model.ICP_tf_team_batch import make_icp_inputs, random_icp_fitting, transform_points
from input.nfl2solution.model.ICP_tf_team_batch import get_nearest_distance, search_nearest_error, points2points_fitting
from input.nfl2solution.model.Tracker import Tracker_2_w_feature, wbf_ensemble_reassign_player_label
from input.nfl2solution.model.similaritymat_to_binary import similarity_matrix_to_team


# from train_utils.evaluation import NFLAssignmentScorer
# from train_utils.extract_img_from_video_subframes import make_rectangle, make_locations


class Batch_data():
    def __init__(self):
        self.imgs = []
        self.img_arrays = []
        self.frames = []
        self.length = 0

    def add(self, img, img_array, frame):
        self.imgs.append(img)
        self.img_arrays.append(img_array)
        self.frames.append(frame)
        self.length = len(self.frames)

    def take(self):
        pass

    def reset(self):
        self.imgs = []
        self.img_arrays = []
        self.frames = []


class HelmetSizeHist():
    '''
    当self.count <= self.update_freq时, 就和平常的求全局均值一样
    当self.count >  self.update_freq时, 会弱化self.mean的重要性, 增加求全局均值时mean_size的重要性
    '''

    def __init__(self, update_freq=20):
        self.update_freq = update_freq
        self.mean = tf.Variable(20.0)  # , dtype='float32')
        self.count = tf.Variable(0)  # , dtype='int32')

    def reset(self):
        self.mean.assign(20.0)
        self.count.assign(0)

    def update_and_get_current(self, mean_size, num_detection):
        mean = self.mean
        count = self.count
        if num_detection > 0:
            c = tf.minimum(count, self.update_freq)
            total = mean * tf.cast(c, tf.float32) + mean_size * tf.cast(num_detection, tf.float32)
            updated_mean = total / tf.cast(c + num_detection, tf.float32)
            updated_count = count + num_detection
            self.mean.assign(updated_mean)
            self.count.assign(updated_count)
        else:
            updated_mean = mean
        return updated_mean

    def get_current_state(self):
        return self.mean  # .numpy(), self.count.numpy()


class TeamFeaturesHolder():
    def __init__(self, num_features, update_freq=100, add_threshold=0.2, softmax_temperature=2.):
        self.team_h_mean = np.zeros((num_features,))
        self.team_v_mean = np.zeros((num_features,))
        self.count_h = 0
        self.count_v = 0
        # self.start_from = update_freq
        self.update_freq = update_freq
        self.add_threshold = add_threshold
        self.softmax_temperature = softmax_temperature
        self.provided = False

    def add(self, pred_label, assign_label, features):
        if self.provided:
            survive_mask = (np.abs(assign_label - pred_label) < self.add_threshold)
        else:
            survive_mask = (np.abs(assign_label - pred_label) < self.add_threshold * 2)
        pred_label = pred_label[survive_mask]
        assign_label = assign_label[survive_mask]
        features = features[survive_mask]
        h_features = features[assign_label > 0.5]
        v_features = features[assign_label < 0.5]
        num_h = len(h_features)
        num_v = len(v_features)
        if num_h > 0:
            if self.count_h < self.update_freq:
                total_h = self.team_h_mean * self.count_h + np.sum(h_features, axis=0)
                self.team_h_mean = total_h / (self.count_h + num_h)
            else:
                total_h = self.team_h_mean * self.update_freq + np.sum(h_features, axis=0)
                self.team_h_mean = total_h / (self.update_freq + num_h)
        if num_v > 0:
            if self.count_v < self.update_freq:
                total_v = self.team_v_mean * self.count_v + np.sum(v_features, axis=0)
                self.team_v_mean = total_v / (self.count_v + num_v)
            else:
                total_v = self.team_v_mean * self.update_freq + np.sum(v_features, axis=0)
                self.team_v_mean = total_v / (self.update_freq + num_v)
        self.count_h += num_h
        self.count_v += num_v
        # print("COUNTER", self.count_h, self.count_v)

    def predict(self, pred_features):
        # 通过新传入的运动员128维特征，分别与h队、v队的平均特征计算内积h_sim, v_sim
        # softmax 计算属于h队的概率
        if (self.count_h >= self.update_freq) and (self.count_v >= self.update_freq):
            h_sim = np.dot(pred_features, self.team_h_mean) * self.softmax_temperature
            v_sim = np.dot(pred_features, self.team_v_mean) * self.softmax_temperature

            pred_binary = np.clip(np.exp(h_sim) / (np.exp(h_sim) + np.exp(v_sim)), 0.0, 1.0)
            self.provided = True
        else:
            pred_binary = None

        return pred_binary


class NFL_Predictor():
    def __init__(self,  # num_classes=30, solo_score_thresh=0.3,
                 input_shape=(288, 288, 4),  # (512, 896, 3)
                 output_shape=(144, 144),  # (128, 224)
                 weight_file=None,  # {...}
                 is_train_model=False,
                 inference_batch=1):

        print("\rLoading Models...", end="")

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.is_train_model = is_train_model
        self.weight_file = weight_file
        self.inference_batch = inference_batch
        if inference_batch >= 1:
            self.batch_run = True
        else:
            self.batch_run = False
        self.load_model(weight_file, is_train_model)
        print("Loading Models......Finish")

    def load_model(self, weight_file=None, is_train_model=False):
        """build model and load weights，构建模型并加载其权重"""
        train_map_model_s, map_model_s, _, _, _ = build_model_map((512, 896, 3),  # self.input_shape,
                                                                  minimum_stride=self.input_shape[0] //
                                                                                 self.output_shape[0],
                                                                  is_train=self.is_train_model,
                                                                  backbone="effv2s",
                                                                  from_scratch=True,
                                                                  )
        train_map_model_s.load_weights(weight_file["map"])  # , by_name=True)
        map_model = [map_model_s]

        det_model, _, _ = build_detection_model((704, 1280, 3),
                                                minimum_stride=2,
                                                is_train=self.is_train_model,
                                                backbone="effv2s",
                                                from_scratch=True,
                                                include_nms=False if self.batch_run else True)
        det_model.load_weights(weight_file["det"])

        l_det_model = []
        for backbone, file in weight_file["detL"]:
            l_det_model_smlxl, _, _ = build_detection_model((None, None, 3),
                                                            minimum_stride=2,
                                                            is_train=self.is_train_model,
                                                            backbone=backbone,
                                                            from_scratch=True,
                                                            include_nms=False if self.batch_run else True)
            l_det_model_smlxl.load_weights(file)
            l_det_model.append(l_det_model_smlxl)

        self.num_det_models = len(l_det_model)

        team_model, _, _, _ = build_model_team(input_shape=(96 + 32, 64 + 32, 3),
                                               # input_shape_view=(256+64,448+128,3),
                                               # minimum_stride=4,
                                               backbone="effv2s",
                                               is_train=self.is_train_model,
                                               from_scratch=True, )
        team_model.load_weights(weight_file["team"])

        self.mapper, self.detector, registrator_side, registrator_end, self.prelocate = self.get_integrated_inference_model(
            map_model,
            det_model,
            l_det_model,
            team_model)

        self.registrator = {"Sideline": registrator_side,
                            "Endzone": registrator_end}

    def get_integrated_inference_model(self, map_model, det_model, l_det_model=None, team_model=None):
        # 输入这些模型的网络结构及权重，得到推断单张图片全局坐标、检测框位置、team相关信息的句柄
        self.hsh = HelmetSizeHist(update_freq=20)  # 只在检测的时候用到了
        self.hsh_select = HelmetSizeHist(update_freq=20) # 只在检测的时候用到了

        self.map_shape = [512, 896]
        self.det_shape = [704, 1280]
        self.player_shape = [96, 64]

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, 720, 1280, 3], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                                      )
                     )
        def mapper(input_rgb, box_tlbr):
            # 没用到
            map_img = tf.image.resize(input_rgb, (self.map_shape[0], self.map_shape[1]), method="bilinear")

            box_tlbr = box_tlbr / tf.constant([[[720, 1280, 720, 1280]]], tf.float32)

            mean_size = self.hsh.get_current_state()
            size_input = tf.stack([[mean_size]])
            use_TTA = False
            if use_TTA:
                f_t = box_tlbr[:, :, :1]
                f_l = 1. - box_tlbr[:, :, 3:4]
                f_b = box_tlbr[:, :, 2:3]
                f_r = 1. - box_tlbr[:, :, 1:2]
                f_box_tlbr = tf.concat([f_t, f_l, f_b, f_r], axis=-1)

                map_img_batch = tf.concat([map_img, map_img[:, :, ::-1, :]], axis=0)
                input_img_batch = tf.concat([input_rgb, input_rgb[:, :, ::-1, :]], axis=0)
                box_tlbr_batch = tf.concat([box_tlbr, f_box_tlbr], axis=0)

                preds = map_model([map_img_batch, box_tlbr_batch])
                pred_location = (preds[0][0:1] + preds[0][1:2] * tf.constant([[[-1, 1]]], tf.float32)) / 2.0
                pred_team = team_model([input_img_batch, box_tlbr_batch])
                pred_simmat = (pred_team[0][0:1, ..., 0] + pred_team[0][1:2, ..., 0]) / 2.0
                pred_teamvec = (pred_team[1][0:1, :, :] + pred_team[1][1:2, :, :]) / 2.0

            else:
                preds = map_model([map_img, box_tlbr, size_input])
                pred_location = preds[0]
                pred_team = team_model([input_rgb, box_tlbr, size_input])
                pred_simmat = pred_team[0][..., 0]
                pred_teamvec = pred_team[1]

            return pred_location, pred_simmat, pred_teamvec

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, 720, 1280, 3], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                                      )
                     )
        def mapper_ensemble(input_rgb, box_tlbr):
            # 输入一张图片及其bbox坐标，输出该图片中各个bbox的全局坐标和128维特征向量，以及该图片各个bbox构成的相似度矩阵
            map_img = tf.image.resize(input_rgb, (self.map_shape[0], self.map_shape[1]), method="bilinear")
            box_tlbr = box_tlbr / tf.constant([[[720, 1280, 720, 1280]]], tf.float32)  # 以图像的宽和高为基准分别归一化
            mean_size = self.hsh_select.get_current_state() / tf.sqrt(720. * 1280.)  # 初始约为0.0208
            size_input = mean_size * tf.ones((tf.shape(map_img)[0], 1),
                                             tf.float32)  # tf.tile(tf.stack([[mean_size]]), [tf.shape(map_img)[0], 1])

            list_pred_location = []
            list_pred_simmat = []
            list_pred_teamvec = []
            for m in map_model:
                # map_img: (None, 512, 896, 3)  box_tlbr: (None, None, 4)  size_input: (None, 1)  input_rgb: (None, 720, 1280, 3)
                print(
                    f"map_img: {map_img.shape}  box_tlbr: {box_tlbr.shape}  size_input: {size_input.shape}  input_rgb: {input_rgb.shape}")
                preds = m([map_img, box_tlbr, size_input])
                pred_location = preds[0]  # [0]
                pred_team = team_model([input_rgb, box_tlbr, size_input])
                pred_simmat = pred_team[0][..., 0]
                pred_teamvec = pred_team[1]  # [:,:,:]
                # Location: (None, None, 2)  Simmat: (None, None, None)  Teamvec: (None, None, 128)
                print(f"Location: {pred_location.shape}  Simmat: {pred_simmat.shape}  Teamvec: {pred_teamvec.shape}")
                list_pred_location.append(pred_location)
                list_pred_simmat.append(pred_simmat)
                list_pred_teamvec.append(pred_teamvec)

            return list_pred_location, list_pred_simmat, list_pred_teamvec

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, 720, 1280, 3], dtype=tf.float32),
                                      tf.TensorSpec(shape=[], dtype=tf.int32))
                     )
        def detector2stage_ensemble_batch(input_rgbs, batch_size=1): # (1, 720, 1280, 3)
            det_imgs = tf.image.resize(input_rgbs, (self.det_shape[0], self.det_shape[1]), method="bilinear") # [704, 1280]
            boxes, scores = det_model(det_imgs)

            NMS = soft_nms_layer
            list_box, list_score, num_boxes = [], [], []
            for j in range(self.inference_batch):
                i = tf.minimum(j, batch_size - 1)
                nms_box, nms_score = NMS([boxes[i:(i + 1)], scores[i:(i + 1)]])
                list_box.append(nms_box)
                list_score.append(nms_score)
                num_boxes.append(tf.shape(nms_box)[0])
            num_boxes = tf.reduce_sum(tf.stack(num_boxes)[:batch_size])
            box = tf.concat(list_box, axis=0)[:num_boxes]
            score = tf.concat(list_score, axis=0)[:num_boxes]

            box = box * tf.constant([[720 / self.det_shape[0], 1280 / self.det_shape[1],
                                      720 / self.det_shape[0], 1280 / self.det_shape[1]]], tf.float32)
            box = tf.clip_by_value(box, tf.constant([0., 0., 0., 0.]), tf.constant([720., 1280., 720., 1280.]))
            tl = box[:, :2]
            hw = box[:, 2:4] - box[:, :2]
            tlhw = tf.concat([tl, hw], axis=-1)
            size_normal = tf.math.sqrt(hw[:, 0] * hw[:, 1])
            size_select = tf.boolean_mask(size_normal, score > 0.25)
            mean_size_normal = tf.reduce_mean(size_normal)
            mean_size_select = tf.reduce_mean(size_select)
            num_detection_normal = tf.shape(size_normal)[0]
            num_detection_select = tf.shape(size_select)[0]
            mean_size_normal = self.hsh.update_and_get_current(mean_size_normal, num_detection_normal)
            mean_size_select = self.hsh_select.update_and_get_current(mean_size_select, num_detection_select)

            rate_to_25 = 25. / mean_size_normal
            resize_h_normal = 64 * ((rate_to_25 * 720) // 64)
            resize_w_normal = 64 * ((rate_to_25 * 1280) // 64)
            l_det_imgs_normal = tf.image.resize(input_rgbs, (int(resize_h_normal), int(resize_w_normal)),
                                                method="bicubic")  # "bilinear")# method="bilinear")

            rate_to_25 = 25. / mean_size_select
            resize_h_select = 64 * ((rate_to_25 * 720) // 64)
            resize_w_select = 64 * ((rate_to_25 * 1280) // 64)

            tlhw_score_each_model = []

            list_box_before_nms = []
            list_score_before_nms = []
            for idx, m in enumerate(l_det_model):
                ##boxes, scores = m(l_det_imgs_select)
                ##resize_h = resize_h_select
                ##resize_w = resize_w_select
                ##else:
                boxes, scores = m(l_det_imgs_normal)
                resize_h = resize_h_normal
                resize_w = resize_w_normal
                list_box_before_nms.append(boxes)
                list_score_before_nms.append(scores)
                list_tlhw_single, list_score_single = [], []
                for j in range(self.inference_batch):
                    i = tf.minimum(j, batch_size - 1)
                    box, score = NMS([boxes[i:(i + 1)], scores[i:(i + 1)]], score_threshold=0.08, max_output_size=30)
                    ##box, score = NMS([boxes[i:(i+1)], scores[i:(i+1)]])
                    box = box * tf.cast(tf.stack([[720 / resize_h, 1280 / resize_w, 720 / resize_h, 1280 / resize_w]]),
                                        tf.float32)
                    box = tf.clip_by_value(box, tf.constant([0., 0., 0., 0.]), tf.constant([720., 1280., 720., 1280.]))
                    tl = box[:, :2]
                    hw = box[:, 2:4] - box[:, :2]
                    tlhw = tf.concat([tl, hw], axis=-1)

                    list_tlhw_single.append(tlhw)
                    list_score_single.append(score)
                tlhw_score_each_model.append([list_tlhw_single, list_score_single])
            """
            #averaging ensemble
            ave_boxes = (list_box_before_nms[0] + list_box_before_nms[1] + list_box_before_nms[2] + list_box_before_nms[3])/4
            ave_scores = (list_score_before_nms[0] + list_score_before_nms[1] + list_score_before_nms[2] + list_score_before_nms[3])/4

            list_tlhw_single, list_score_single = [], []
            for j in range(self.inference_batch):
                i = tf.minimum(j, batch_size-1)
                box, score = NMS([ave_boxes[i:(i+1)], ave_scores[i:(i+1)]],score_threshold = 0.1, max_output_size=30)
                box = box * tf.cast(tf.stack([[720/resize_h,1280/resize_w,720/resize_h,1280/resize_w]]), tf.float32)
                box = tf.clip_by_value(box, tf.constant([0.,0.,0.,0.]), tf.constant([720.,1280.,720.,1280.]))
                tl = box[:,:2]
                hw = box[:,2:4] - box[:,:2]
                tlhw = tf.concat([tl, hw], axis=-1)  

                #mean_size = tf.reduce_mean(tf.math.sqrt(hw[:,0]*hw[:,1]))
                #num_detection = tf.shape(box)[0]
                #_ = self.hsh.update_and_get_current(mean_size, num_detection)

                list_tlhw_single.append(tlhw)
                list_score_single.append(score) 
            tlhw_score_each_model.append([list_tlhw_single, list_score_single])
            tlhw_score_each_model = tlhw_score_each_model[::-1]
            #"""
            return tlhw_score_each_model

        def registrator_wrapper(view):
            @tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                          tf.TensorSpec(shape=[None], dtype=tf.float32),
                                          tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                          tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                          # tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                          tf.TensorSpec(shape=[], dtype=tf.bool),
                                          tf.TensorSpec(shape=[], dtype=tf.bool),
                                          tf.TensorSpec(shape=[], dtype=tf.bool),
                                          tf.TensorSpec(shape=[3], dtype=tf.float32),
                                          tf.TensorSpec(shape=[3], dtype=tf.float32),
                                          tf.TensorSpec(shape=[2, 3], dtype=tf.float32),
                                          tf.TensorSpec(shape=[], dtype=tf.float32),  ##増やした
                                          tf.TensorSpec(shape=[], dtype=tf.int32),  ##増やした
                                          tf.TensorSpec(shape=[], dtype=tf.int32),  ##増やした
                                          tf.TensorSpec(shape=[], dtype=tf.int32),  ##増やした
                                          )
                         )
            def registrator(pred_location, pred_team, confidence, locations,
                            team_labels, cost_matrix,
                            # motions,
                            team_provided,

                            use_provided_params, use_random_params,
                            zoom_params, rz_params, txy_params,
                            confidence_threshold,
                            num_harddrop,
                            num_softdrop,
                            num_trial):
                results = random_icp_fitting_team_drop(locations,  # target
                                                       pred_location,  # pred
                                                       confidence,
                                                       team_labels,
                                                       pred_team,
                                                       cost_matrix,
                                                       team_provided=team_provided,
                                                       # motions,
                                                       # pred_motions,
                                                       num_trial=num_trial,  # 120,
                                                       num_fitting_iter=8,  # 8,
                                                       use_provided_params=use_provided_params,
                                                       use_random_params=use_random_params,
                                                       zoom_params=zoom_params,
                                                       rz_params=rz_params,
                                                       txy_params=txy_params,
                                                       confidence_threshold=confidence_threshold,
                                                       num_harddrop=num_harddrop,
                                                       num_softdrop=num_softdrop,
                                                       mode=view)
                return results

            return registrator

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.int32),
                                      tf.TensorSpec(shape=[None], dtype=tf.int32),
                                      tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
                                      tf.TensorSpec(shape=[], dtype=tf.float32),
                                      tf.TensorSpec(shape=[], dtype=tf.int32),
                                      tf.TensorSpec(shape=[], dtype=tf.float32),)
                     )
        def prelocation(pred_location, ref_box_indexs, not_ref_box_indexes, ref_locations, all_locations, l2_reg,
                        num_iter, rot_init):
            ref_box_pred_locations = tf.gather(pred_location, ref_box_indexs)
            trans_sources, transmatrix, k_init, rz_init, tx_init, ty_init = points2points_fitting(
                ref_locations[tf.newaxis, :, :],
                ref_box_pred_locations[tf.newaxis, :, :],
                num_iter=num_iter,
                l2_reg=l2_reg,
                rot_init=rot_init)
            trans_sources_all = transform_points(transmatrix[0], pred_location)
            # initial parameters used for registration
            rz_params = tf.stack([rz_init[0], 0.05, 500.])
            zoom_params = tf.stack([k_init[0], 0.05, 20.])
            txy_params = tf.stack([[tx_init[0], 0.05, 20.],
                                   [ty_init[0], 0.05, 20.]])
            init_error, assigned_targets, assignments, not_assigned_targets = search_nearest_error(trans_sources[0],
                                                                                                   all_locations)
            not_assigned_sources = tf.gather(trans_sources_all, not_ref_box_indexes)
            dist_to_near_targets_not_assigned, dist_to_near_sources_not_assigned = get_nearest_distance(
                not_assigned_sources, not_assigned_targets)
            dist_to_near_targets, dist_to_near_sources = get_nearest_distance(trans_sources_all, all_locations)
            return trans_sources, trans_sources_all, transmatrix, \
                   rz_params, zoom_params, txy_params, init_error, \
                   assigned_targets, assignments, dist_to_near_targets, \
                   dist_to_near_sources, dist_to_near_targets_not_assigned

        return mapper_ensemble, detector2stage_ensemble_batch, registrator_wrapper("Sideline"), registrator_wrapper(
            "Endzone"), prelocation

    @staticmethod
    def draw_bbox(img, boxes, gt_boxes=None, save_only=True, save_path="", save_title=""):
        from PIL import Image, ImageDraw
        save_dir = save_path + save_title + ".jpg"
        pil_img = Image.fromarray((img).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        # text_w, text_h = draw.textsize(text)
        # label_y = y if y <= text_h else y - text_h
        # draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline=bbcolor, fill=bbcolor)
        # draw.text((x, label_y), text, fill=textcolor)
        if gt_boxes is not None:
            for t, l, h, w in gt_boxes:
                draw.rectangle((int(l), int(t), int(l + w), int(t + h)), outline="blue", width=3)
        for t, l, h, w in boxes:
            draw.rectangle((int(l), int(t), int(l + w), int(t + h)), outline="red", width=3)
        if save_only:
            pil_img.save(save_dir)
        else:
            plt.figure(figsize=(9, 5))
            plt.imshow(pil_img)
            plt.show()

    @staticmethod
    def draw_tracking_box(img, boxes, tracking_ids, save_only=True, save_path="", save_title=""):
        from PIL import Image, ImageDraw
        save_dir = save_path + save_title + ".jpg"
        cmap = plt.get_cmap("tab20")
        tracking_colors = [
            (int(cmap(tr_id % 20)[0] * 255), int(cmap(tr_id % 20)[1] * 255), int(cmap(tr_id % 20)[2] * 255)) for tr_id
            in tracking_ids]
        pil_img = Image.fromarray((img).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        for [t, l, h, w], c in zip(boxes, tracking_colors):
            draw.rectangle((int(l), int(t), int(l + w), int(t + h)), outline=c, width=3)
        if save_only:
            pil_img.save(save_dir)
        else:
            plt.figure(figsize=(9, 5))
            plt.imshow(pil_img)
            plt.show()

    def run_detection_ensemble_batch(self, list_img, frame, view, game_play):
        # 将图片交给各个检测模型预测出bbox位置和得分
        # [[[(num_box, 4)], [(num_box, 1)]], ..., [[(num_box, 4)], [(num_box, 1)]]]
        # 之后对这些bbox略微处理了一下，去除太小的bbox，去除重复的bbox
        # 返回[[df1, ..., dfn]], [[tlbr_boxes1, ..., tlbr_boxesn]]
        batch_size = len(list_img)
        batch_img = tf.concat(list_img, axis=0)  # (b, h, w, 3)
        # list_tlhw, list_confidence
        tlhw_score_each_model = self.detector(batch_img, batch_size)
        # [[[xxx.shape=(num_box, 4)], [xxx.shape=(num_box, 1)]], ..., [[xxx.shape=(num_box, 4)], [xxx.shape=(num_box, 1)]]]
        tlhw_score_each_model = [[out[0][:batch_size], out[1][:batch_size]] for out in tlhw_score_each_model]
        # [[[(num_box, 4)], [(num_box, 1)]], ..., [[(num_box, 4)], [(num_box, 1)]]]
        batch_list_current_frame_helmets = [[] for _ in range(batch_size)] # [[]]
        batch_list_tlbr_boxes = [[] for _ in range(batch_size)]  # [[]]
        for list_tlhw, list_confidence in tlhw_score_each_model: # list_tlhw=[(num_box, 4)], list_confidence=[(num_box, 1)]
            for frame_idx, [confidence, tlhw] in enumerate(zip(list_confidence, list_tlhw)): # 0, [(num_box, 1), (num_box, 4)]
                current_frame_helmets = pd.DataFrame(tlhw.numpy(), columns=["top", "left", "height", "width"])
                current_frame_helmets["conf"] = confidence.numpy().reshape(-1)
                current_frame_helmets["frame"] = frame[frame_idx]
                current_frame_helmets["view"] = view
                current_frame_helmets["game_play"] = game_play
                current_frame_helmets = current_frame_helmets.sort_values('conf', ascending=False) # 降序
                current_frame_helmets["bottom"] = current_frame_helmets["top"] + current_frame_helmets["height"]
                current_frame_helmets["right"] = current_frame_helmets["left"] + current_frame_helmets["width"]
                current_frame_helmets = current_frame_helmets[
                    ~np.round(current_frame_helmets[["left", "width", "top", "height"]]).duplicated()] # 将坐标去除小数点后bbox去重
                # 去除太小的检测框
                current_frame_helmets = current_frame_helmets[np.round(current_frame_helmets["height"]) > 1.]
                current_frame_helmets = current_frame_helmets[np.round(current_frame_helmets["width"]) > 1.]
                tlbr_boxes = tf.cast(current_frame_helmets[["top", "left", "bottom", "right"]].values, tf.float32)
                # list_current_frame_helmets.append(current_frame_helmets)
                # list_tlbr_boxes.append(tlbr_boxes)
                batch_list_current_frame_helmets[frame_idx].append(current_frame_helmets)
                batch_list_tlbr_boxes[frame_idx].append(tlbr_boxes)
        """
        for frame_idx in range(batch_size):
            #WBF HERE?   
            list_current_frame_helmets = batch_list_current_frame_helmets[frame_idx]
            list_tlbr_boxes = batch_list_tlbr_boxes[frame_idx]
            list_boxes = [df[["top", "left", "bottom", "right"]].values for df in list_current_frame_helmets]
            list_confs = [df["conf"].values for df in list_current_frame_helmets]
            fusion_box, fusion_conf = wbf(list_boxes, list_confs, 
                                          #model_weights=model_weights, 
                                          iou_thresh=0.5,
                                          mode="average") 
            mask = fusion_conf > 0.03
            fusion_box = fusion_box[mask]
            fusion_conf = fusion_conf[mask]
            current_frame_helmets = pd.DataFrame(fusion_box, columns=["top", "left", "bottom", "right"])
            current_frame_helmets["conf"] = fusion_conf.reshape(-1)
            current_frame_helmets["frame"] = frame[frame_idx]
            current_frame_helmets["view"] = view
            current_frame_helmets["game_play"] = game_play            
            current_frame_helmets = current_frame_helmets.sort_values('conf', ascending=False)
            current_frame_helmets["height"] = current_frame_helmets["bottom"] - current_frame_helmets["top"]
            current_frame_helmets["width"] = current_frame_helmets["right"] - current_frame_helmets["left"]
            current_frame_helmets = current_frame_helmets[~np.round(current_frame_helmets[["left", "width", "top", "height"]]).duplicated()]
            current_frame_helmets = current_frame_helmets[np.round(current_frame_helmets["height"])>1.]
            current_frame_helmets = current_frame_helmets[np.round(current_frame_helmets["width"])>1.]
            tlbr_boxes = tf.cast(current_frame_helmets[["top", "left", "bottom", "right"]].values, tf.float32)            
            list_current_frame_helmets.append(current_frame_helmets)
            list_tlbr_boxes.append(tlbr_boxes)
            batch_list_current_frame_helmets[frame_idx] = list_current_frame_helmets
            batch_list_tlbr_boxes[frame_idx] = list_tlbr_boxes
        """
        # [[df1, ..., dfn]], [[tlbr_boxes1, ..., tlbr_boxesn]]
        return batch_list_current_frame_helmets, batch_list_tlbr_boxes

    def run_mapping_ensemble(self, img, tlbr_boxes, list_current_frame_helmets, f_columns, list_tfh, list_params_set):
        '''
        将map模型预测的坐标，team模型预测的128维向量和预测得出的所属队伍，添加到df中
        img: 这一帧经过归一化处理后的图像, (1, h, w, 3)
        tlbr_boxes: 第ensemble_idx个检测模型对这帧图像的检测结果, (n, 4)
        list_current_frame_helmets: [dfi],
        f_columns: 0-127的字符串字典,
        list_tfh: [tfhi],
        list_params_set: 第ensemble_idx个检测模型对应的字典,
        '''
        tlbr_boxes = tf.reshape(tlbr_boxes, [1, -1, 4])
        list_pred_location, list_pred_simmat, list_pred_features = self.mapper(img, tlbr_boxes)
        # Location: [(b, num_box, 2)]  Simmat: [(b, num_box, num_box)]  Teamvec: [(b, num_box, 128)]

        """        
        f_t = tlbr_boxes[:,:,:1]
        f_l = 1280. - tlbr_boxes[:,:,3:4]
        f_b = tlbr_boxes[:,:,2:3]
        f_r = 1280. - tlbr_boxes[:,:,1:2]
        f_tlbr_boxes = tf.concat([f_t,f_l,f_b,f_r], axis=-1)
        f_img = img[:,:,::-1,:]
        f_pred_location, f_pred_simmat, f_pred_features = self.mapper(f_img, f_tlbr_boxes)
        f_pred_location = f_pred_location * tf.constant([[[-1, 1]]],tf.float32)
        """
        # list_pred_location = [pred_location]#, f_pred_location]
        # list_pred_simmat = [pred_simmat]#, f_pred_simmat]
        # list_pred_features = [pred_features]#, f_pred_features]
        # 0, [dfi, tfhi, adicti, (b, num_box, 2), (b, num_box, num_box), (b, num_box, 128)]
        for i, [current_frame_helmets, tfh, params_set, pred_location, pred_simmat, pred_features] in enumerate(
                zip(list_current_frame_helmets,
                    list_tfh,
                    list_params_set,
                    list_pred_location,
                    list_pred_simmat,
                    list_pred_features)):
            current_frame_helmets.loc[:, "loc_x"] = pred_location[0].numpy()[:, 0]
            current_frame_helmets.loc[:, "loc_y"] = pred_location[0].numpy()[:, 1]  # zero is batch dim
            # current_frame_helmets["team"] = pred_team.numpy().reshape(-1)
            for k in range(128):
                current_frame_helmets[f_columns[k]] = pred_features[0].numpy()[:, k]
            binary_predict = tfh.predict(pred_features[0].numpy())
            if binary_predict is not None:
                params_set["team_provided"] = True
                current_frame_helmets["team_pred"] = binary_predict
            else:
                pred_team = tf.numpy_function(func=similarity_matrix_to_team, inp=[pred_simmat[0]], Tout=[tf.float32])
                current_frame_helmets["team_pred"] = pred_team.numpy().reshape(-1)
                params_set["team_provided"] = False
            list_current_frame_helmets[i] = current_frame_helmets
            list_params_set[i] = params_set
        return list_current_frame_helmets, list_params_set

    def preprocess_registration(self, current_frame_helmets, current_tracking,
                                trk, params_set,
                                game_play, view, frame,
                                start_frame=1, view_frequency=10000,
                                only_return_inputs=False):
        # current_frame_helmets：加入map坐标、team128维特征、team分组信息的dataframe
        # current_tracking：当前game_play当前frame的tracking数据
        # trk：Tracker_2_w_feature(0.3)
        # return：
        # 用置信度和tracking中点的数量过滤了一波检测框，过滤后的检测框df为current_frame_helmets
        # 过滤掉的检测框集中到了current_frame_helmets_low_conf
        # params_set的键值对有更新，且其中真实坐标除以20
        # test_inputs： 运动员的坐标和所属队伍（0/1）
        # all_data：包含test_inputs中内容的大杂烩
        try:
            box_idx_high_iou, ious = trk.precheck_iou(game_play, view, frame,
                                                      current_boxes=current_frame_helmets[
                                                          ["top", "left", "bottom", "right"]].values,
                                                      iou_threshold=0.3)
            if box_idx_high_iou is not None:
                conf_rescore = current_frame_helmets["conf"].values  # - 0.1# - 0.05
                conf_rescore[box_idx_high_iou] += 0.1  # ious*0.1
                current_frame_helmets["conf"] = conf_rescore
        except:
            pass
            # print(len(box_idx_high_iou), "in", len(conf_rescore), "is high iou")

        all_locations = current_tracking[["x", "y"]].values  # make_locations(current_tracking)
        all_players = current_tracking["player"].values.tolist()
        # locations坐标全部除以20
        test_inputs, all_data = self.preprocess_inputs(all_locations, all_players)
        # test_inputs： 运动员的坐标和所属队伍（0/1）
        # all_data：杂
        if only_return_inputs:
            return test_inputs, all_data
        try:
            ref_box_indexs, notrack_box_indexs, ref_locations = trk.precheck_and_get_location(game_play, view, frame,
                                                                                              current_boxes=
                                                                                              current_frame_helmets[
                                                                                                  ["top", "left",
                                                                                                   "bottom",
                                                                                                   "right"]].values,
                                                                                              )
        except:
            ref_box_indexs = None
        # default setting
        params_set["random"] = None
        if params_set["lost_track_frame"] > 2:
            params_set["determined"] = None

        if ((frame - start_frame) < 5) or (params_set["side_fixed"] == False):  # provide filtered initial
            use_provided_params = tf.constant(False)
            rz_params = tf.ones((3), tf.float32)
            txy_params = tf.ones((2, 3), tf.float32)
            zoom_params = tf.ones((3), tf.float32)
            params_set["random"] = {"use_provided_params": use_provided_params,
                                    "rz_params": rz_params,
                                    "txy_params": txy_params,
                                    "zoom_params": zoom_params,
                                    "use_random_params": tf.constant(True),
                                    }
        if ((frame - start_frame) >= 5):
            if np.std(params_set["hist_rot_angles"][-5:]) < 0.075:
                mean_angle = np.mean(params_set["hist_rot_angles"][-5:])
                if view == "Endzone":
                    if mean_angle < (np.pi / 2 + 0.65) and mean_angle > (np.pi / 2 - 0.65):
                        params_set["side_fixed"] = True
                        params_set["base_angle"] = mean_angle
                    elif mean_angle < (-np.pi / 2 + 0.65) and mean_angle > (-np.pi / 2 - 0.65):
                        params_set["side_fixed"] = True
                        params_set["base_angle"] = mean_angle

                else:  # Sideline
                    if mean_angle < (np.pi + 0.65) and mean_angle > (np.pi - 0.65):
                        params_set["side_fixed"] = True
                        params_set["base_angle"] = mean_angle
                    elif mean_angle < 0.65 and mean_angle > -0.65:
                        params_set["side_fixed"] = True
                        params_set["base_angle"] = mean_angle

            if params_set["side_fixed"]:
                use_provided_params = tf.constant(True)
                # use_random_params = tf.constant(True)
                # rz_params = tf.constant([rotation_angle_filtered, 0.10, 2000],tf.float32)
                rz_params = tf.constant([params_set["base_angle"], 0.20, 100], tf.float32)
                # txyは今無効。num_fitting_iter=8, 1stepの動作量を1/Nにする。
                txy_params = tf.constant([[params_set["xy_location_filtered"][0], 0.10, 5],
                                          [params_set["xy_location_filtered"][1], 0.10, 5]], tf.float32)
                zoom_params = tf.stack([0.0, 0.5, 5.])
                params_set["random"] = {"use_provided_params": tf.constant(True),
                                        "rz_params": rz_params,
                                        "txy_params": txy_params,
                                        "zoom_params": zoom_params,
                                        "use_random_params": tf.constant(True),
                                        }

            if ref_box_indexs is not None and use_provided_params:
                ##print("num_tracked", len(ref_box_indexs))
                temp_current_frame_helmets = current_frame_helmets.copy().iloc[ref_box_indexs, :]
                success = True
                pred_location = tf.cast(current_frame_helmets[["loc_x", "loc_y"]].values, tf.float32)
                rot_init = params_set["base_angle"]

                try:
                    num_iter = 50
                    l2_reg = 0.1
                    trans_sources, trans_sources_all, transmatrix, \
                    rz_params, zoom_params, txy_params, init_error, \
                    assigned_targets, assignments, dist_to_near_targets, \
                    dist_to_near_sources, dist_to_near_targets_not_assigned = self.prelocate(pred_location,
                                                                                             tf.cast(ref_box_indexs,
                                                                                                     tf.int32),
                                                                                             tf.cast(notrack_box_indexs,
                                                                                                     tf.int32),
                                                                                             tf.cast(ref_locations,
                                                                                                     tf.float32),
                                                                                             tf.cast(all_data[
                                                                                                         "all_locations"],
                                                                                                     tf.float32),
                                                                                             l2_reg,
                                                                                             num_iter,
                                                                                             rot_init)
                except:
                    try:
                        num_iter = 500
                        l2_reg = 1.0
                        trans_sources, trans_sources_all, transmatrix, \
                        rz_params, zoom_params, txy_params, init_error, \
                        assigned_targets, assignments, dist_to_near_targets, \
                        dist_to_near_sources, dist_to_near_targets_not_assigned = self.prelocate(pred_location,
                                                                                                 tf.cast(ref_box_indexs,
                                                                                                         tf.int32),
                                                                                                 tf.cast(
                                                                                                     notrack_box_indexs,
                                                                                                     tf.int32),
                                                                                                 tf.cast(ref_locations,
                                                                                                         tf.float32),
                                                                                                 tf.cast(all_data[
                                                                                                             "all_locations"],
                                                                                                         tf.float32),
                                                                                                 l2_reg,
                                                                                                 num_iter,
                                                                                                 rot_init)
                    except:
                        try:
                            num_iter = 500
                            l2_reg = 5.0
                            trans_sources, trans_sources_all, transmatrix, \
                            rz_params, zoom_params, txy_params, init_error, \
                            assigned_targets, assignments, dist_to_near_targets, \
                            dist_to_near_sources, dist_to_near_targets_not_assigned = self.prelocate(pred_location,
                                                                                                     tf.cast(
                                                                                                         ref_box_indexs,
                                                                                                         tf.int32),
                                                                                                     tf.cast(
                                                                                                         notrack_box_indexs,
                                                                                                         tf.int32),
                                                                                                     tf.cast(
                                                                                                         ref_locations,
                                                                                                         tf.float32),
                                                                                                     tf.cast(all_data[
                                                                                                                 "all_locations"],
                                                                                                             tf.float32),
                                                                                                     l2_reg,
                                                                                                     num_iter
                                                                                                     )
                        except:
                            success = False
                            ref_box_indexs = None

                if success:

                    # judge out of fieald helmets. by y_location and distance to nearest target
                    min_x = 0.05
                    max_x = (120.0 / 20.0) - 0.05
                    min_y = 0.05
                    max_y = (53.3 / 20.0) - 0.05
                    allowable_dist_error_0 = 0.10  # レンジ外とあわせてNGにする条件
                    allowable_dist_error_1 = 0.25  # 単独でNGにする条件
                    in_field_box_x = tf.logical_and(trans_sources_all[:, 0] > min_x, trans_sources_all[:, 0] < max_x)
                    in_field_box_y = tf.logical_and(trans_sources_all[:, 1] > min_y, trans_sources_all[:, 1] < max_y)
                    in_field_box = tf.logical_and(in_field_box_x, in_field_box_y)
                    in_field_box = tf.logical_or(in_field_box,
                                                 dist_to_near_targets < (allowable_dist_error_0 ** 2)).numpy()
                    neglected_helmets_loc = trans_sources_all.numpy()[~in_field_box]

                    # neglect targets far from predicted points
                    if params_set["neglect_far_targets"]:
                        source_is_near = (dist_to_near_sources < (allowable_dist_error_1 ** 2))
                        neglected_helmets_from_targets = all_data["all_locations"][~source_is_near.numpy()]
                        test_inputs["team_labels"] = tf.boolean_mask(test_inputs["team_labels"], source_is_near, axis=0)
                        test_inputs["all_locations"] = tf.boolean_mask(test_inputs["all_locations"], source_is_near,
                                                                       axis=0)
                        all_data["all_locations"] = all_data["all_locations"][source_is_near.numpy()]
                        all_data["all_players"] = all_data["all_players"][source_is_near.numpy()]
                        all_data["team_labels"] = tf.boolean_mask(all_data["team_labels"], source_is_near, axis=0)
                        # print(len(neglected_helmets_from_targets), "targets are neglected.")

                    # neglect predicted points far from targets
                    target_is_near = (dist_to_near_targets < (allowable_dist_error_1 ** 2)).numpy()
                    neglected_helmets_far = trans_sources_all.numpy()[~target_is_near]
                    ok_mask = np.logical_and(in_field_box, target_is_near)

                    notrack_box_indexs = np.array(notrack_box_indexs).astype(int)
                    if len(notrack_box_indexs) > 0:
                        target_is_near_not_assigned = (
                                    dist_to_near_targets_not_assigned < (allowable_dist_error_1 ** 2)).numpy()
                        neglected_helmets_far_not_assigned = trans_sources_all.numpy()[np.array(notrack_box_indexs)][
                            ~target_is_near_not_assigned]
                        mask_not_assigned_dist = np.array([True for _ in range(len(current_frame_helmets))])
                        mask_not_assigned_dist[np.array(notrack_box_indexs)] = target_is_near_not_assigned
                        drop_by_this = np.logical_and(ok_mask, ~mask_not_assigned_dist)
                        ok_mask = np.logical_and(ok_mask, mask_not_assigned_dist)
                    else:
                        neglected_helmets_far_not_assigned = []
                    current_frame_helmets = current_frame_helmets[ok_mask]

                    if DRAW_PREREGI and frame % view_frequency == 0:
                        plt.scatter(assigned_targets[0, :, 0], assigned_targets[0, :, 1], color="blue")
                        plt.scatter(trans_sources[0, :, 0], trans_sources[0, :, 1], color="red")
                        plt.scatter(trans_sources_all[:, 0], trans_sources_all[:, 1], color="red", alpha=0.3)
                        if len(neglected_helmets_loc) > 0:
                            plt.scatter(neglected_helmets_loc[:, 0], neglected_helmets_loc[:, 1], facecolors="none",
                                        edgecolors="green")
                        if len(neglected_helmets_far) > 0:
                            plt.scatter(neglected_helmets_far[:, 0], neglected_helmets_far[:, 1], facecolors="none",
                                        edgecolors="gray")
                        if params_set["neglect_far_targets"]:
                            if len(neglected_helmets_from_targets) > 0:
                                print(len(neglected_helmets_from_targets), "targets are neglected.")
                                plt.scatter(all_data["all_locations"][:, 0], all_data["all_locations"][:, 1],
                                            color="black", alpha=0.15)
                                plt.scatter(neglected_helmets_from_targets[:, 0], neglected_helmets_from_targets[:, 1],
                                            color="black")
                        plt.title("frame {} preregistration".format(str(frame)))
                        plt.show()

                    if len(current_frame_helmets) > 1:
                        params_set["determined"] = {"use_provided_params": tf.constant(True),
                                                    "rz_params": rz_params,
                                                    "txy_params": txy_params,
                                                    "zoom_params": zoom_params,
                                                    "use_random_params": tf.constant(False),
                                                    }
                        params_set["lost_track_frame"] = 0
                    else:
                        current_frame_helmets = temp_current_frame_helmets

        if ref_box_indexs is None:
            params_set["lost_track_frame"] += 1
            # params_set["neglect_far_targets"] = False

        base_thresh = 0.2  # minimum detection score use for registration
        num_min_mapping = 2
        num_detect = len(current_frame_helmets)
        min_thresh = np.minimum(base_thresh,
                                current_frame_helmets["conf"].values[np.minimum(num_min_mapping, num_detect) - 1])
        current_frame_helmets_low_conf = current_frame_helmets[current_frame_helmets["conf"] < min_thresh]
        current_frame_helmets = current_frame_helmets[current_frame_helmets["conf"] >= min_thresh]
        if len(current_tracking) < len(current_frame_helmets):  # 過剰捲縮t時は
            current_frame_helmets_low_conf = pd.concat(
                [current_frame_helmets[len(current_tracking):], current_frame_helmets_low_conf])
            current_frame_helmets = current_frame_helmets[:len(
                current_tracking)]  # .sort_values('conf', ascending=False)[:len(current_tracking)]

        params_set["num_harddrop"] = 0  # not use
        params_set["num_softdrop"] = 0

        # change confidence threshold for drop during registration
        conf_limit = current_frame_helmets["conf"].values[-params_set["num_harddrop"]] if params_set[
                                                                                              "num_harddrop"] > 0 else 0.0
        params_set["conf_threshold"] = np.maximum(0.4, conf_limit)  # 0.4
        # 用置信度和tracking中点的数量过滤了一波检测框，过滤后的检测框df为current_frame_helmets
        # 过滤掉的检测框集中到了current_frame_helmets_low_conf
        # params_set的键值对有更新，且真实坐标除以20
        # test_inputs： 运动员的坐标和所属队伍（0/1）
        # all_data：包含test_inputs中内容的大杂烩
        return current_frame_helmets, current_frame_helmets_low_conf, params_set, test_inputs, all_data

    def run_registration(self, current_frame_helmets, current_frame_helmets_low_conf,
                         params_set, test_inputs, all_data,
                         tfh, f_columns,
                         game_play, view, frame):

        pred_location = tf.cast(current_frame_helmets[["loc_x", "loc_y"]].values, tf.float32)
        pred_team = tf.cast(current_frame_helmets["team_pred"].values.reshape(-1, 1), tf.float32)
        confidence = tf.cast(current_frame_helmets["conf"].values, tf.float32)
        feature_dist_mat = np.zeros((len(pred_location), len(test_inputs["all_locations"])), np.float32) # (num_box, 22)

        if params_set["determined"] is None:  # random initialized ICP
            icp_inputs = make_icp_inputs(targets=test_inputs["all_locations"],
                                         sources=pred_location,
                                         targets_team=test_inputs["team_labels"],
                                         sources_team=pred_team,
                                         team_provided=params_set["team_provided"],
                                         confidence=confidence,
                                         confidence_threshold=params_set["conf_threshold"],
                                         num_trial=100,
                                         is_sideline=(view == "Sideline"),
                                         **params_set["random"])
            results = random_icp_fitting(*icp_inputs,
                                         st_cost_matrix=feature_dist_mat,
                                         num_fitting_iter=8,
                                         num_harddrop=params_set["num_harddrop"],
                                         num_softdrop=params_set["num_softdrop"], )

        else:  # use random and fixed initialized ICP
            if params_set["team_provided"]:
                num_try_r = 40
                num_try = 20
            else:
                num_try_r = 60
                num_try = 30
            icp_inputs_r = make_icp_inputs(targets=test_inputs["all_locations"],
                                           sources=pred_location,
                                           targets_team=test_inputs["team_labels"],
                                           sources_team=pred_team,
                                           team_provided=params_set["team_provided"],
                                           confidence=confidence,
                                           confidence_threshold=params_set["conf_threshold"],
                                           num_trial=num_try_r,
                                           is_sideline=(view == "Sideline"),
                                           **params_set["random"])
            icp_inputs = make_icp_inputs(targets=test_inputs["all_locations"],
                                         sources=pred_location,
                                         targets_team=test_inputs["team_labels"],
                                         sources_team=pred_team,
                                         team_provided=params_set["team_provided"],
                                         confidence=confidence,
                                         confidence_threshold=params_set["conf_threshold"],
                                         num_trial=num_try,
                                         is_sideline=(view == "Sideline"),
                                         **params_set["determined"])
            icp_inputs = [tf.concat([x, y], axis=0) for x, y in zip(icp_inputs, icp_inputs_r)]
            results = random_icp_fitting(*icp_inputs,
                                         st_cost_matrix=feature_dist_mat,
                                         num_fitting_iter=8,
                                         num_harddrop=params_set["num_harddrop"],
                                         num_softdrop=params_set["num_softdrop"], )

        keys = ["residual", "trans_matrix", "trans_sources", "final_assignment", "raw_results", "assigned_mask",
                "xy_residual"]
        results = {k: v for k, v in zip(keys, results)}

        def check_team_sign(before_team_labels, after_team_labels):
            if len(before_team_labels) != len(after_team_labels):
                raise Exception("length of before and after is different")
            before = np.argmax(before_team_labels)  # [-1]-before_team_labels[0])
            after = np.argmax(after_team_labels)  # [-1]-after_team_labels[0])
            if before == after:
                sign = 1.0
                # before*rate = after
                rate = after_team_labels[0] / before_team_labels[0]
            else:
                sign = -1.0
                # (1-before)*rate = after
                rate = after_team_labels[0] / (1.0 - before_team_labels[0])
            return sign, 0.1  # team weight used for icp registration

        results["trans_sources"] = tf.boolean_mask(results["trans_sources"], results["assigned_mask"])

        pred_team_finally = results["trans_sources"].numpy()[:, 2]
        team_sign, team_w = check_team_sign(pred_team.numpy()[results["assigned_mask"].numpy()].reshape(-1, 1),
                                            pred_team_finally)

        results_assignment = results["final_assignment"].numpy()  # .astype(int)
        assigned_label = all_data["all_players"][results_assignment]
        xy_residual = results["xy_residual"].numpy().reshape(-1)

        if params_set["num_harddrop"] > 0:
            current_frame_helmets = current_frame_helmets[results["assigned_mask"].numpy()]

        # save team features
        tfh.add(pred_team_finally / team_w,
                test_inputs["team_labels"].numpy().reshape(-1)[results_assignment],
                current_frame_helmets[f_columns].values)

        if len(current_frame_helmets) < len(all_data["all_players"]):
            # assign from remaining targets and remaining predictions(low detection score)
            remain_idx = np.array(list(set(range(len(all_data["all_players"]))) - set(results_assignment)))
            not_assigned_targets = all_data["all_locations"][remain_idx]
            not_assigned_targets_team = team_w * test_inputs["team_labels"].numpy()[remain_idx]
            not_assigned_sources = current_frame_helmets_low_conf[["loc_x", "loc_y"]].values
            not_assigned_sources = transform_points(results["trans_matrix"],
                                                    tf.cast(not_assigned_sources, tf.float32)).numpy()

            not_assigned_sources_team = team_w * current_frame_helmets_low_conf["team_pred"].values
            not_assigned_sources_team = not_assigned_sources_team if team_sign == 1.0 else (
                        team_w - not_assigned_sources_team)

            not_assigned_sources = np.concatenate([not_assigned_sources, not_assigned_sources_team.reshape(-1, 1)],
                                                  axis=-1)
            not_assigned_targets = np.concatenate([not_assigned_targets, not_assigned_targets_team.reshape(-1, 1)],
                                                  axis=-1)
            thresh_dist_add = 0.12 ** 2

            median_assigned = np.median(xy_residual)
            max_assigned = np.max(xy_residual)
            # print("median_error: ", median_assigned)
            # thresh_dist_add_0 = np.minimum(thresh_dist_add, median_assigned*2)
            thresh_dist_add = np.minimum(thresh_dist_add, max_assigned)

            conf_th_for_rest_assign = 0.1
            distmat = np.sum((not_assigned_sources[:, np.newaxis, :] - not_assigned_targets[np.newaxis, :, :]) ** 2,
                             axis=-1)
            distmat_conf = distmat + (current_frame_helmets_low_conf["conf"].values < conf_th_for_rest_assign).reshape(
                -1, 1).astype(float) * 100

            # take from minimum
            xy_redisual_add = []
            pred_idx_add = []
            assigned_idx_add = []
            idx_list_s = np.arange(distmat.shape[0])
            idx_list_t = np.arange(distmat.shape[1])

            from_high_conf = True
            # not implemented(forget adding 100 to threshold)
            while True:
                num_s, num_t = distmat.shape[:2]
                if num_s == 0 or num_t == 0:
                    break
                if from_high_conf:
                    mat = distmat_conf
                else:
                    mat = distmat
                argmin = np.argmin(mat)
                idx_s, idx_t = argmin // num_t, argmin % num_t
                min_value = mat[idx_s, idx_t]
                if min_value > 100.0:
                    from_high_conf = False
                    continue
                if min_value > thresh_dist_add:
                    break
                else:
                    pred_idx_add.append(idx_list_s[idx_s])
                    assigned_idx_add.append(idx_list_t[idx_t])
                    xy_redisual_add.append(min_value)
                    distmat = np.delete(distmat, idx_s, axis=0)
                    distmat = np.delete(distmat, idx_t, axis=1)
                    distmat_conf = np.delete(distmat_conf, idx_s, axis=0)
                    distmat_conf = np.delete(distmat_conf, idx_t, axis=1)
                    idx_list_s = np.delete(idx_list_s, idx_s)
                    idx_list_t = np.delete(idx_list_t, idx_t)

            if len(pred_idx_add) > 0:
                # print("NEW ASSIGN", len(pred_idx_add))
                HIGH_RESIDUAL = False
                xy_redisual_add = np.array(xy_redisual_add)
                add_helmets = current_frame_helmets_low_conf.iloc[np.array(pred_idx_add), :]
                current_frame_helmets = pd.concat([current_frame_helmets, add_helmets], axis=0)
                assigned_idx_add = remain_idx[np.array(assigned_idx_add)]
                results_assignment = np.concatenate([results_assignment, assigned_idx_add], axis=0)
                assigned_label_add = all_data["all_players"][assigned_idx_add]
                assigned_label = np.concatenate([assigned_label, assigned_label_add], axis=0)
                xy_residual = np.concatenate([xy_residual, xy_redisual_add], axis=0)

        results["xy_residual"] = xy_residual
        results["final_assignment"] = results_assignment
        rot = results["raw_results"][1].numpy()
        params_set["hist_rot_angles"].append(rot)
        update_ratio = 0.1
        if params_set["xy_location_filtered"] is None:
            params_set["xy_location_filtered"] = np.mean(results["trans_sources"].numpy()[:, :2], axis=0)
        else:
            params_set["xy_location_filtered"] = params_set["xy_location_filtered"] * (1 - update_ratio) + \
                                                 np.mean(results["trans_sources"].numpy()[:, :2], axis=0) * update_ratio

        return results, assigned_label, params_set, current_frame_helmets

    def test_predict_ensemble_batch(self, test_tracking, test_helmets, video_path, labels=None,
                                    save_path="output/temp/"):
        all_labels = []
        all_game_predictions = []
        f_columns = ["f{}".format(i) for i in range(128)]
        time_measurement = False
        view_frequency = VIEW_FREQUENCY
        for game_play, _df_tracking in test_tracking.groupby(["game_play"]):
            # 几乎没有语句平行于该for循环
            current_test_helmets = test_helmets[test_helmets["game_play"] == game_play]
            views = current_test_helmets["view"].unique()
            # gameKey = game_play.split("_")[0]

            for view in views:
                # 没有语句平行于该for循环
                print("GAME_PLAY: ", game_play, " VIEW:", view, len(_df_tracking))

                current_view_test_helmets = current_test_helmets[current_test_helmets["view"] == view]
                video_file = video_path + game_play + "_" + view + ".mp4"
                cap = cv2.VideoCapture(video_file)
                frames = current_view_test_helmets["frame"].unique()

                frame_scores = []
                self.hsh.reset()  # 每次到新视频时重置，随着算法见到的该视频画面越来越多，met size越来越精准
                self.hsh_select.reset()

                start_frame = 1
                end_frame = 1e7

                # 视频中的每帧共享下面的变量
                # prepare params and tracker for ensemble
                num_det_ensemble = self.num_det_models
                num_map_ensemble = 1
                num_ensemble = num_det_ensemble * num_map_ensemble
                list_residuals = [[] for _ in range(num_ensemble)]
                list_all_predictions = [[] for _ in range(num_ensemble)]
                list_trk = [Tracker_2_w_feature(0.3) for _ in range(num_ensemble)]
                list_previous_frame_helmets = [None for _ in range(num_ensemble)]
                list_previous_assigned_label = [None for _ in range(num_ensemble)]
                list_previous_results = [None for _ in range(num_ensemble)]
                list_params_set = [{"random": None, "determined": None, "lost_track_frame": 0, "side_fixed": False,
                                    "conf_threshold": 0.4, "num_harddrop": 0, "num_softdrop": 0, "base_angle": 0.0,
                                    "hist_rot_angles": [], "xy_location_filtered": None, "team_provided": False,
                                    "neglect_far_targets": False} for _ in range(num_ensemble)]
                frame_info = {"game_play": game_play, "view": view, "frame": 1}
                list_tfh = [TeamFeaturesHolder(num_features=128, update_freq=100) for _ in range(num_ensemble)]
                batch_data = Batch_data()

                if labels is not None:
                    _labels = labels[labels["game_play"] == frame_info["game_play"]]
                    _labels = _labels[_labels["view"] == view]

                # 这个for循环做完，会进行目标追踪
                for f in frames:
                    frame_info["frame"] = f
                    ret, img = cap.read()

                    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.preprocess_rgb(img_array) # 像素值/255，新增1个维度

                    batch_data.add(img, img_array, f)

                    if not (f == frames[
                        -1] or batch_data.length == self.inference_batch):  # if batch size>1, stack inputs
                        continue

                    frame_info["frame"] = batch_data.frames # [f]

                    # ----- HELMETS DETECTION -----
                    S = time.time()
                    batch_list_current_frame_helmets, batch_list_tlbr_boxes = self.run_detection_ensemble_batch(
                        batch_data.imgs,
                        **frame_info)
                    # [[df1, ..., dfn]], [[tlbr_boxes1, ..., tlbr_boxesn]]
                    # df (num_box, 10)
                    # tlbr_boxes (num_box, 4)

                    if time_measurement: print("det", time.time() - S)
                    S = time.time()

                    # [df1, ..., dfn], [tlbr_boxes1, ..., tlbr_boxesn], frame, img, img_array
                    # 这个for循环做完了会batch_data.reset()，即batch对象状态清空
                    for _list_current_frame_helmets, _list_tlbr_boxes, frame, img, img_array in zip(
                            batch_list_current_frame_helmets,
                            batch_list_tlbr_boxes,
                            batch_data.frames,
                            batch_data.imgs,
                            batch_data.img_arrays):
                        print("\r----- predicting {}/{} -----".format(frame, len(frames)), end="")
                        frame_info["frame"] = frame
                        current_tracking = _df_tracking[_df_tracking["frame"] == frame]

                        # copy box and df
                        list_tlbr_boxes = []  # [tlbr_boxes1, ..., tlbr_boxesn]
                        list_current_frame_helmets = []  # [df1, ..., dfn]

                        for d_idx in range(num_det_ensemble):
                            for m_idx in range(num_map_ensemble):
                                list_tlbr_boxes.append(_list_tlbr_boxes[d_idx])
                                list_current_frame_helmets.append(_list_current_frame_helmets[d_idx].copy())

                        # for each detection model, run mapping-registration-tracking
                        # 这个for循环之后也就是外层for循环结束时候，会batch_data.reset()，即batch对象状态清空
                        for ensemble_idx in range(num_ensemble):

                            print("\r----- predicting {}/{}_{} -----".format(frame, len(frames), ensemble_idx), end="")
                            tfh = list_tfh[ensemble_idx]
                            trk = list_trk[ensemble_idx]
                            # current_frame_helmets = list_current_frame_helmets[ensemble_idx]
                            previous_results = list_previous_results[ensemble_idx]               # None
                            previous_frame_helmets = list_previous_frame_helmets[ensemble_idx]   # None
                            previous_assigned_label = list_previous_assigned_label[ensemble_idx] # None
                            tlbr_boxes = list_tlbr_boxes[ensemble_idx]           # tlbr_boxesi
                            # params_set = list_params_set[ensemble_idx]
                            residuals = list_residuals[ensemble_idx]             # []
                            all_predictions = list_all_predictions[ensemble_idx] # []

                            # ----- MAPPING -----
                            # current_frame_helmets, params_set = self.run_mapping(img, tlbr_boxes, current_frame_helmets, f_columns, tfh, params_set)
                            if len(tlbr_boxes) > 0:
                                if ensemble_idx % num_map_ensemble == 0: # 都满足
                                    l_current_frame_helmets = list_current_frame_helmets[
                                                              ensemble_idx:(ensemble_idx + num_map_ensemble)] # [dfi]
                                    l_params_set = list_params_set[ensemble_idx:(ensemble_idx + num_map_ensemble)] # [adict]
                                    l_tfh = list_tfh[ensemble_idx:(ensemble_idx + num_map_ensemble)] # [tfh]
                                    l_current_frame_helmets, l_params_set = self.run_mapping_ensemble(img,  # 这一帧处理后的图像
                                                                                                      tlbr_boxes, # tlbr_boxesi
                                                                                                      l_current_frame_helmets, # [dfi]
                                                                                                      f_columns, # 0-127的字符串字典
                                                                                                      l_tfh, # 第ensemble_idx个检测模型对应的[tfh]
                                                                                                      l_params_set)  # 第ensemble_idx个检测模型对应的[adict]
                                    # 用新添加了map坐标、team特征和队伍信息的df替换掉之前的df
                                    # 用新的字典替换之前旧的字典
                                    for j, [helmets, params] in enumerate(zip(l_current_frame_helmets, l_params_set)):
                                        list_current_frame_helmets[ensemble_idx + j] = helmets
                                        list_params_set[ensemble_idx + j] = params

                            # current_frame_helmets：加入map坐标、team128维特征、team分组信息的dataframe
                            # params_set： 加入一些新键值对的字典
                            current_frame_helmets = list_current_frame_helmets[ensemble_idx]
                            params_set = list_params_set[ensemble_idx]

                            if time_measurement: print("map", time.time() - S)
                            S = time.time()

                            # ----- POINTS TO POINTS REGISTRATION -----
                            # if few bbox, skip registration
                            if not len(current_frame_helmets) < 2:
                                # PREPROCESS of REGISTRATION
                                try:
                                    with tf.device('/CPU:0'):  # cpu is faster than gpu
                                        # current_tracking：当前game_play当前frame的tracking数据
                                        # 用置信度和tracking中点的数量过滤了一波检测框，过滤后的检测框df为current_frame_helmets
                                        # 过滤掉的检测框集中到了current_frame_helmets_low_conf
                                        # params_set的键值对有更新，且其中真实坐标除以20
                                        # test_inputs： 运动员的坐标和所属队伍（0/1）
                                        # all_data：包含test_inputs中内容的大杂烩
                                        current_frame_helmets, current_frame_helmets_low_conf, \
                                        params_set, test_inputs, all_data = self.preprocess_registration(
                                            current_frame_helmets, current_tracking,
                                            trk, params_set,
                                            start_frame=start_frame, view_frequency=view_frequency, **frame_info)

                                except:
                                    return None  # notfound error instead of exception

                                if time_measurement: print("pre registration", time.time() - S)
                                S = time.time()

                            if len(current_frame_helmets) < 2:
                                if (previous_frame_helmets is not None) and (previous_assigned_label is not None):
                                    results = previous_results
                                    current_frame_helmets = previous_frame_helmets
                                    assigned_label = previous_assigned_label
                                    residual = 1.0e-3
                                    params_set["hist_rot_angles"].append(params_set["base_angle"])
                                    test_inputs, all_data = self.preprocess_registration(
                                        current_frame_helmets, current_tracking,
                                        trk, params_set,
                                        start_frame=start_frame, view_frequency=view_frequency,
                                        only_return_inputs=True, **frame_info)
                            else:
                                # MAIN REGISTRATION
                                try:
                                    with tf.device('/CPU:0'):  # cpu is faster than gpu
                                        results, assigned_label, params_set, current_frame_helmets = self.run_registration(
                                            current_frame_helmets,
                                            current_frame_helmets_low_conf,
                                            params_set,
                                            test_inputs,
                                            all_data,
                                            tfh,
                                            f_columns,
                                            **frame_info)


                                except:
                                    print("RETRY")
                                    if params_set["random"] is not None:
                                        params_set["random"]["zoom_params"] = params_set["random"][
                                                                                  "zoom_params"] * tf.constant(
                                            [1.0, 1.0, 10.0], tf.float32)

                                    try:
                                        with tf.device('/CPU:0'):
                                            results, assigned_label, params_set, current_frame_helmets = self.run_registration(
                                                current_frame_helmets,
                                                current_frame_helmets_low_conf,
                                                params_set,
                                                test_inputs,
                                                all_data,
                                                tfh,
                                                f_columns,
                                                **frame_info)

                                    except:
                                        current_output = pd.concat(all_game_predictions, axis=0)
                                        current_output["height"] = 1
                                        current_output = current_output[~current_output[
                                            ["video_frame", "left", "width", "top", "height"]].duplicated()]
                                        return current_output  # LOW SCORE instead of exception

                                if time_measurement: print("main registration", time.time() - S)
                                S = time.time()
                                residual = results["residual"].numpy()
                                residuals.append(residual)

                                if DRAW_BBOX and frame % view_frequency == 0:
                                    self.draw_bbox(img_array,
                                                   current_frame_helmets[["top", "left", "height", "width"]].values,
                                                   # frame_label[["top", "left", "height", "width"]].values,
                                                   save_only=False,
                                                   )
                                if DRAW_REGI and frame % view_frequency == 0:
                                    team_color_gt = all_data["team_labels"].numpy().reshape(-1)
                                    results_xy = results["trans_sources"].numpy()[:, :2]
                                    team_color = results["trans_sources"].numpy()[:, 2] * 10
                                    gt_loc_all = all_data["all_locations"]
                                    plt.scatter(gt_loc_all[:, 0][team_color_gt < 0.5],
                                                gt_loc_all[:, 1][team_color_gt < 0.5], c="blue", alpha=0.2)
                                    plt.scatter(results_xy[..., 0][team_color < 0.5],
                                                results_xy[..., 1][team_color < 0.5], c="blue")
                                    plt.scatter(gt_loc_all[:, 0][team_color_gt > 0.5],
                                                gt_loc_all[:, 1][team_color_gt > 0.5], c="red", alpha=0.2)
                                    plt.scatter(results_xy[..., 0][team_color > 0.5],
                                                results_xy[..., 1][team_color > 0.5], c="red")
                                    plt.title("ICP registration error {}".format(residual))
                                    plt.show()

                            # make submit dataframe
                            predictions = np.round(
                                current_frame_helmets[["left", "width", "top", "height"]].copy()).astype(int)
                            predictions["label"] = assigned_label
                            predictions["video_frame"] = frame_info["game_play"] + "_" + frame_info["view"] + "_" + str(
                                frame_info["frame"])
                            all_predictions.append(predictions)

                            # scoring for validation
                            if labels is not None:
                                frame_label = _labels[_labels["frame"] == frame]
                                scorer = NFLAssignmentScorer(frame_label, impact_weight=1)
                                scorer_w = NFLAssignmentScorer(frame_label)
                                frame_score = scorer.score(predictions)
                                frame_score_w = scorer_w.score(predictions)
                                if ensemble_idx == 0:
                                    all_labels.append(frame_label)
                                frame_scores.append(frame_score)
                                print(frame_score, frame_score_w)

                            # ----- Accumurate data in Tracker -----
                            try:
                                trk.predict_and_add(assigned_player=assigned_label,
                                                    current_boxes=current_frame_helmets[
                                                        ["top", "left", "bottom", "right"]].values,
                                                    locations=all_data["all_locations"][results["final_assignment"]],
                                                    player_feature=current_frame_helmets[f_columns].values,
                                                    weight=(-np.log(results["xy_residual"])),  # **2,
                                                    icp_errors=np.log(results["xy_residual"]),
                                                    conf=current_frame_helmets["conf"].values,
                                                    **frame_info)

                            except:
                                current_output = pd.concat(all_game_predictions, axis=0)
                                current_output["height"] = (current_output["height"].values * 0.65).astype(int)
                                current_output = current_output[
                                    ~current_output[["video_frame", "left", "width", "top", "height"]].duplicated()]
                                return current_output  # lowscore

                            if time_measurement: print("aft registration", time.time() - S)
                            S = time.time()

                            previous_frame_helmets = current_frame_helmets.copy()
                            previous_assigned_label = assigned_label.copy()
                            previous_results = results

                            list_previous_frame_helmets[ensemble_idx] = previous_frame_helmets
                            list_previous_assigned_label[ensemble_idx] = previous_assigned_label
                            list_previous_results[ensemble_idx] = previous_results
                            list_params_set[ensemble_idx] = params_set
                            list_residuals[ensemble_idx] = residuals
                            list_all_predictions[ensemble_idx] = all_predictions
                    batch_data.reset()
                    if frame == end_frame:
                        break

                end_frame = frame
                print("\r----- reassignmenting using tracking data-----", end="")

                list_df_preds = []
                for frame in range(1, end_frame + 1):
                    frame_info["frame"] = frame
                    fusion_boxes, fusion_confs, assigned_label = wbf_ensemble_reassign_player_label(list_trk,
                                                                                                    **frame_info)
                    predictions = pd.DataFrame(fusion_boxes, columns=["top", "left", "bottom", "right"])
                    predictions["height"] = predictions["bottom"] - predictions["top"]
                    predictions["width"] = predictions["right"] - predictions["left"]
                    predictions = np.round(predictions[["left", "width", "top", "height"]].copy()).astype(int)
                    predictions["label"] = assigned_label
                    predictions["video_frame"] = game_play + "_" + view + "_" + str(frame)
                    predictions = predictions[~predictions[["left", "width", "top", "height"]].duplicated()]
                    list_df_preds.append(predictions)

                all_game_predictions.append(pd.concat(list_df_preds, axis=0))
                print("----- reassignmenting finished -----")

        if labels is None:
            return pd.concat(all_game_predictions, axis=0)
        else:
            return pd.concat(all_game_predictions, axis=0), pd.concat(all_labels, axis=0)

    def preprocess_rgb(self, file_or_array):
        if type(file_or_array) == str:
            rgb = tf.io.read_file(file_or_array)
            rgb = tf.image.decode_jpeg(rgb, channels=3)
        else:
            if type(file_or_array) == list:
                rgb = tf.concat(file_or_array, axis=-1)
            else:
                rgb = file_or_array
        rgb = tf.cast(rgb, tf.float32) / 255.0
        rgb = rgb[tf.newaxis, :, :, :]
        return rgb

    def preprocess_inputs(self,
                          all_locations,
                          all_players,  # all_motions,
                          locations=None,  # label
                          players=None,  # label
                          img_height=720,
                          img_width=1280,
                          **kwargs):

        all_locations = tf.cast(all_locations, tf.float32) / 20.0
        all_locations = tf.reshape(all_locations, [-1, 2])
        # all_motions = tf.cast(all_motions, tf.float32)
        # all_motions = tf.reshape(all_motions, [-1,2])

        if locations is not None:
            locations = tf.cast(locations, tf.float32) / 20.0
            locations = tf.reshape(locations, [-1, 2])
            gt_points = locations[tf.newaxis, :, :]
        else:
            gt_points = None

        team_labels = tf.constant(np.array(["H" in p for p in all_players]).reshape(-1, 1).astype(np.float32))

        inputs_registration = {
            "all_locations": all_locations,
            "team_labels": team_labels}

        all_data = {
            "locations": locations,
            "team_labels": team_labels,
            "players": players,
            # "all_motions": all_motions,
            "all_locations": np.array(all_locations),
            "all_players": np.array(all_players),
        }
        return inputs_registration, all_data


def set_seeds(num=111):
    tf.random.set_seed(num)
    np.random.seed(num)
    random.seed(num)
    os.environ["PYTHONHASHSEED"] = str(num)


def run_test(nfl_model):
    set_seeds(111)
    debug = False

    conf_thresh = 0.02
    tracking_df = te_tracking
    helmets_df = te_helmets
    if debug:
        tracking_df = tr_tracking.copy()
        helmets_df = tr_helmets.copy()

    helmets_df = helmets_df[helmets_df["conf"] > conf_thresh]

    split_names = helmets_df["video_frame"].str.rsplit('_', n=2, expand=True).rename(
        columns={0: 'game_play', 1: 'view', 2: "frame"})
    helmets_df = pd.concat([helmets_df, split_names], axis=1)
    helmets_df["frame"] = helmets_df["frame"].astype('int')
    helmets_df = helmets_df.sort_values(['game_play', "frame"])

    if len(helmets_df) < 100000 and debug == False:
        helmets_df = helmets_df[
            (helmets_df["game_play"] == helmets_df["game_play"].values[0]) & (helmets_df["view"] == "Sideline")]
        helmets_df.to_csv('submission.csv', index=False)
        return None
    if debug:
        game_play = '58005_001254'
        view = 'Sideline'
        helmets_df = helmets_df[(helmets_df["game_play"] == game_play) & (helmets_df["view"] == view)]

    video_path = "../input/nfl-health-and-safety-helmet-assignment/test/"
    if debug:
        video_path = "../input/nfl-health-and-safety-helmet-assignment/train/"

    results = nfl_model.test_predict_ensemble_batch(tracking_df, helmets_df, video_path)
    if results is not None:
        results.to_csv('submission.csv', index=False)
    if debug:
        scorer = NFLAssignmentScorer(labels[labels["video"] == "{}_{}.mp4".format(game_play, view)])
        print(scorer.score(results))
    return results


def build_val_model():
    K.clear_session()
    set_seeds(111)
    debug = True

    model_params = {"input_shape": (512, 896, 3),
                    "output_shape": (128, 224),
                    "weight_file": {"map": SRC_PATH + "/model/weights/map/final_weights.h5",
                                    "team": SRC_PATH + "/model/weights/team/final_weights.h5",
                                    "det": SRC_PATH + "/model/weights/det_base/final_weights.h5",
                                    "detL": [["effv2s", SRC_PATH + "/model/weights/det_v2s/final_weights.h5"],
                                             ["effv2m", SRC_PATH + "/model/weights/det_v2m/final_weights.h5"],
                                             ["effv2l", SRC_PATH + "/model/weights/det_v2l/final_weights.h5"],
                                             ["effv2xl", SRC_PATH + "/model/weights/det_v2xl/final_weights.h5"],
                                             ],
                                    },
                    "is_train_model": False,
                    "inference_batch": 1,
                    }
    if not ENSEMBLE:
        model_params["weight_file"]["detL"] = model_params["weight_file"]["detL"][0:1]
    nfl = NFL_Predictor(**model_params)
    return nfl


def run_val(nfl_model,
            game_play='58005_001254',
            view='Sideline',
            ):
    # K.clear_session()
    set_seeds(111)

    conf_thresh = 0.02
    tracking_df = te_tracking
    helmets_df = te_helmets

    tracking_df = tr_tracking.copy()
    helmets_df = tr_helmets.copy()

    helmets_df = helmets_df[helmets_df["conf"] > conf_thresh]

    split_names = helmets_df["video_frame"].str.rsplit('_', n=2, expand=True).rename(
        columns={0: 'game_play', 1: 'view', 2: "frame"})
    helmets_df = pd.concat([helmets_df, split_names], axis=1)
    helmets_df["frame"] = helmets_df["frame"].astype('int')
    helmets_df = helmets_df.sort_values(['game_play', "frame"])

    helmets_df = helmets_df[(helmets_df["game_play"] == game_play) & (helmets_df["view"] == view)]

    video_path = "../input/nfl-health-and-safety-helmet-assignment/train/"

    ##nfl = NFL_Predictor(**model_params)#_deepsort
    # 传入完整的tracking数据和该场比赛对应视角的helmets数据，得到最后的预测结果
    temp_results = nfl_model.test_predict_ensemble_batch(tracking_df, helmets_df, video_path)
    # temp_results = nfl_model.test_predict_2(tracking_df, helmets_df, video_path)
    temp_results.to_csv('submission.csv', index=False)
    """
    scorer = NFLAssignmentScorer(labels[labels["video"]=="{}_{}.mp4".format(game_play, view)])
    print(scorer.score(temp_results))
    scorer = NFLAssignmentScorer(labels[labels["video"]=="{}_{}.mp4".format(game_play, view)], impact_weight=1)
    print("no weight score", scorer.score(temp_results))

    scorer = NFLAssignmentScorer(labels[labels["video"]=="{}_{}.mp4".format(game_play, view)],
                                check_constraints=False,
                                 impact_weight=1,
                                 check_iou_only=True)
    print("no assignment score", scorer.score(temp_results))
    """
    return temp_results


DEBUG = True
if DEBUG:
    # samples
    game_play_and_views = [['57790_002792', 'Endzone'],
                           ['57992_000350', 'Sideline'],
                           ]

    ENSEMBLE = True  # if false, predict by single model
    DRAW_PREREGI = True  # show preprocessing registration results using previous frame
    DRAW_REGI = True  # show final registration results
    DRAW_BBOX = True  # show bounding boxes
    VIEW_FREQUENCY = 20  # draw outputs every N frames

    nfl_model = build_val_model()
    for game_play, view in game_play_and_views:
        S = time.time()
        run_val(nfl_model, game_play, view)
        print(time.time() - S, "SEC / SINGLE PLAY")
else:
    ENSEMBLE = True
    DRAW_PREREGI = False  # show registration results using previous frame
    DRAW_REGI = False
    DRAW_BBOX = False
    VIEW_FREQUENCY = 1e7
    nfl_model = build_val_model()
    run_test(nfl_model)