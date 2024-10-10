#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import math
import logging
from collections import deque, defaultdict
from typing import Dict, List, Any
import json
import random
import glob

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from ss_baselines.common.utils import observations_to_image
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation,
    NpEncoder
)
from ss_baselines.savi.ppo.policy import AudioNavBaselinePolicy, AudioNavSMTPolicy
from ss_baselines.savi.ppo.ppo import PPO
from ss_baselines.savi.ppo.slurm_utils import (
    EXIT,
    REQUEUE,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from ss_baselines.savi.models.rollout_storage import RolloutStorage, ExternalMemory
from ss_baselines.savi.models.belief_predictor import BeliefPredictor
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from soundspaces.tasks.nav import LocationBelief, CategoryBelief, SpectrogramSensor

from agents.vlm_agents import LLM_Agent
from vistools.vis_tools import Decision_Generation_Vis, Visualize
from src.vlm import CogVLM2
from src.SystemPrompt import (
    form_prompt_for_PerceptionVLM_Step1, 
    form_prompt_for_PerceptionVLM_Step234,
    form_prompt_for_FN,
    form_prompt_for_FN_Step1,
    form_prompt_for_DecisionVLM_Frontier,
    form_prompt_for_DecisionVLM_History,

    form_prompt_for_DecisionVLM_MetaPreprocess,
    form_prompt_for_Module_Decision,
    Perception_weight_decision,
    Perception_weight_decision4,
    Perception_weight_decision26,
    extract_scene_image_description_results,
    extract_scene_object_detection_results,
    extract_scenario_exploration_analysis_results
)

import utils.pose as pu

import utils.visualization as vu

from constants import (
    coco_categories, coco_categories_hm3d2mp3d,
    gibson_coco_categories, color_palette, category_to_id, object_category
)

from scipy.signal import correlate
import cv2
import librosa
import soundfile as sf
from skimage import measure

##### 如果是安装了yolov9环境，请将其注释取消
# from detect_yolov9 import Detect
from detect.ultralytics import YOLOv10

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@baseline_registry.register_trainer(name="savi")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        self._static_smt_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]

        if not ppo_cfg.use_external_memory:
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                extra_rgb=self.config.EXTRA_RGB
            )
        else:
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=smt_cfg.use_label_belief,
                use_location_belief=smt_cfg.use_location_belief
            )

            if ppo_cfg.use_belief_predictor:
                belief_cfg = ppo_cfg.BELIEF_PREDICTOR
                smt = self.actor_critic.net.smt_state_encoder
                self.belief_predictor = BeliefPredictor(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                        smt.hidden_state_size, self.envs.num_envs,
                                                        ).to(device=self.device)
                for param in self.belief_predictor.parameters():
                    param.requires_grad = False

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

        if self.config.RESUME:
            ckpt_dict = self.load_checkpoint('data/models/smt_with_pose/ckpt.400.pth', map_location="cpu")
            self.agent.actor_critic.net.visual_encoder.load_state_dict(self.search_dict(ckpt_dict, 'visual_encoder'))
            self.agent.actor_critic.net.goal_encoder.load_state_dict(self.search_dict(ckpt_dict, 'goal_encoder'))
            self.agent.actor_critic.net.action_encoder.load_state_dict(self.search_dict(ckpt_dict, 'action_encoder'))

        if ppo_cfg.use_external_memory and smt_cfg.freeze_encoders:
            self._static_smt_encoder = True
            self.actor_critic.net.freeze_encoders()

        self.actor_critic.to(self.device)

    @staticmethod
    def search_dict(ckpt_dict, encoder_name):
        encoder_dict = {}
        for key, value in ckpt_dict['state_dict'].items():
            if encoder_name in key:
                encoder_dict['.'.join(key.split('.')[3:])] = value

        return encoder_dict

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            external_memory = None
            external_memory_masks = None
            if self.config.RL.PPO.use_external_memory:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                external_memory_features
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device),
            external_memory_features,
        )

        if self.config.RL.PPO.use_belief_predictor:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in [LocationBelief.cls_uuid, CategoryBelief.cls_uuid]:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def train_belief_predictor(self, rollouts):
        bp = self.belief_predictor
        num_epoch = 5
        num_mini_batch = 1

        advantages = torch.zeros_like(rollouts.returns)
        value_loss_epoch = 0
        running_regressor_corrects = 0
        num_sample = 0

        for e in range(num_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    external_memory,
                    external_memory_masks,
                ) = sample

                bp.optimizer.zero_grad()

                inputs = obs_batch[SpectrogramSensor.cls_uuid].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch)

                masks = (torch.sum(torch.reshape(obs_batch[SpectrogramSensor.cls_uuid],
                        (obs_batch[SpectrogramSensor.cls_uuid].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
                transformed_gts = torch.stack([gts[:, 1], -gts[:, 0]], dim=1)
                masked_preds = masks.expand_as(preds) * preds
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts)

                bp.before_backward(loss)
                loss.backward()
                # self.after_backward(loss)

                bp.optimizer.step()
                value_loss_epoch += loss.item()

                rounded_preds = torch.round(preds)
                bitwise_close = torch.bitwise_and(torch.isclose(rounded_preds[:, 0], transformed_gts[:, 0]),
                                                  torch.isclose(rounded_preds[:, 1], transformed_gts[:, 1]))
                running_regressor_corrects += torch.sum(torch.bitwise_and(bitwise_close, masks.bool().squeeze(1)))
                num_sample += torch.sum(masks).item()

        value_loss_epoch /= num_epoch * num_mini_batch
        if num_sample == 0:
            prediction_accuracy = 0
        else:
            prediction_accuracy = running_regressor_corrects / num_sample

        return value_loss_epoch, prediction_accuracy

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if ppo_cfg.use_external_memory:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # add_signal_handlers()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        if ppo_cfg.use_external_memory:
            memory_dim = self.actor_critic.net.memory_dim
        else:
            memory_dim = None

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            memory_dim,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state(model_dir=self.config.MODEL_DIR)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optimizer_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_scheduler_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set():
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optimizer_state=self.agent.optimizer.state_dict(),
                                lr_scheduler_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            model_dir=self.config.MODEL_DIR
                        )
                        requeue_job()
                    return

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    # writer.add_scalars("metrics", metrics, count_steps)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                writer.add_scalar("Policy/value_loss", value_loss, count_steps)
                writer.add_scalar("Policy/policy_loss", action_loss, count_steps)
                writer.add_scalar("Policy/entropy_loss", dist_entropy, count_steps)
                writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()
    
    def calculate_itd(self, left_channel, right_channel, sampling_rate):
        # 计算交叉相关
        correlation = correlate(left_channel, right_channel, mode='full')
        # 找到最大相关的位置
        max_corr_index = np.argmax(correlation)
        # 计算时差
        sample_difference = max_corr_index - len(left_channel) + 1
        time_difference = sample_difference / sampling_rate
        return time_difference
    
    def calculate_ild(self, left_channel, right_channel):
        # 计算左右耳信号的能量
        left_power = np.sum(left_channel**2)
        right_power = np.sum(right_channel**2)
        # 计算能量差异
        ild = left_power - right_power

        power_all = left_power + right_power

        return ild, power_all
    
    # 定义频谱分析估计函数
    def estimate_direction_from_spectrogram(self, left_channel, right_channel):
        S_left = np.abs(librosa.stft(left_channel))
        S_right = np.abs(librosa.stft(right_channel))
        spectral_diff = np.mean(S_left - S_right, axis=1)
        
        # if np.mean(spectral_diff) > 0:
        #     return "左侧"
        # else:
        #     return "右侧"
        return np.mean(spectral_diff)
    
    def spectrogram_to_audio(self, spectrogram, sampling_rate=44100, hop_length=512, n_fft=1024):
            # 分别提取左右耳的频谱图
            left_spectrogram = spectrogram[:, :, 0]
            right_spectrogram = spectrogram[:, :, 1]

            # 对左右耳的频谱进行逆STFT变换，得到时域音频
            left_audio = librosa.istft(left_spectrogram, hop_length=hop_length, length=None)
            right_audio = librosa.istft(right_spectrogram, hop_length=hop_length, length=None)

            # 合并为双声道音频信号
            stereo_audio = np.vstack((left_audio, right_audio)).T

            return stereo_audio
        
    def save_binaural_audio(self, audio_data, sampling_rate=16000, file_name='binaural_audio.wav'):
        # audio_data 是大小为 2 x N 的矩阵，其中 2 表示两个通道（左耳和右耳）
        # 转置矩阵以适应 stereo 格式 (N, 2)
        stereo_audio = audio_data.T
    
        # 将双声道音频数据保存为 WAV 文件
        sf.write(file_name, stereo_audio, sampling_rate)
    
    def Frontiers(self, full_map_pred):
        # ------------------------------------------------------------------
        ##### Get the frontier map and filter
        # ------------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        full_w = full_map_pred.shape[1]
        local_ex_map = np.zeros((full_w, full_w))
        local_ob_map = np.zeros((full_w, full_w))

        local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

        show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

        contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            contour = max(contours, key = cv2.contourArea)
            cv2.drawContours(local_ex_map,contour,-1,1,1)

        # clear the boundary
        local_ex_map[0:2, 0:full_w]=0.0
        local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
        local_ex_map[0:full_w, 0:2]=0.0
        local_ex_map[0:full_w, full_w-2:full_w]=0.0

        target_edge = local_ex_map-local_ob_map
        # print("local_ob_map ", self.local_ob_map[200])
        # print("full_map ", self.full_map[0].cpu().numpy()[200])

        target_edge[target_edge>0.8]=1.0
        target_edge[target_edge!=1.0]=0.0

        wall_edge = local_ex_map - target_edge

        # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours)>0:
        #     dst = np.zeros(wall_edge.shape)
        #     cv2.drawContours(dst, contours, -1, 1, 1)

        # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
        Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
        img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = []
        Goal_area_list = []
        dict_cost = {}
        for i in range(1, len(props)):
            if props[i].area > 4:
                dict_cost[i] = props[i].area

        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

            for i, (key, value) in enumerate(dict_cost):
                Goal_edge[img_label == key + 1] = 1
                Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
                Goal_area_list.append(value)
                if i == 3:
                    break
            # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

            # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
            # if frontiers is not None:
            #     for frontier in frontiers:
            #         x1, y1, x2, y2 = frontier[0]
            #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return Wall_lines, Goal_area_list, Goal_edge, Goal_point
    
    def Objects_Extract(self, args, full_map_pred, use_sam):

        semantic_map = full_map_pred[4:]

        dst = np.zeros(semantic_map[0, :, :].shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

        Object_list = {}
        for i in range(len(semantic_map)):
            if semantic_map[i, :, :].sum() != 0:
                Single_object_list = []
                se_object_map = semantic_map[i, :, :].cpu().numpy()
                se_object_map[se_object_map>0.1] = 1
                se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
                contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    if len(cnt) > 30:
                        epsilon = 0.05 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        Single_object_list.append(approx)
                        cv2.polylines(dst, [approx], True, 1)
                if len(Single_object_list) > 0:
                    if use_sam:
                        Object_list[object_category[i]] = Single_object_list
                    else:
                        if 'mp3d' in args.task_config:
                            Object_list[object_category[i]] = Single_object_list
                        elif 'hm3d' in args.task_config:
                            Object_list[hm3d_category[i]] = Single_object_list
        return Object_list

    def _eval_checkpoint(
        self,
        args,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        '''
        重写eval要求:
        (1) 加入VLM思维链 用于连结每个模态的信息 暂时不知道怎么写接口
        (2) 加入语义音频积累打分算法，什么时候停止
        (3) 加入FMM 本地策略
        时间计划:
        9.1-9.10 验证原始代码 了解每个部分的含义 准备重构代码 V
        9.10-9.20 加入语义建图 随机实验
        9.20-9.30 加入FMM 辛苦一点 把VLM+音频打分加上
        10月 消融实验
        '''
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        
        self.config.defrost()
        self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.05
        self.config.freeze()

        robot = LLM_Agent(args, 0, self.device)

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        # Load ALL MODELS
        if args.yolo == 'yolov9':
            # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
            pass
        else:
            yolo = YOLOv10.from_pretrained(args.yolo_weights)

        # Load VLM
        # vlm = VLM(args.vlm_model_id, args.hf_token, device)
        base_url = args.base_url 
        cogvlm2 = CogVLM2(base_url) 


        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()


        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        if self.config.RL.PPO.use_belief_predictor and "belief_predictor" in ckpt_dict:
            self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

        self.metric_uuids = []
        # get name of performance metric, e.g. "spl"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if config.DISPLAY_RESOLUTION != model_resolution:
            obs_copy = resize_observation(observations, model_resolution)
        else:
            obs_copy = observations
        batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                            'oracle_action_sensor'])
        #####
        ## 定义机器人中间变量
        #####
        pre_g_points = []

        target_point = []
        pre_goal_points = []
        cur_goal_points = []
        history_nodes = []
        history_score = []
        history_count = []
        history_states = []

        audio_goal = None
        goal_points = [0,0]
        full_map = []
        visited_vis = []
        pose_pred = []
        agent_objs = {} # 记录单个时间步内每个智能体的目标检测信息
        #####

        # 9.7+9.8任务: 构建简单的语义地图+可视化(构建整体框架)
        # 示例数据
        
        # print(batch)
        # import cv2
        # rgb=batch['rgb'][0].cpu().numpy().astype(np.uint8)
        # print(rgb.shape)
        # cv2.imwrite('test_img.png', rgb)
    
        # 将频谱转换为双声道音频
        # stereo_audio = spectrogram_to_audio(observations[0]['spectrogram'])
        # 保存音频文件（WAV格式）
        # sf.write('stereo_audio.wav', stereo_audio, 44100)
        # save_binaural_audio(observations[0]['audiogoal'], sampling_rate=16000, file_name='binaural_audiogoal.wav')   

        robot.reset()
        robot.mapping(batch)

        local_map1, _ = torch.max(robot.local_map.unsqueeze(0), 0)
        full_map.append(robot.local_map)
        visited_vis.append(robot.visited_vis)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = robot.planner_pose_inputs
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, robot.local_map[0, :, :].cpu().numpy().shape)

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        pos = (
                (start_x * 100. / args.map_resolution - gy1)
                * 480 / robot.visited_vis.shape[0],
                (robot.visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                * 480 / robot.visited_vis.shape[1],
                np.deg2rad(-start_o)
            )
        pose_pred.append(pos)

        full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
        full_map_pred, _ = torch.max(full_map2, 0)
        _, full_Frontier_list, full_target_edge_map, full_target_point_map = self.Frontiers(full_map_pred)

        agents_seg_list = self.Objects_Extract(args, full_map_pred, args.use_sam)

        pre_goal_points.clear()
        if len(cur_goal_points) > 0:
            pre_goal_points = cur_goal_points.copy()
            cur_goal_points.clear()

        rgb = batch['rgb'][0].cpu().numpy().astype(np.uint8)
        dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        vis_ep_dir = '{}/episodes/eps_{}/Agent1_vis'.format(
                        dump_dir, robot.episode_n)
        if not os.path.exists(vis_ep_dir):
            os.makedirs(vis_ep_dir)
        fn2 = '{}/episodes/eps_{}/Agent1_vis/IMGStep-{}.png'.format(
                        dump_dir, robot.episode_n,
                        robot.l_step)
        cv2.imwrite(fn2, rgb)    
        if args.yolo == 'yolov9':
            agent_objs[f"robot_{0}"] = yolo.run(rgb) # 记录一个时间步内每个智能体的目标检测信息
        else:
            yolo_output = yolo(source=rgb,conf=0.2)
            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
            agent_objs[f"robot_{0}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
        logging.info(agent_objs)

        # 示例数据

        # # 假设我们有一个65x26的频谱矩阵
        # spectrogram = data[0][:, :, 1]

        # 创建热图
        # plt.imshow(spectrogram, cmap='viridis', aspect='auto', interpolation='nearest')
        # plt.colorbar()  # 显示颜色条，说明颜色对应强度
        # plt.title('Spectrogram Intensity Heatmap')
        # plt.xlabel('Time Frames')
        # plt.ylabel('Frequency Bins')

        # 保存图像到本地
        # plt.savefig("right_spectrogram_intensity_heatmap.png")

        intensity_ratio = np.sum(batch['spectrogram'].cpu().numpy()[0][:, :, 0]) / np.sum(batch['spectrogram'].cpu().numpy()[0][:, :, 1] + 1e-10)  # 添加一个小数以避免除以0的情况
        print("!!!!!intensity_ratio:",intensity_ratio)

        if intensity_ratio > 1.001:
            audio_direction = 'Left side of the red arrow.'
            img_direction = 'Left side of image.'
        elif intensity_ratio <= 1.001 and intensity_ratio > 0.999:
            audio_direction = 'In the Front of the red arrow.'
            img_direction = 'Middle of image.'
        elif intensity_ratio <= 0.999:
            audio_direction = 'Right side of the red arrow.'
            img_direction = 'Right side of image.'


        # 展示用gt goal_name代替goal_name 正常情况应该重新设计提示
        goal_name = object_category[np.argmax(batch['category'].cpu().numpy()[0])]

        if len(full_target_point_map) > 0:
            full_Frontiers_dict = {}
            for j in range(len(full_target_point_map)):
                full_Frontiers_dict['frontier_' + str(j)] = f"<centroid: {full_target_point_map[j][0], full_target_point_map[j][1]}, number: {full_Frontier_list[j]}>"
            logging.info(f'=====> Origin Frontier: {full_Frontiers_dict}')

            if intensity_ratio > 1.001:
                # 朝向：右 声源：左
                if int(start_o % 360) == 0:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find('(') + 1:value.find(', ')]) >= start[0]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：上 声源：左
                elif int(start_o % 360) == 90:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find(', ') + 1:value.find(')')]) < start[1]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：左 声源：左
                elif int(start_o % 360) == 180:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find('(') + 1:value.find(', ')]) < start[0]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：下 声源：左
                elif int(start_o % 360) == 270:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find(', ') + 1:value.find(')')]) >= start[1]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value


            elif intensity_ratio <= 1.001 and intensity_ratio > 0.999:
                #声源：中！！！
                full_Frontiers_dict_del = full_Frontiers_dict
            
            elif intensity_ratio <= 0.999:  
                # 朝向：右 声源：右
                if int(start_o % 360) == 0:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find('(') + 1:value.find(', ')]) < start[0]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：上 声源：右
                elif int(start_o % 360) == 90:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find(', ') + 1:value.find(')')]) >= start[1]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：左 声源：右
                elif int(start_o % 360) == 180:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find('(') + 1:value.find(', ')]) >= start[0]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value
                # 朝向：下 声源：右
                elif int(start_o % 360) == 270:
                    # 过滤出左侧符合条件的键值对
                    filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                    if int(value[value.find(', ') + 1:value.find(')')]) < start[1]}
                    # 重新命名键
                    full_Frontiers_dict_del = {}
                    for i, (key, value) in enumerate(filtered_items.items()):
                        renamed_key = f'frontier_{i}'
                        full_Frontiers_dict_del[renamed_key] = value

            logging.info(f'=====> New Frontier: {full_Frontiers_dict_del}')


            if len(history_nodes) > 0:
                logging.info(f'=====> history_nodes: {history_nodes}')
                logging.info(f'=====> history_score: {history_score}')

            if len(pre_goal_points) > 0:
                # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict_del, goal_points=[], pre_goal_point=pre_goal_points)
            else:
                # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                        robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict_del, goal_points=[], pre_goal_point=None)
            # full_rgb = np.hstack((rgb, sem_map)


            #### 感知LM，给出感知分数，
            Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM_Step1(goal_name, agent_objs[f'robot_{0}'], img_direction, args.yolo)
            _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                            return_string_probabilities=None, img=rgb)
            Perception_Rel, Perception_Pred = cogvlm2.COT2(User_Prompt1=Caption_Prompt, 
                                                       User_Prompt2=VLM_Perception_Prompt,
                                                       cot_pred1=Scene_Information,
                                                       return_string_probabilities="[Yes, No]", img=rgb)
            Perception_Rel = np.array(Perception_Rel)
            Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
            logging.info(f"robot-VLM_PerceptionPR: {Perception_PR}")




        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        if self.actor_critic.net.num_recurrent_layers == -1:
            num_recurrent_layers = 1
        else:
            num_recurrent_layers = self.actor_critic.net.num_recurrent_layers
        test_recurrent_hidden_states = torch.zeros(
            num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        if ppo_cfg.use_external_memory:
            test_em = ExternalMemory(
                self.config.NUM_PROCESSES,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                self.actor_critic.net.memory_dim,
            )
            test_em.to(self.device)
        else:
            test_em = None
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        self.count_max_category = np.zeros(batch['category'][0].shape[0])
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)
            batch['count_max_category'] = self.count_max_category

            descriptor_pred_gt = [[] for _ in range(self.config.NUM_PROCESSES)]
            for i in range(len(descriptor_pred_gt)):
                category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                location_prediction = batch['location_belief'].cpu().numpy()[i]
                category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                geodesic_distance = -1
                pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                if 'view_point_goals' in observations[i]:
                    pair += (observations[i]['view_point_goals'],)
                descriptor_pred_gt[i].append(pair)

                # left_channel = batch['spectrogram'][:, :, 0].cpu().numpy()
                # right_channel = batch['spectrogram'][:, :, 1].cpu().numpy()
                # sampling_rate = 44100  # 假设采样率为44.1kHz

                print("pose: ",batch['pose'])
                print("location_gt:",location_gt)
                # print("location_prediction:",location_prediction)

                # itd = self.calculate_itd(left_channel, right_channel, sampling_rate)
                # print(f"Interaural Time Difference (ITD): {itd} seconds")

                # ild, power_all = self.calculate_ild(left_channel, right_channel)
                # print(f"Interaural Level Difference (ILD): {ild} dB")
                # print(f"power_all: {power_all}")
                # spectral_estimation = self.estimate_direction_from_spectrogram(left_channel, right_channel)
                # print(f"spectral_estimation: {spectral_estimation}")

                self.count_max_category[category_prediction] += 1
                batch['count_max_category'][category_prediction] += 1


        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        self.actor_critic.eval()
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.eval()
        t = tqdm(total=self.config.TEST_EPISODE_COUNT)

        '''
        这个循环很有意思：将两个循环(大循环(1000个eps)和小循环(每个eps每次执行action后的循环)结合在一起了)
        '''
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            # 重写
            if len(stats_episodes) == 1:
                exit()
            current_episodes = self.envs.current_episodes()
            ### 设置迭代条件：20次为一个全局策略过程

            # print(robot.l_step) # 0

            if robot.l_step % args.num_local_steps == args.num_local_steps - 1 or robot.l_step == 0:
                if len(full_target_point_map) > 0:
                    ### 定义历史节点（构建带权图）
                    is_exist_oldhistory = False
                    if len(history_nodes) > 0:
                        closest_index = -1
                        min_distance = float('inf')
                        new_x, new_y = start
                        for i, (x, y) in enumerate(history_nodes):
                            distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                            if distance < 25 and distance < min_distance:
                                min_distance = distance
                                closest_index = i
                                is_exist_oldhistory = True

                        if  is_exist_oldhistory == False:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)
                        else:
                            history_count[closest_index] = history_count[closest_index] + 1

                    else:
                        history_nodes.append(start)
                        history_count.append(1)
                        history_state = np.zeros(360)

                    cur_goal_points = start.copy()

                    logging.info(f'=====> Robot state: Step: {robot.l_step}; Angle: {start_o}; Audio Direction: {audio_direction}; Audio Category: {goal_name}')
                    ### 请在这里写出判断LM

                if len(full_target_point_map) > 0:
                    ### f/n 判断VLM
                    if len(history_nodes) > 0:
                        FN_Prompt = form_prompt_for_FN_Step1(goal_name, audio_direction, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                        # if robot.l_step == 0:
                        #     FN_Prompt = form_prompt_for_FN_Step1(goal_name, audio_direction, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                        # else:
                        #     FN_Prompt = form_prompt_for_FN_Step1(goal_name, audio_direction, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                        # logging.info(FN_Prompt)
                    
                        FN_Rel, FN_Decision = cogvlm2.simple_image_chat(User_Prompt=FN_Prompt, 
                                                                        return_string_probabilities="[Yes, No]", img=sem_map)
                        FN_PR = Perception_weight_decision(FN_Rel, FN_Decision)
                        if FN_PR == 'Neither':
                            FN_PR = FN_Rel
                        logging.info(f"FN_PR: {FN_PR}")

                        angle_score = Perception_PR[0] * 2 + FN_PR[0]
                        robot.angle_score = angle_score
                        c_angle = int(start_o % 360)

                        if is_exist_oldhistory == False:
                            # if c_angle >= 45 and c_angle < 315:
                            #     history_state[c_angle-45:c_angle+45] = angle_score
                            # elif c_angle < 45:
                            #     history_state[:c_angle+45] = angle_score
                            #     history_state[360-c_angle-45:] = angle_score

                            # elif c_angle >= 315:
                            #     history_state[c_angle-45:] = angle_score
                            #     history_state[:c_angle+45-360] = angle_score
                            # h_score = history_state.sum()
                            # history_states.append(history_state)
                            tmp_a_score = {0:0, 90:0, 180:0, 270:0}
                            tmp_a_score[c_angle] = angle_score
                            history_score.append(tmp_a_score)
                        else:
                            # if c_angle >= 45 and c_angle < 315:
                            #     history_states[closest_index][c_angle-45:c_angle+45] = angle_score
                            # elif c_angle < 45:
                            #     history_states[closest_index][:c_angle] = angle_score
                            #     history_states[closest_index][360-c_angle:] = angle_score
                            # elif c_angle >= 315:
                            #     history_states[closest_index][c_angle:] = angle_score
                            #     history_states[closest_index][:360-c_angle] = angle_score
                            # h_score = history_states[closest_index].sum() / history_count[closest_index]

                            history_score[closest_index][c_angle] = angle_score
                    
                    # Scores = []

                    history_nodes_copy = history_nodes.copy()
                    history_score_copy = history_score.copy()
                    full_Frontiers_dict_copy = full_Frontiers_dict.copy()

                    ### 决策LM

                    #### 打分：二维矩阵，大小：

                    if FN_PR[0] >= 0.3 or robot.l_step <= 125:
                        
                        if len(pre_goal_points) > 0:
                            Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points, goal_name, start, full_Frontiers_dict_copy)
                        else:
                            Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points, goal_name, start, full_Frontiers_dict_copy)
                        
                        Meta_Score, Meta_Choice = cogvlm2.simple_image_chat(User_Prompt=Meta_Prompt,
                                                    return_string_probabilities="[A, B, C, D]", img=sem_map_frontier)
                        # print(Meta_Choice)

                        # Meta_Score, Meta_Choice = cogvlm2.COT2(User_Prompt1=Single_Prompt, 
                        #                             User_Prompt2=Meta_Prompt,
                                                    # cot_pred1=Single_Choice,
                                                    # return_string_probabilities="[A, B, C, D]", img=sem_map_frontier)
                        Final_PR = Perception_weight_decision4(Meta_Score, Meta_Choice)


                        Choice = Final_PR.index(max(Final_PR))
                    else:
                        # 由于不稳定性，将其替换为分数最高的nodes
                        # Meta_Prompt = form_prompt_for_DecisionVLM_History(pre_goal_points[j], goal_name, start, history_score_copy, history_nodes_copy)
                        # Meta_Score, Meta_Choice = cogvlm2.COT3(User_Prompt1=VLM_Perception_Prompt, 
                        #                             User_Prompt2=FN_Prompt,
                                                    # User_Prompt3=Meta_Prompt,
                                                    # cot_pred1=Perception_Pred,
                                                    # cot_pred2=FN_Decision,
                                                    # return_string_probabilities="[a, b, c, d]", img=full_rgb)
                        # Decisions.append(Meta_Choice)
                        # Final_PR = Perception_weight_decision26(Meta_Score, Meta_Choice)
                        Final_PR = history_score_copy
                        max_value = None
                        Choice = -1

                        # 遍历列表中的每个字典
                        for index, dictionary in enumerate(Final_PR):
                            # 计算字典中的最大值
                            current_max = max(dictionary.values())
                            
                            # 如果这是第一次迭代或者找到了更大的值，则更新最大值和索引
                            if max_value is None or current_max > max_value:
                                max_value = current_max
                                Choice = index

                    logging.info(f"Final_PR: {Final_PR}")
                    # Scores.append(Final_PR)
                    # 初始化最大值和索引
                    
                    if FN_PR[0] >= 0.3 or robot.l_step <= 125:
                        logging.info(f"Decision Choice: frontier_{Choice}")
                        Choice2 = Meta_Score.index(max(Meta_Score))

                        if len(full_Frontiers_dict) == 1:
                            goal_points = [int(x) for x in full_Frontiers_dict['frontier_0'].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                        else:
                            if len(full_Frontiers_dict) == 4:
                                frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                            elif len(full_Frontiers_dict) == 3:
                                frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                            elif len(full_Frontiers_dict) == 2:
                                frontier_keys = ['frontier_0', 'frontier_1']
                            else:
                                frontier_keys = ['frontier_0']
                                
                            
                            invalid_answer = False
                            for i, key in enumerate(frontier_keys):
                                if Choice == i:
                                    if key in full_Frontiers_dict_copy:
                                        goal_points = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                    else:
                                        invalid_answer = True
                                    break
                            if invalid_answer:
                                for i, key in enumerate(frontier_keys):
                                    if Choice2 == i:
                                        try:
                                            goal_points = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                            break
                                        except:
                                            goal_points = [int(x) for x in full_Frontiers_dict_copy[frontier_keys[0]].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                            break
                                

                    else:
                        logging.info(f"Decision Choice: history_{Choice}")
                        if len(history_nodes_copy)==1:
                            goal_points = history_nodes_copy[0]
                        else:
                            for i in range(len(history_nodes_copy)):
                                if Choice == i:
                                    goal_points = history_nodes_copy[i]
                                    break
                else:
                    actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                    goal_points = [int(actions[0]), int(actions[1])]
                
                def calculate_distance(coord1, coord2):
                    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

                # 当前场景值得探索，且智能体并没有到Frontier
                if len(pre_g_points) != 0:
                    if calculate_distance(cur_goal_points, pre_g_points) >= 25 and robot.is_Frontier == True:
                        # print(calculate_distance(cur_goal_points[i], pre_g_points[i]))
                        goal_points = pre_g_points

                # Local_Policy = 1
                # 判断距离，如果两次间隔距离过短就选择随机点进行导航
                if len(pre_goal_points) > 0 and calculate_distance(pre_goal_points, cur_goal_points) <= 2.5:
                    actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                    goal_points = [int(actions[0]), int(actions[1])]
                
                logging.info(f"goal_points: {goal_points}")
                pre_g_points = goal_points.copy()
                logging.info("===== Starting local strategy ===== ")      


            # ractions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1) ##随机采样点
            # goal_points = [int(ractions[0]), int(ractions[1])]
            logging.info(f"goal_points: {goal_points}")
            action = robot.act(goal_points)

            print("VLM_ACTION:",action)
            print("pose_pred:",pose_pred)

            #### 这一段建议放到循环action结束后面
            # if robot.l_step == 0:
            #     Visualize(args, robot.episode_n, robot.l_step, pose_pred, full_map_pred, robot.goal_id, \
            #     visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=[goal_points], audiogoal=batch['pointgoal_with_gps_compass'])
            #     audio_goal = batch['pointgoal_with_gps_compass']
            # else:
            #     Visualize(args, robot.episode_n, robot.l_step, pose_pred, full_map_pred, robot.goal_id, \
            #     visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=[goal_points], audiogoal=audio_goal)   

            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    test_em.memory[:, 0] if ppo_cfg.use_external_memory else None,
                    test_em.masks if ppo_cfg.use_external_memory else None,
                    deterministic=False
                )

                prev_actions.copy_(actions)

            # actions = [a[0].item() for a in actions]

            # print("RL_ACTION:",actions)

            
            outputs = self.envs.step([action]) ### 执行本地策略结束, 开始下一阶段

            if robot.l_step == 0 or robot.l_step == 1:
                Visualize(args, robot.episode_n, robot.l_step, pose_pred, full_map_pred, robot.goal_id, \
                visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=[goal_points], audiogoal=batch['pointgoal_with_gps_compass'])
                audio_goal = batch['pointgoal_with_gps_compass']
            else:
                Visualize(args, robot.episode_n, robot.l_step, pose_pred, full_map_pred, robot.goal_id, \
                visited_vis, full_target_edge_map, Frontiers_dict=None, goal_points=[goal_points], audiogoal=audio_goal)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if config.DISPLAY_RESOLUTION != model_resolution:
                obs_copy = resize_observation(observations, model_resolution)
            else:
                obs_copy = observations
            batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                                'oracle_action_sensor'])
            # print(batch)

            full_map = []
            visited_vis = []
            pose_pred = []
            robot.mapping(batch)
            

            local_map1, _ = torch.max(robot.local_map.unsqueeze(0), 0)
            full_map.append(robot.local_map)
            visited_vis.append(robot.visited_vis)
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = robot.planner_pose_inputs
            r, c = start_y, start_x
            start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
            start = pu.threshold_poses(start, robot.local_map[0, :, :].cpu().numpy().shape)

            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / robot.visited_vis.shape[0],
                    (robot.visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / robot.visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
            pose_pred.append(pos)


            full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map], dim=0)
            full_map_pred, _ = torch.max(full_map2, 0)
            _, full_Frontier_list, full_target_edge_map, full_target_point_map = self.Frontiers(full_map_pred)

            
            if robot.l_step % args.num_local_steps == args.num_local_steps - 1 or robot.l_step == 0:
                agents_seg_list = self.Objects_Extract(args, full_map_pred, args.use_sam)

                pre_goal_points.clear()
                if len(cur_goal_points) > 0:
                    pre_goal_points = cur_goal_points.copy()
                    cur_goal_points.clear()
                
                
                rgb = batch['rgb'][0].cpu().numpy().astype(np.uint8)
                if not os.path.exists(dump_dir):
                    os.makedirs(dump_dir)
                vis_ep_dir = '{}/episodes/eps_{}/Agent1_vis'.format(
                            dump_dir, robot.episode_n)
                if not os.path.exists(vis_ep_dir):
                    os.makedirs(vis_ep_dir)
                fn2 = '{}/episodes/eps_{}/Agent1_vis/IMGstep-{}.png'.format(
                            dump_dir, robot.episode_n,
                            robot.l_step)
                        # print(fn)
                cv2.imwrite(fn2, rgb)  
                if args.yolo == 'yolov9':
                    agent_objs[f"robot_{0}"] = yolo.run(rgb) # 记录一个时间步内每个智能体的目标检测信息
                else:
                    yolo_output = yolo(source=rgb,conf=0.2)
                    yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                    agent_objs[f"robot_{0}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                # logging.info(agent_objs)

                intensity_ratio = np.sum(batch['spectrogram'].cpu().numpy()[0][:, :, 0]) / np.sum(batch['spectrogram'].cpu().numpy()[0][:, :, 1] + 1e-10)  # 添加一个小数以避免除以0的情况
                print("!!!!!intensity_ratio:",intensity_ratio)

                if intensity_ratio > 1.001:
                    audio_direction = 'Left side of the red arrow.'
                    img_direction = 'Left side of image.'
                elif intensity_ratio <= 1.001 and intensity_ratio > 0.999:
                    audio_direction = 'In the Front of the red arrow.'
                    img_direction = 'Middle of image.'
                elif intensity_ratio <= 0.999:
                    audio_direction = 'Right side of the red arrow.'
                    img_direction = 'Right side of image.'


                # 展示用gt goal_name代替goal_name 正常情况应该重新设计提示
                goal_name = object_category[np.argmax(batch['category'].cpu().numpy()[0])]


                if len(full_target_point_map) > 0:
                    full_Frontiers_dict = {}
                    for j in range(len(full_target_point_map)):
                        full_Frontiers_dict['frontier_' + str(j)] = f"<centroid: {full_target_point_map[j][0], full_target_point_map[j][1]}, number: {full_Frontier_list[j]}>"
                    logging.info(f'=====> Origin Frontier: {full_Frontiers_dict}')

                    if intensity_ratio > 1.001:
                        # 朝向：右 声源：左
                        if int(start_o % 360) == 0:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find('(') + 1:value.find(', ')]) >= start[0]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：上 声源：左
                        elif int(start_o % 360) == 90:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find(', ') + 1:value.find(')')]) < start[1]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：左 声源：左
                        elif int(start_o % 360) == 180:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find('(') + 1:value.find(', ')]) < start[0]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：下 声源：左
                        elif int(start_o % 360) == 270:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find(', ') + 1:value.find(')')]) >= start[1]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value


                    elif intensity_ratio <= 1.001 and intensity_ratio > 0.999:
                        #声源：中！！！
                        full_Frontiers_dict_del = full_Frontiers_dict
                    
                    elif intensity_ratio <= 0.999:  
                        # 朝向：右 声源：右
                        if int(start_o % 360) == 0:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find('(') + 1:value.find(', ')]) < start[0]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：上 声源：右
                        elif int(start_o % 360) == 90:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find(', ') + 1:value.find(')')]) >= start[1]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：左 声源：右
                        elif int(start_o % 360) == 180:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find('(') + 1:value.find(', ')]) >= start[0]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value
                        # 朝向：下 声源：右
                        elif int(start_o % 360) == 270:
                            # 过滤出左侧符合条件的键值对
                            filtered_items = {key: value for key, value in full_Frontiers_dict.items()
                                            if int(value[value.find(', ') + 1:value.find(')')]) < start[1]}
                            # 重新命名键
                            full_Frontiers_dict_del = {}
                            for i, (key, value) in enumerate(filtered_items.items()):
                                renamed_key = f'frontier_{i}'
                                full_Frontiers_dict_del[renamed_key] = value

                    logging.info(f'=====> New Frontier: {full_Frontiers_dict_del}')


                    if len(history_nodes) > 0:
                        logging.info(f'=====> history_nodes: {history_nodes}')
                        logging.info(f'=====> history_score: {history_score}')

                    if len(pre_goal_points) > 0:
                        # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                        #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                        if intensity_ratio == 0.0:
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                            robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points)
                        else:
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                            robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict_del, goal_points=[], pre_goal_point=pre_goal_points)
                    else:
                        # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                        #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                        if intensity_ratio == 0.0:
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                            robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict_del, goal_points=[], pre_goal_point=pre_goal_points)
                        else:
                            sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, 0, robot.episode_n, robot.l_step, pose_pred, full_map_pred, 
                            robot.goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict_del, goal_points=[], pre_goal_point=pre_goal_points)
                    # full_rgb = np.hstack((rgb, sem_map)


                    #### 感知LM，给出感知分数，
                    Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM_Step1(goal_name, agent_objs[f'robot_{0}'], img_direction, args.yolo)
                    _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                    return_string_probabilities=None, img=rgb)
                    Perception_Rel, Perception_Pred = cogvlm2.COT2(User_Prompt1=Caption_Prompt, 
                                                            User_Prompt2=VLM_Perception_Prompt,
                                                            cot_pred1=Scene_Information,
                                                            return_string_probabilities="[Yes, No]", img=rgb)
                    Perception_Rel = np.array(Perception_Rel)
                    Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                    logging.info(f"robot-VLM_PerceptionPR: {Perception_PR}")

            

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            # Update external memory
            if ppo_cfg.use_external_memory:
                test_em.insert(test_em_features, not_done_masks)
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.update(batch, dones)
                batch['count_max_category'] = self.count_max_category

                for i in range(len(descriptor_pred_gt)):
                    category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                    # print("category_prediction:",category_prediction)
                    # print("pose:",batch['pose'])
                    location_prediction = batch['location_belief'].cpu().numpy()[i]
                    category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                    location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                    if dones[i]:
                        geodesic_distance = -1
                    else:
                        geodesic_distance = infos[i]['distance_to_goal']
                    pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                    if 'view_point_goals' in observations[i]:
                        pair += (observations[i]['view_point_goals'],)
                    descriptor_pred_gt[i].append(pair)

                    # left_channel = batch['spectrogram'][:, :, 0].cpu().numpy()
                    # right_channel = batch['spectrogram'][:, :, 1].cpu().numpy()
                    # sampling_rate = 44100  # 假设采样率为44.1kHz

                    print("pose: ",batch['pose'])
                    print("location_gt:",location_gt)
                    # print("location_prediction:",location_prediction)

                    # itd = self.calculate_itd(left_channel, right_channel, sampling_rate)
                    # print(f"Interaural Time Difference (ITD): {itd} seconds")

                    # ild, power_all = self.calculate_ild(left_channel, right_channel)
                    # print(f"power_all: {power_all}")
                    # print(f"Interaural Level Difference (ILD): {ild} dB")

                    # spectral_estimation = self.estimate_direction_from_spectrogram(left_channel, right_channel)
                    # print(f"spectral_estimation: {spectral_estimation}")

                    self.count_max_category[category_prediction] += 1
                    batch['count_max_category'][category_prediction] += 1


            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if self.config.RL.PPO.use_belief_predictor:
                        pred = descriptor_pred_gt[i][-1]
                    else:
                        pred = Noneobservations
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in [i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i], pred=pred)
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i], pred=pred)
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # pause envs which runs out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                ### 由此处进入小循环结束
                
                if not_done_masks[i].item() == 0 or robot.l_step > 500:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                        logging.info(f'{metric_uuid}: {episode_stats[metric_uuid]}')
                    logging.info('=========================================')
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
                                                               np.array(current_episodes[i].start_position))
                    episode_stats['audio_duration'] = int(current_episodes[i].duration)
                    episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])
                    if self.config.RL.PPO.use_belief_predictor:
                        episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])
                        episode_stats['descriptor_pred_gt'] = descriptor_pred_gt[i][:-1]
                        descriptor_pred_gt[i] = [descriptor_pred_gt[i][-1]]
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                    if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                        if 'sound' in current_episodes[i].info:
                            sound = current_episodes[i].info['sound']
                        else:
                            sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps
                        )

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        if self.config.RL.PPO.use_belief_predictor:
                            pred = episode_stats['descriptor_pred_gt'][-1]
                        else:
                            pred = None
                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET,
                                                         pred=pred)
                        scene = current_episodes[i].scene_id.split('/')[3]
                        sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        writer.add_image(f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}/"
                                         f"{infos[i]['spl']}",
                                         top_down_map,
                                         dataformats='WHC')
                    # exit()
                    print(1111111111111111111111111111111111)
                    robot.reset()
                    history_nodes.clear()
                    history_score.clear()
                    history_count.clear()
                    history_states.clear()
                    pre_g_points.clear()
                    target_point.clear()

            if not self.config.RL.PPO.use_belief_predictor:
                descriptor_pred_gt = None

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                test_em,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                test_em,
                descriptor_pred_gt
            )

        # dump stats for each episode
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        with open(stats_file, 'w') as fo:
            json.dump({','.join(key): value for key, value in stats_episodes.items()}, fo, cls=NpEncoder)

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ['audio_duration', 'gt_na', 'descriptor_pred_gt', 'view_point_goals']:
                continue
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
            )

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result


def compute_distance_to_pred(pred, sim):
    from habitat.utils.geometry_utils import quaternion_rotate_vector
    import networkx as nx

    current_position = sim.get_agent_state().position
    agent_state = sim.get_agent_state()
    source_position = agent_state.position
    source_rotation = agent_state.rotation

    rounded_pred = np.round(pred)
    direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
    direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)
    pred_goal_location = source_position + direction_vector.astype(np.float32)
    pred_goal_location[1] = source_position[1]

    try:
        if sim.position_encoding(pred_goal_location) not in sim._position_to_index_mapping:
            pred_goal_location = sim.find_nearest_graph_node(pred_goal_location)
        distance_to_target = sim.geodesic_distance(current_position, [pred_goal_location])
    except nx.exception.NetworkXNoPath:
        distance_to_target = -1
    return distance_to_target
