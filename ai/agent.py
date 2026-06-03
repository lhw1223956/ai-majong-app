import os
from pathlib import Path

import numpy as np
import streamlit as st

from core.calculator import get_waiting_tiles
from core.config import CODE_TO_IDX, IDX_TO_CODE, TILE_INFO

NUM_TILE_TYPES = 34
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_LOAD_ERRORS = {}


def _resolve_model_path(model_path):
    path = Path(model_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_model_load_error(model_path):
    resolved = str(_resolve_model_path(model_path))
    return MODEL_LOAD_ERRORS.get(resolved)


def _build_rl_observation_8x34(hand_codes, exp_codes, discard_pool=None):
    obs = np.zeros((8, NUM_TILE_TYPES), dtype=np.float32)

    seen = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    for c in hand_codes:
        if c in CODE_TO_IDX:
            seen[CODE_TO_IDX[c]] += 1
    for c in exp_codes:
        if c in CODE_TO_IDX:
            seen[CODE_TO_IDX[c]] += 1
    if discard_pool:
        for c in discard_pool:
            if c in CODE_TO_IDX:
                seen[CODE_TO_IDX[c]] += 1

    seen = np.minimum(seen, 4)
    unseen = np.maximum(0, 4 - seen)

    for k in range(4):
        obs[k] = (seen > k).astype(np.float32)
        obs[4 + k] = (unseen > k).astype(np.float32)
    return obs


def _build_rl_observation_12x34(hand_codes, exp_codes, discard_pool=None):
    obs = np.zeros((12, NUM_TILE_TYPES), dtype=np.float32)

    own_visible = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    for c in hand_codes:
        if c in CODE_TO_IDX:
            own_visible[CODE_TO_IDX[c]] += 1
    for c in exp_codes:
        if c in CODE_TO_IDX:
            own_visible[CODE_TO_IDX[c]] += 1

    public_visible = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    for c in exp_codes:
        if c in CODE_TO_IDX:
            public_visible[CODE_TO_IDX[c]] += 1
    if discard_pool:
        for c in discard_pool:
            if c in CODE_TO_IDX:
                public_visible[CODE_TO_IDX[c]] += 1

    own_visible = np.minimum(own_visible, 4)
    public_visible = np.minimum(public_visible, 4)
    seen = np.minimum(own_visible + public_visible, 4)
    unseen = np.maximum(0, 4 - seen)

    for k in range(4):
        obs[k] = (own_visible > k).astype(np.float32)
        obs[4 + k] = (public_visible > k).astype(np.float32)
        obs[8 + k] = (unseen > k).astype(np.float32)
    return obs


def _build_observation_for_model(hand_codes, exp_codes, discard_pool, ppo_model):
    obs_shape = getattr(getattr(ppo_model, "observation_space", None), "shape", None)
    obs_planes = int(obs_shape[0]) if obs_shape else 12
    if obs_planes == 8:
        return _build_rl_observation_8x34(hand_codes, exp_codes, discard_pool)
    return _build_rl_observation_12x34(hand_codes, exp_codes, discard_pool)


def _load_maskable_model(model_path, policy_kwargs):
    from sb3_contrib import MaskablePPO  # type: ignore

    return MaskablePPO.load(
        model_path,
        device="cpu",
        custom_objects={"policy_kwargs": policy_kwargs},
    )


def _load_ppo_model(model_path, policy_kwargs):
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.buffers import RolloutBuffer  # type: ignore
    from stable_baselines3.common.policies import ActorCriticPolicy  # type: ignore

    return PPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "policy_kwargs": policy_kwargs,
            "policy_class": ActorCriticPolicy,
            "rollout_buffer_class": RolloutBuffer,
            "rollout_buffer_kwargs": {},
        },
    )


def _build_current_cnn_policy_kwargs():
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class MahjongCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.Space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            in_channels = observation_space.shape[0]

            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
            )

            with torch.no_grad():
                sample = torch.zeros(1, in_channels, observation_space.shape[1])
                n_out = self.cnn(sample).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_out, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))

    return dict(
        features_extractor_class=MahjongCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )


def _load_old_ppo_model(model_path):
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class OldMahjongCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.Space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            in_channels = observation_space.shape[0]

            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
            )

            with torch.no_grad():
                sample = torch.zeros(1, in_channels, observation_space.shape[1])
                n_out = self.cnn(sample).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_out, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))

    policy_kwargs = dict(
        features_extractor_class=OldMahjongCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    return PPO.load(
        model_path,
        device="cpu",
        custom_objects={"policy_kwargs": policy_kwargs},
    )


@st.cache_resource
def load_ppo_agent(model_path="models/cnn/mahjong_cnn_agent_v1.zip", loader_version="2026-05-05-dual-cnn-v1"):
    """載入 RL 模型，優先支援目前專案使用的 MaskablePPO CNN。"""
    resolved_model_path = _resolve_model_path(model_path)
    resolved_model_path_str = str(resolved_model_path)
    try:
        MODEL_LOAD_ERRORS.pop(resolved_model_path_str, None)

        if not resolved_model_path.exists():
            MODEL_LOAD_ERRORS[resolved_model_path_str] = f"找不到模型檔案：{resolved_model_path_str}"
            return None

        policy_kwargs = _build_current_cnn_policy_kwargs()

        maskable_error = None
        try:
            return _load_maskable_model(resolved_model_path_str, policy_kwargs)
        except Exception as exc:
            maskable_error = exc

        try:
            return _load_ppo_model(resolved_model_path_str, policy_kwargs)
        except Exception as exc:
            ppo_error = exc

        try:
            return _load_old_ppo_model(resolved_model_path_str)
        except Exception as exc:
            err = (
                f"模型載入失敗：{resolved_model_path_str}\n"
                f"MaskablePPO: {maskable_error}\n"
                f"PPO: {ppo_error}\n"
                f"LegacyPPO: {exc}"
            )
            MODEL_LOAD_ERRORS[resolved_model_path_str] = err
            print(err)
            return None
    except Exception as e:
        MODEL_LOAD_ERRORS[resolved_model_path_str] = f"初始化載入器失敗：{e}"
        print(f"Error preparing RL model loader: {e}")
        return None


def get_rl_agent_recommendation(hand_codes, exp_codes, discard_pool=None, ppo_model=None):
    """依模型實際 observation shape 產生 8x34 或 12x34 棄牌建議。"""
    if ppo_model is None:
        return None, "模型未載入", []

    obs = _build_observation_for_model(hand_codes, exp_codes, discard_pool, ppo_model)

    try:
        import torch

        device = next(ppo_model.policy.parameters()).device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            distribution = ppo_model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs[0].cpu().numpy()

        hand_tile_indices = {
            CODE_TO_IDX[c]
            for c in hand_codes
            if c in CODE_TO_IDX
        }
        if not hand_tile_indices:
            return None, "目前沒有可分析的手牌", []

        raw_discards = []
        sum_prob = 0.0
        for idx in hand_tile_indices:
            prob = float(action_probs[idx])
            raw_discards.append((idx, prob))
            sum_prob += prob

        if sum_prob > 0:
            norm_discards = [(idx, prob / sum_prob) for idx, prob in raw_discards]
        else:
            uniform = 1.0 / len(raw_discards)
            norm_discards = [(idx, uniform) for idx, _ in raw_discards]

        norm_discards.sort(key=lambda x: -x[1])

        top_candidates = []
        candidate_conf = {}
        for idx, prob in norm_discards:
            candidate = IDX_TO_CODE.get(idx)
            if candidate:
                top_candidates.append(candidate)
                candidate_conf[candidate] = prob
            if len(top_candidates) >= 5:
                break

        if top_candidates:
            try:
                from algorithms.flat_mc import get_best_discard_with_flatmc  # type: ignore
            except Exception as e:
                get_best_discard_with_flatmc = None
                flatmc_err = e
            else:
                flatmc_err = None

            if get_best_discard_with_flatmc is None:
                all_results = [{"tile": tc, "win_rate": 0.0} for tc in top_candidates]
            else:
                _, _, all_results = get_best_discard_with_flatmc(
                    hand_codes, top_candidates, discard_pool, TILE_INFO, exp_codes=exp_codes
                )

            combined_results = []
            for res in all_results:
                tile = res["tile"]
                win_rate = res["win_rate"]
                cnn_prob = candidate_conf.get(tile, 0.0)
                score = (cnn_prob * 0.4) + (win_rate * 0.6)
                combined_results.append(
                    {
                        "tile": tile,
                        "win_rate": score,
                        "raw_win_rate": win_rate,
                        "cnn_prob": cnn_prob,
                        "is_tenpai": False,
                    }
                )

            combined_results.sort(key=lambda x: -x["win_rate"])
            best_candidate = combined_results[0]["tile"]
            final_score = combined_results[0]["win_rate"]

            hand_only_codes = [c for c in hand_codes if c in TILE_INFO and TILE_INFO[c]["type"] != "h"]
            non_flower_exp = [c for c in exp_codes if c in CODE_TO_IDX]
            melds_count = len(non_flower_exp) // 3
            n = 5 - melds_count
            wait_phase_size = 3 * n + 1
            pool_note = f"，含 {len(discard_pool)} 張桌面資訊" if discard_pool else ""

            for item in combined_results:
                rem = list(hand_only_codes)
                if item["tile"] in rem:
                    rem.remove(item["tile"])
                if len(rem) == wait_phase_size and get_waiting_tiles(rem):
                    item["is_tenpai"] = True

            msg = f"AI 綜合評估建議，分數 {final_score * 100:.1f}%{pool_note}"

            best_item = combined_results[0]
            if best_item.get("is_tenpai"):
                rem = list(hand_only_codes)
                rem.remove(best_candidate)
                after_waiting = get_waiting_tiles(rem)
                if after_waiting:
                    wait_names = [
                        f"{TILE_INFO[t]['icon']}{TILE_INFO[t]['name']}"
                        for t in after_waiting[:4]
                        if t in TILE_INFO
                    ]
                    wait_str = "、".join(wait_names) + ("..." if len(after_waiting) > 4 else "")
                    msg += f"\n打出後進入聽牌，可聽：{wait_str}"

            return best_candidate, msg, combined_results

        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)
        if 0 <= action <= 33:
            target_code = IDX_TO_CODE.get(action)
            if target_code and target_code in hand_codes:
                note = "以 PPO 直接輸出建議"
                if "flatmc_err" in locals() and flatmc_err is not None:
                    note = f"{note}\nFlatMC 錯誤：{flatmc_err}"
                return target_code, note, []
        action_names = {39: "胡", 34: "吃", 35: "碰", 36: "明槓", 40: "過"}
        return None, f"模型輸出非棄牌動作：{action_names.get(action, str(action))}", []

    except Exception as e:
        try:
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = int(action)
            if 0 <= action <= 33:
                target_code = IDX_TO_CODE.get(action)
                if target_code and target_code in hand_codes:
                    return target_code, "回退為 PPO 直接建議", []
        except Exception:
            pass
        return None, f"RL 建議失敗 ({e})", []


def get_structured8x34_ev_flatmc_recommendation(hand_codes, exp_codes, discard_pool=None, ppo_model=None):
    """Structured 8x34 policy + EV top-3 candidate source + FlatMC rerank."""
    if ppo_model is None:
        return None, "模型未載入", []

    obs_shape = getattr(getattr(ppo_model, "observation_space", None), "shape", None)
    if not obs_shape or int(obs_shape[0]) != 8:
        return None, f"模型 observation shape 不是 8x34：{obs_shape}", []

    hand_only_codes = [
        c for c in (hand_codes or [])
        if c in TILE_INFO and TILE_INFO[c]["type"] != "h" and c in CODE_TO_IDX
    ]
    legal_discards = sorted(set(hand_only_codes), key=lambda code: CODE_TO_IDX[code])
    if not legal_discards:
        return None, "手牌沒有可丟棄的有效牌", []

    discard_pool = discard_pool or []
    visible_tiles = [
        c for c in (exp_codes or []) + discard_pool
        if c in TILE_INFO and TILE_INFO[c]["type"] != "h"
    ]
    fixed_melds = len([c for c in (exp_codes or []) if c in CODE_TO_IDX]) // 3

    try:
        import torch
        from algorithms.flat_mc import flat_mc_evaluate_details  # type: ignore
        from algorithms.lyl_progress_ev_judgement import rank_discards_by_progress_ev  # type: ignore

        obs = _build_rl_observation_8x34(hand_codes, exp_codes, discard_pool)
        device = next(ppo_model.policy.parameters()).device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            distribution = ppo_model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs[0].cpu().numpy()

        raw_policy = {code: float(action_probs[CODE_TO_IDX[code]]) for code in legal_discards}
        policy_total = sum(raw_policy.values())
        if policy_total > 0:
            policy_conf = {code: value / policy_total for code, value in raw_policy.items()}
        else:
            uniform = 1.0 / len(legal_discards)
            policy_conf = {code: uniform for code in legal_discards}

        ev_report = rank_discards_by_progress_ev(
            hand_only_codes,
            visible_tiles=visible_tiles,
            legal_discards=legal_discards,
            fixed_melds=fixed_melds,
        )
        ev_rows = ev_report.get("results") or []
        ev_by_tile = {row["tile"]: row for row in ev_rows}
        ev_top_candidates = [row["tile"] for row in ev_rows[:3] if row.get("tile") in legal_discards]
        if not ev_top_candidates:
            ev_top_candidates = sorted(legal_discards, key=lambda code: -policy_conf.get(code, 0.0))[:3]

        combined_results = []
        for candidate in ev_top_candidates:
            flat = flat_mc_evaluate_details(
                hand_only_codes,
                candidate,
                discard_pool,
                TILE_INFO,
                fixed_melds,
                exp_codes,
                num_simulations=40,
                max_depth=20,
            )
            ev_row = ev_by_tile.get(candidate, {})
            policy_score = float(policy_conf.get(candidate, 0.0))
            ev_score = float(ev_row.get("ev_normalized", 0.0))
            flatmc_score = float(flat.get("mc_score", flat.get("win_rate", 0.0)))
            final_score = (policy_score * 0.25) + (ev_score * 0.35) + (flatmc_score * 0.40)
            combined_results.append(
                {
                    "tile": candidate,
                    "score": final_score,
                    "win_rate": final_score,
                    "cnn_prob": policy_score,
                    "ev_normalized": ev_score,
                    "ev_rank": ev_row.get("progress_ev_rank"),
                    "ev": ev_row.get("ev"),
                    "progress_distance": ev_row.get("progress_distance"),
                    "progress_score": ev_row.get("progress_score"),
                    "flatmc_score": flatmc_score,
                    "raw_win_rate": float(flat.get("win_rate", 0.0)),
                    "tenpai_rate": float(flat.get("tenpai_rate", 0.0)),
                    "avg_deficiency_improvement": float(flat.get("avg_deficiency_improvement", 0.0)),
                    "candidate_source": "ev_top3",
                    "is_tenpai": False,
                }
            )

        combined_results.sort(key=lambda row: -float(row["score"]))
        if not combined_results:
            return None, "EV + FlatMC 沒有產生候選牌", []

        non_flower_exp = [c for c in (exp_codes or []) if c in CODE_TO_IDX]
        melds_count = len(non_flower_exp) // 3
        wait_phase_size = 3 * (5 - melds_count) + 1
        for item in combined_results:
            rem = list(hand_only_codes)
            if item["tile"] in rem:
                rem.remove(item["tile"])
            if len(rem) == wait_phase_size and get_waiting_tiles(rem):
                item["is_tenpai"] = True

        best_candidate = combined_results[0]["tile"]
        final_score = float(combined_results[0]["score"])
        pool_note = f"，含 {len(discard_pool)} 張桌面資訊" if discard_pool else ""
        msg = f"Structured 8x34 + EV top-3 + FlatMC 綜合分數 {final_score * 100:.1f}%{pool_note}"

        best_item = combined_results[0]
        if best_item.get("is_tenpai"):
            rem = list(hand_only_codes)
            rem.remove(best_candidate)
            after_waiting = get_waiting_tiles(rem)
            if after_waiting:
                wait_names = [
                    f"{TILE_INFO[t]['icon']}{TILE_INFO[t]['name']}"
                    for t in after_waiting[:4]
                    if t in TILE_INFO
                ]
                wait_str = "、".join(wait_names) + ("..." if len(after_waiting) > 4 else "")
                msg += f"\n打出後可進入聽牌，等待：{wait_str}"

        return best_candidate, msg, combined_results

    except Exception as e:
        return None, f"Structured 8x34 + EV + FlatMC 推薦失敗：{e}", []
