# diffusion_policy/dataset/maniskill_image_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any
from pathlib import Path
import os, glob, copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


# --------------------------
# small helpers
# --------------------------
def _sorted_numeric_suffix(names: List[str], prefix: str) -> List[str]:
    items = [n for n in names if n.startswith(prefix)]
    def _idx(n: str) -> int:
        try:
            return int(n.split("_")[-1])
        except Exception:
            return 0
    return sorted(items, key=_idx)

def _to_chw01(img: np.ndarray) -> np.ndarray:
    """Convert image to CHW in [0,1]. Accepts HWC or CHW, uint8 or float."""
    if img.ndim == 3 and img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    if img.max() > 1.5:  # handles [0,255]
        img = img / 255.0
    return img

def _scalarize(v: Any) -> float:
    try:
        v = v[()] if isinstance(v, h5py.Dataset) else v
    except Exception:
        pass
    v = np.asarray(v).squeeze()
    try:
        return float(v)
    except Exception:
        return float(bool(v))


@dataclass
class _EpisodeIndex:
    start: int  # inclusive index in the global arrays
    end: int    # exclusive
    # Convenience for debugging / provenance (not required at runtime)
    env_name: Optional[str] = None
    episode_name: Optional[str] = None


class ManiSkillLDPImageDataset(BaseImageDataset):
    """
    Adapts ManiSkill-style HDF5 demos to the LDP repo's BaseImageDataset interface.

    HDF5 expected structure:
        env_<name>/episode_<k>/record_timestep_<t>/{action, image, [wrist_image], state, [success-like keys], ...}

    Output sample (dict of torch.Tensors):
        {
          "obs": {
              "image":  (T, C[, H, W]),
              "state":  (T, D_state)
          },
          "action":    (T, D_action)
        }
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        # keys
        image_key: str = "image",
        wrist_image_key: str = "wrist_image",
        use_wrist_camera: bool = False,
        state_key: str = "state",
        action_key: str = "action",
        # split
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        # filtering
        success_key: Optional[str] = None,
        success_filter: str = "auto",  # "auto" | "none" | "last" | "any" | "exclude_true"
        exclude_current_task_timesteps: bool = False,
        # debug
        env_filter: Optional[str] = None,
        max_envs: Optional[int] = None,
        max_episodes_per_env: Optional[int] = None,
        # (optional) limit how many episodes to parse total (after filtering)
        max_total_episodes: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)

        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.use_wrist_camera = use_wrist_camera
        self.state_key = state_key
        self.action_key = action_key

        self.requested_success_key = success_key
        self.success_filter = str(success_filter).lower().strip()
        assert self.success_filter in {"auto", "none", "last", "any", "exclude_true"}

        self.exclude_current_task_timesteps = exclude_current_task_timesteps
        self.env_filter = env_filter
        self.max_envs = max_envs
        self.max_episodes_per_env = max_episodes_per_env
        self.max_total_episodes = max_total_episodes

        # ---- load raw arrays (concat over episodes on time axis) ----
        data, episode_ends = self._load_all()
        # Store raw arrays (numpy) for sampler / normalizer
        self.images: np.ndarray = data["image"]   # (TotalT, C, H, W)
        self.states: np.ndarray = data["state"]   # (TotalT, D_state)
        self.actions: np.ndarray = data["action"] # (TotalT, D_action)
        self.episode_ends: np.ndarray = episode_ends  # shape (num_episodes,)

        # ---- split train/val by episodes ----
        rng = np.random.RandomState(seed)
        n_eps = len(self.episode_ends)
        perm = np.arange(n_eps)
        rng.shuffle(perm)
        n_val = int(round(val_ratio * n_eps))
        val_ids = set(perm[:n_val].tolist())
        train_ids = set(perm[n_val:].tolist())
        if max_train_episodes is not None:
            # softly downsample train episodes
            train_ids = set(list(train_ids)[:max_train_episodes])

        self.train_episode_mask = np.array([i in train_ids for i in range(n_eps)], dtype=bool)
        self.val_episode_mask = ~self.train_episode_mask

        # Build "samplers" (actually just lists of (start,end) indices for windows)
        self._train_windows = self._build_windows(self.train_episode_mask)
        self._val_windows = self._build_windows(self.val_episode_mask)

        # Active sampler defaults to train
        self._use_val = False

    # --------------- BaseDataset API ---------------

    def get_validation_dataset(self) -> "ManiSkillLDPImageDataset":
        val_set = copy.copy(self)
        val_set._use_val = True
        return val_set

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        """
        Only normalize numeric channels (state/action). Images are already in [0,1].
        We'll concatenate state along time, leave image out.
        """
        # (TotalT, D_state) and (TotalT, D_action)
        data = {
            "obs": self.states,          # T x D_o
            "action": self.actions,      # T x D_a
        }
        normalizer = LinearNormalizer()
        # last_n_dims=1 => per-feature stats over trailing dim
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.actions)

    def __len__(self) -> int:
        return len(self._val_windows if self._use_val else self._train_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        windows = self._val_windows if self._use_val else self._train_windows
        s, e = windows[idx]  # window [s, e) in global time
        # Pad if needed to exactly match horizon + pad_before + pad_after
        T_needed = self.horizon + self.pad_before + self.pad_after
        arr_slice = slice(s, e)

        img = self.images[arr_slice]   # (L, C, H, W)
        st  = self.states[arr_slice]   # (L, D_s)
        act = self.actions[arr_slice]  # (L, D_a)

        L = img.shape[0]
        if L < T_needed:
            # pad on both sides to center the original within the padded window
            to_pad = T_needed - L
            # Prefer symmetric padding; if odd, put extra on the right
            left = to_pad // 2
            right = to_pad - left
            img = np.pad(img, ((left, right), (0, 0), (0, 0), (0, 0)), mode="edge")
            st  = np.pad(st,  ((left, right), (0, 0)), mode="edge")
            act = np.pad(act, ((left, right), (0, 0)), mode="edge")
        elif L > T_needed:
            # trim if overshoot (shouldn't happen with our window builder)
            img = img[:T_needed]
            st  = st[:T_needed]
            act = act[:T_needed]

        # Return tensors with shapes expected by LDP
        return {
            "obs": {
                "image": torch.from_numpy(img.copy()),  # (T, C, H, W)
                "state": torch.from_numpy(st.copy()),   # (T, D_s)
            },
            "action": torch.from_numpy(act.copy())       # (T, D_a)
        }

    # --------------- internals ---------------

    def _detect_success_key(self, ts_grp: h5py.Group) -> Optional[str]:
        if self.requested_success_key is not None and self.requested_success_key in ts_grp:
            return self.requested_success_key
        candidates = ['demonstration']
        # breakpoint()
        for k in candidates:
            if k in ts_grp:
                return k
        return None

    def _episode_success(self, episode_grp: h5py.Group, timestep_names: List[str], success_key: str) -> bool:
        flags: List[float] = []
        for ts in timestep_names:
            g = episode_grp[ts]
            if success_key in g:
                flags.append(_scalarize(g[success_key]))
        if len(flags) == 0:
            return self.success_filter == "none"

        policy = self.success_filter
        if policy == "auto":
            policy = "any" if success_key == "current_task_demonstration" else "last"

        if policy == "any":
            return any(f > 0.0 for f in flags)
        if policy == "last":
            return flags[-1] > 0.0
        if policy == "none":
            return True
        if policy == "exclude_true":
            return not any(f > 0.0 for f in flags)
        return flags[-1] > 0.0

    def _load_all(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load all episodes, concatenate along time. Return (data_dict, episode_ends)."""
        path = self.dataset_path
        if os.path.isdir(path):
            file_list = sorted(glob.glob(os.path.join(path, "*.h5")))
            if len(file_list) == 0:
                raise FileNotFoundError(f"No .h5 files found in directory: {path}")
        else:
            file_list = [path]

        all_imgs, all_states, all_actions = [], [], []
        episode_ends: List[int] = []
        total_steps = 0
        total_eps = 0

        for fp in file_list:
            with h5py.File(fp, "r") as f:
                # env groups
                env_names = [k for k in f.keys() if k.startswith("env_")]
                if not env_names:
                    env_names = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                if self.env_filter:
                    env_names = [e for e in env_names if self.env_filter in e]
                if self.max_envs is not None:
                    env_names = env_names[: self.max_envs]
                for env_name in env_names:
                    env_grp = f[env_name]
                    episodes = _sorted_numeric_suffix(list(env_grp.keys()), "episode_")
                    if self.max_episodes_per_env is not None:
                        episodes = episodes[: self.max_episodes_per_env]
                    # FIXME episode only load two episodes(for debugging)
                    episodes = episodes[:2]
                    for ep_name in episodes:
                        ep_grp = env_grp[ep_name]
                        ts_names = _sorted_numeric_suffix(list(ep_grp.keys()), "record_timestep_")
                        if not ts_names:
                            continue

                        # success handling
                        first_ts = ep_grp[ts_names[0]]
                        success_key = self._detect_success_key(first_ts)
                        use_filter = (self.success_filter != "none") and (success_key is not None)
                        if use_filter and not self._episode_success(ep_grp, ts_names, success_key):
                            continue

                        # collect timestep arrays
                        ep_imgs, ep_states, ep_actions = [], [], []
                        for ts in ts_names:
                            g = ep_grp[ts]

                            # optional per-step exclusion (current_task_demonstration)
                            if (
                                self.exclude_current_task_timesteps
                                and success_key == "current_task_demonstration"
                                and (success_key in g)
                            ):
                                flag = _scalarize(g[success_key]) > 0.0
                                if flag:
                                    continue

                            if self.action_key not in g or self.state_key not in g or self.image_key not in g:
                                ep_imgs, ep_states, ep_actions = [], [], []
                                break

                            a = np.asarray(g[self.action_key][...], dtype=np.float32)
                            if a.ndim == 0:
                                a = np.array([a], dtype=np.float32)
                            s = np.asarray(g[self.state_key][...]).squeeze().astype(np.float32)
                            img = _to_chw01(np.asarray(g[self.image_key][...]))

                            if self.use_wrist_camera and (self.wrist_image_key in g):
                                w = _to_chw01(np.asarray(g[self.wrist_image_key][...]))
                                try:
                                    img = np.concatenate([img, w], axis=0)  # (6,H,W)
                                except Exception:
                                    pass

                            ep_actions.append(a)
                            ep_states.append(s)
                            ep_imgs.append(img)

                        if len(ep_actions) == 0:
                            continue

                        try:
                            ep_imgs = np.stack(ep_imgs, axis=0)      # (L,C,H,W)
                            ep_states = np.stack(ep_states, axis=0)  # (L,D_s)
                            ep_actions = np.stack(ep_actions, axis=0) # (L,D_a)
                        except Exception:
                            continue

                        if not (len(ep_imgs) == len(ep_states) == len(ep_actions)):
                            continue

                        all_imgs.append(ep_imgs)
                        all_states.append(ep_states)
                        all_actions.append(ep_actions)

                        total_steps += len(ep_actions)
                        episode_ends.append(total_steps)
                        total_eps += 1

                        if self.max_total_episodes is not None and total_eps >= self.max_total_episodes:
                            break
                    if self.max_total_episodes is not None and total_eps >= self.max_total_episodes:
                        break

        if total_eps == 0:
            raise ValueError("No usable trajectories found in the provided ManiSkill files/filters.")

        data = {
            "image": np.concatenate(all_imgs, axis=0).astype(np.float32),
            "state": np.concatenate(all_states, axis=0).astype(np.float32),
            "action": np.concatenate(all_actions, axis=0).astype(np.float32),
        }
        return data, np.asarray(episode_ends, dtype=np.int64)

    def _build_windows(self, episode_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Build list of (start,end) indices (global time) for each valid window with
        length = horizon + pad_before + pad_after.
        We enforce start/end to stay inside each episode to avoid crossing boundaries.
        """
        T_needed = self.horizon + self.pad_before + self.pad_after
        windows: List[Tuple[int, int]] = []
        prev_end = 0
        for epi, end in enumerate(self.episode_ends):
            if not episode_mask[epi]:
                prev_end = end
                continue
            epi_start = prev_end
            epi_end = end
            L = epi_end - epi_start
            if L <= 0:
                prev_end = end
                continue
            # slide a window center so that total slice stays within episode
            # We define center positions from 0..L-1; build slices [c - left, c + right + 1]
            for c in range(L):
                left = self.pad_before + (self.horizon // 2)
                right = T_needed - left
                s = epi_start + max(0, c - left)
                e = s + T_needed
                if e > epi_end:
                    # shift left to stay in-episode
                    shift = e - epi_end
                    s = max(epi_start, s - shift)
                    e = s + T_needed
                # clip (final guard)
                s = max(epi_start, s)
                e = min(epi_end, e)
                if (e - s) <= 0:
                    continue
                windows.append((s, e))
            prev_end = end
        return windows
