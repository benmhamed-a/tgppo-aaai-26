import torch as T
import logging
from typing import List, Tuple, Iterator, Optional, Dict, Any
import numpy as np


class Memory:
    """Tree-Gate Policy Gradient Branching Memory with import/export APIs.

    Notes
    -----
    - Supports variable-length candidate sets per transition.
    - `export_dict()` outputs plain Python/NumPy objects (CPU) safe for pickling.
    - `import_dict()` reconstructs tensors on this Memory's device.
    - Advantages/returns are optional. If not filled, they are omitted in exports.
    """

    def __init__(self, batch_size, state_dims, device, logger=None):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.batch_size = batch_size
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.cands_states: List[T.Tensor] = []   # (n_cands, var_dim)
        self.mip_states: List[T.Tensor] = []     # (mip_dim,)
        self.node_states: List[T.Tensor] = []    # (node_dim,)
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        # Optional – filled when GAE is computed
        self.advantages: List[float] = []
        self.returns: List[float] = []

        self.var_dim = state_dims["var_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.node_dim = state_dims["node_dim"]

    # ------------------------------------------------------------------
    # Core buffer ops
    # ------------------------------------------------------------------
    def clear(self):
        self.cands_states.clear()
        self.mip_states.clear()
        self.node_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
        self.logger.info("Memory cleared successfully.")

    def _validate_inputs(self, cands_state, mip_state, node_state, action, reward, done, log_prob, value):
        if not isinstance(cands_state, T.Tensor):
            raise TypeError("cands_state must be a torch.Tensor")
        if cands_state.dim() != 2:
            raise ValueError("cands_state must be a 2D tensor (num_candidates x var_dim)")
        if cands_state.size(1) != self.var_dim:
            raise ValueError(f"cands_state must have {self.var_dim} columns, got {cands_state.size(1)}")

        if not isinstance(mip_state, T.Tensor) or mip_state.dim() != 1 or mip_state.size(0) != self.mip_dim:
            raise ValueError("mip_state must be a 1D tensor of correct length")
        if not isinstance(node_state, T.Tensor) or node_state.dim() != 1 or node_state.size(0) != self.node_dim:
            raise ValueError("node_state must be a 1D tensor of correct length")

        if not isinstance(action, int) or action < 0 or action >= cands_state.size(0):
            raise ValueError("Invalid action index")

        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            raise ValueError("reward must be finite number")
        if not isinstance(done, (bool, np.bool_, np.bool8)):
            raise TypeError("done must be a boolean")
        if not isinstance(log_prob, (int, float)) or np.isnan(log_prob) or np.isinf(log_prob):
            raise ValueError("log_prob must be finite number")
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            raise ValueError("value must be finite number")

    def store(self, cands_state: T.Tensor, mip_state: T.Tensor, node_state: T.Tensor,
              action: int, reward: float, done: bool, log_prob: float, value: float):
        self._validate_inputs(cands_state, mip_state, node_state, action, reward, done, log_prob, value)

        cands_state = cands_state.detach().to(self.device)
        mip_state = mip_state.detach().to(self.device)
        node_state = node_state.detach().to(self.device)

        self.cands_states.append(cands_state)
        self.mip_states.append(mip_state)
        self.node_states.append(node_state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.logger.debug(f"Stored transition #{len(self.cands_states)}")

    def set_advantages_returns(self, advantages: List[float], returns: List[float]):
        if len(advantages) != len(self.cands_states) or len(returns) != len(self.cands_states):
            raise ValueError("advantages/returns must match the number of transitions")
        if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
            raise ValueError("advantages contain NaNs/Infs")
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            raise ValueError("returns contain NaNs/Infs")
        self.advantages = [float(a) for a in advantages]
        self.returns = [float(r) for r in returns]

    def __len__(self) -> int:
        return len(self.cands_states)

    def is_empty(self) -> bool:
        return len(self.cands_states) == 0

    # ------------------------------------------------------------------
    # Batching (kept as in your version)
    # ------------------------------------------------------------------
    def get_batch_generator(self, batch_size: Optional[int] = None) -> Iterator[Tuple[T.Tensor, ...]]:
        if self.is_empty():
            raise RuntimeError("Memory is empty. Cannot generate batches.")

        batch_size = batch_size or self.batch_size
        n_states = len(self.cands_states)
        indices = np.arange(n_states)
        np.random.shuffle(indices)

        for start_idx in range(0, n_states, batch_size):
            end_idx = min(start_idx + batch_size, n_states)
            batch_indices = indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue

            batch_cands_states = [self.cands_states[i] for i in batch_indices]
            batch_mip_states = [self.mip_states[i] for i in batch_indices]
            batch_node_states = [self.node_states[i] for i in batch_indices]
            batch_actions = [self.actions[i] for i in batch_indices]
            batch_rewards = [self.rewards[i] for i in batch_indices]
            batch_dones = [self.dones[i] for i in batch_indices]
            batch_log_probs = [self.log_probs[i] for i in batch_indices]
            batch_values = [self.values[i] for i in batch_indices]

            max_candidates = max(cs.size(0) for cs in batch_cands_states)
            padded_cands_states = []
            cands_masks = []
            for cs in batch_cands_states:
                n_cands = cs.size(0)
                if n_cands < max_candidates:
                    padding = T.zeros(max_candidates - n_cands, self.var_dim, device=self.device)
                    padded_cs = T.cat([cs, padding], dim=0)
                    mask = T.cat([
                        T.zeros(n_cands, dtype=T.bool, device=self.device),
                        T.ones(max_candidates - n_cands, dtype=T.bool, device=self.device),
                    ], dim=0)
                else:
                    padded_cs = cs
                    mask = T.zeros(n_cands, dtype=T.bool, device=self.device)
                padded_cands_states.append(padded_cs)
                cands_masks.append(mask)

            batch_cands_states_tensor = T.stack(padded_cands_states)
            batch_cands_masks_tensor = T.stack(cands_masks)
            batch_mip_states_tensor = T.stack(batch_mip_states)
            batch_node_states_tensor = T.stack(batch_node_states)
            batch_actions_tensor = T.tensor(batch_actions, dtype=T.long, device=self.device)
            batch_rewards_tensor = T.tensor(batch_rewards, dtype=T.float32, device=self.device)
            batch_dones_tensor = T.tensor(batch_dones, dtype=T.bool, device=self.device)
            batch_log_probs_tensor = T.tensor(batch_log_probs, dtype=T.float32, device=self.device)
            batch_values_tensor = T.tensor(batch_values, dtype=T.float32, device=self.device)

            yield (
                batch_cands_states_tensor,
                batch_cands_masks_tensor,
                batch_mip_states_tensor,
                batch_node_states_tensor,
                batch_actions_tensor,
                batch_rewards_tensor,
                batch_dones_tensor,
                batch_log_probs_tensor,
                batch_values_tensor,
                batch_indices,
            )

    def batch(self, batch_size: Optional[int] = None) -> List[Tuple[T.Tensor, ...]]:
        return list(self.get_batch_generator(batch_size))

    # ------------------------------------------------------------------
    # Export / Import APIs
    # ------------------------------------------------------------------
    def export_dict(self) -> Dict[str, Any]:
        """Export the whole buffer to a CPU/NumPy friendly dict for pickling.
        Variable-length candidate tensors are stored as a list of 2D float32 arrays.
        """
        n = len(self.cands_states)
        if n == 0:
            return {"num_transitions": 0}

        cands_list = [cs.detach().cpu().to(T.float32).numpy() for cs in self.cands_states]
        mip_mat = np.stack([ms.detach().cpu().to(T.float32).numpy() for ms in self.mip_states], axis=0)
        node_mat = np.stack([ns.detach().cpu().to(T.float32).numpy() for ns in self.node_states], axis=0)

        data = {
            "num_transitions": n,
            "var_dim": self.var_dim,
            "mip_dim": self.mip_dim,
            "node_dim": self.node_dim,
            "cands_states": cands_list,              # list of (n_cands_i, var_dim)
            "mip_states": mip_mat,                   # (n, mip_dim)
            "node_states": node_mat,                 # (n, node_dim)
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
        }
        # Optional
        if len(self.advantages) == n and len(self.returns) == n:
            data["advantages"] = np.asarray(self.advantages, dtype=np.float32)
            data["returns"] = np.asarray(self.returns, dtype=np.float32)
        return data

    def import_dict(self, payload: Dict[str, Any]):
        """Append transitions from an exported dict into this memory (to `self.device`)."""
        if not payload or payload.get("num_transitions", 0) == 0:
            return
        n = int(payload["num_transitions"])  # type: ignore[index]
        cands_list = payload["cands_states"]
        mip_mat = payload["mip_states"]
        node_mat = payload["node_states"]
        actions = payload["actions"]
        rewards = payload["rewards"]
        dones = payload["dones"]
        log_probs = payload["log_probs"]
        values = payload["values"]
        advantages = payload.get("advantages", None)
        returns = payload.get("returns", None)

        # Basic sanity checks
        if len(cands_list) != n:
            raise ValueError("cands_states length mismatch")
        if mip_mat.shape[0] != n or node_mat.shape[0] != n:
            raise ValueError("state matrices length mismatch")
        if actions.shape[0] != n:
            raise ValueError("actions length mismatch")

        for i in range(n):
            cs_np = np.asarray(cands_list[i], dtype=np.float32)
            ms_np = np.asarray(mip_mat[i], dtype=np.float32)
            ns_np = np.asarray(node_mat[i], dtype=np.float32)

            cs = T.from_numpy(cs_np).to(self.device)
            ms = T.from_numpy(ms_np).to(self.device)
            ns = T.from_numpy(ns_np).to(self.device)

            act = int(actions[i])
            rew = float(rewards[i])
            dn = bool(dones[i])
            lp = float(log_probs[i])
            val = float(values[i])

            # Minimal validation
            if cs.dim() != 2 or cs.size(1) != self.var_dim:
                raise ValueError("Imported cands_state has wrong shape")
            if ms.dim() != 1 or ms.size(0) != self.mip_dim:
                raise ValueError("Imported mip_state has wrong shape")
            if ns.dim() != 1 or ns.size(0) != self.node_dim:
                raise ValueError("Imported node_state has wrong shape")
            if act < 0 or act >= cs.size(0):
                # Clamp or raise; raising helps catch bugs early
                raise ValueError("Imported action out of range for its candidate set")

            self.cands_states.append(cs)
            self.mip_states.append(ms)
            self.node_states.append(ns)
            self.actions.append(act)
            self.rewards.append(rew)
            self.dones.append(dn)
            self.log_probs.append(lp)
            self.values.append(val)

            if advantages is not None and returns is not None:
                # Append lazily; convert to float
                self.advantages.append(float(advantages[i]))
                self.returns.append(float(returns[i]))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_all_data(self) -> Tuple[List[T.Tensor], List[T.Tensor], List[T.Tensor], List[int], List[float], List[bool], List[float], List[float]]:
        if self.is_empty():
            raise RuntimeError("Memory is empty.")
        return (
            self.cands_states.copy(), self.mip_states.copy(), self.node_states.copy(),
            self.actions.copy(), self.rewards.copy(), self.dones.copy(),
            self.log_probs.copy(), self.values.copy(),
        )

    def get_memory_info(self) -> dict:
        if self.is_empty():
            return {
                "num_transitions": 0,
                "memory_empty": True,
                "batch_size": self.batch_size,
                "device": str(self.device),
            }
        cands_sizes = [cs.size(0) for cs in self.cands_states]
        has_adv = len(self.advantages) == len(self.cands_states)
        has_ret = len(self.returns) == len(self.cands_states)
        return {
            "num_transitions": len(self.cands_states),
            "memory_empty": False,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "min_candidates": min(cands_sizes),
            "max_candidates": max(cands_sizes),
            "avg_candidates": float(sum(cands_sizes) / len(cands_sizes)),
            "var_dim": self.var_dim,
            "mip_dim": self.mip_dim,
            "node_dim": self.node_dim,
            "has_advantages": has_adv,
            "has_returns": has_ret,
        }
