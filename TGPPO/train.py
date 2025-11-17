from __future__ import annotations

import os
import io
import gc
import sys
import json
import math
import time
import argparse
import logging
import random
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, NamedTuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import torch

# ---------------- Project imports ----------------
from project.utils import (
    setup_logging, strip_extension, get_device, settings, state_dims, scip_limits, get_reward
)
from project.policy import Actor, Critic
from project.agent import Agent
from project.environment import Environment

# ---------------- Thread hygiene ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

def _fix_torch_threads():
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
_fix_torch_threads()

# ---------------- Data structures ----------------
@dataclass
class TrainArgs:
    logs_dir: str
    scip_setting: str
    time_limit: float
    seeds: List[int]
    per_job_timeout: int
    max_workers: int
    train_iterations: int
    checkpoint_every: int
    # Hyperparameters
    actor_lr: float
    critic_lr: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    policy_clip: float
    entropy_weight: float
    gamma: float
    gae_lambda: float
    batch_size: int
    n_epochs: int
    reward_function: str

class EpisodeMetrics(NamedTuple):
    instance: str
    seed: int
    status: str
    nnodes: int
    solve_time: float
    gap: float
    pdi: float
    episode_return: float

# ---------------- Builders ----------------

def build_models(cfg: TrainArgs) -> Tuple[Actor, Critic]:
    actor = Actor(
        var_dim=state_dims["var_dim"],
        node_dim=state_dims["node_dim"],
        mip_dim=state_dims["mip_dim"],
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    critic = Critic(
        var_dim=state_dims["var_dim"],
        node_dim=state_dims["node_dim"],
        mip_dim=state_dims["mip_dim"],
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    return actor, critic


def build_agent_env_with_models(actor: Actor, critic: Critic, cfg: TrainArgs, seed: int, logger: logging.Logger) -> Tuple[Agent, Environment]:
    device = get_device(device="cpu")

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, weight_decay=1e-2)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=1e-2)

    reward_func = get_reward(cfg.reward_function)

    agent = Agent(
        actor_network=actor,
        actor_optimizer=actor_opt,
        critic_network=critic,
        critic_optimizer=critic_opt,
        policy_clip=cfg.policy_clip,
        entropy_weight=cfg.entropy_weight,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        device=device,
        state_dims=state_dims,
        logger=logger,
    )

    scip_params = settings.get(cfg.scip_setting, {}).copy()
    limits = scip_limits.copy()
    limits["time_limit"] = float(cfg.time_limit)

    env = Environment(
        device=device,
        agent=agent,
        state_dims=state_dims,
        scip_limits=limits,
        scip_params=scip_params,
        scip_seed=seed,
        reward_func=reward_func,
        logger=logger,
    )
    return agent, env

# ---------------- Serialization ----------------

def serialize_weights(actor: Actor, critic: Critic) -> bytes:
    buf = io.BytesIO()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, buf)
    return buf.getvalue()

def load_weights_bytes(actor: Actor, critic: Critic, payload: bytes) -> None:
    buf = io.BytesIO(payload)
    state = torch.load(buf, map_location="cpu")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])

# ---------------- Helpers: parallel batches ----------------

def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

def run_parallel_in_batches(jobs, fn, max_workers, per_job_timeout, tasks_per_child=1,
                            returns_path=False, time_limit_pad=60):
    batch_size = max(1, max_workers * max(1, tasks_per_child))
    outputs = []
    for batch in _batched(jobs, batch_size):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fn, job) for job in batch]
            deadline = per_job_timeout * len(batch) + time_limit_pad
            for fut in as_completed(futs, timeout=deadline):
                try:
                    res = fut.result(timeout=per_job_timeout)
                    outputs.append(res)
                except TimeoutError:
                    outputs.append(None)
                except Exception:
                    outputs.append(None)
    if returns_path:
        metrics, paths = [], []
        for item in outputs:
            if item is None:
                metrics.append(None); paths.append(None)
            else:
                m, p = item
                metrics.append(m); paths.append(p)
        return metrics, paths
    else:
        return outputs

# ---------------- Workers ----------------

def _make_worker_logger() -> logging.Logger:
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


def collect_rollout_job(job: Tuple[str, int, bytes, TrainArgs, Dict[str, Any]]) -> Tuple[EpisodeMetrics, str]:
    instance_path, seed, weights_bytes, cfg, info_dict = job
    logger = _make_worker_logger()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor, critic = build_models(cfg)
    load_weights_bytes(actor, critic, weights_bytes)

    agent, env = build_agent_env_with_models(actor, critic, cfg, seed, logger)

    name = strip_extension(os.path.basename(instance_path)).split(".")[0]
    meta = info_dict.get(name, {})
    cutoff = meta.get("cutoff")
    baseline_nodes = meta.get("baseline_nodes")
    baseline_gap = meta.get("baseline_gap")
    baseline_integral = meta.get("baseline_integral")
    baseline_status = meta.get("baseline_status")

    try:
        env.reset(instance_path, cutoff, baseline_nodes, baseline_gap, baseline_integral, baseline_status)
        done, info, ep_reward = env.run_episode()
        if hasattr(agent, "finalize_advantages"):
            agent.finalize_advantages()
        if not hasattr(agent, "memory") or not hasattr(agent.memory, "export_dict"):
            raise RuntimeError("Agent.memory.export_dict() required")
        traj = agent.memory.export_dict()

        tmp_dir = os.path.join(cfg.logs_dir, "tmp_traj")
        os.makedirs(tmp_dir, exist_ok=True)
        traj_path = os.path.join(tmp_dir, f"traj_{os.getpid()}_{seed}_{time.time_ns()}.pkl")
        with open(traj_path, "wb") as f:
            pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

        status = str(info.get("status"))
        nnodes = int(info.get("nnodes", 0))
        solve_time = float(info.get("scip_solve_time", 0.0))
        gap = float(info.get("gap", 1.0))
        pdi = float(info.get("primalDualIntegral", info.get("pdi", 0.0)))
        metrics = EpisodeMetrics(name, seed, status, nnodes, solve_time, gap, pdi, float(ep_reward))
        return metrics, traj_path
    except Exception:
        logger.exception("collect_rollout_job failed")
        tmp_dir = os.path.join(cfg.logs_dir, "tmp_traj")
        os.makedirs(tmp_dir, exist_ok=True)
        traj_path = os.path.join(tmp_dir, f"traj_error_{os.getpid()}_{seed}_{time.time_ns()}.pkl")
        with open(traj_path, "wb") as f:
            pickle.dump({}, f)
        return EpisodeMetrics(name, seed, "error", int(1e9), cfg.time_limit, 1.0, 1e9, -1.0), traj_path
    finally:
        try:
            del env, agent
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validation_job(job: Tuple[str, int, bytes, TrainArgs, Dict[str, Any]]) -> EpisodeMetrics:
    instance_path, seed, weights_bytes, cfg, info_dict = job
    logger = _make_worker_logger()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor, critic = build_models(cfg)
    load_weights_bytes(actor, critic, weights_bytes)
    actor.eval(); critic.eval()
    agent, env = build_agent_env_with_models(actor, critic, cfg, seed, logger)

    name = strip_extension(os.path.basename(instance_path)).split(".")[0]
    meta = info_dict.get(name, {})
    cutoff = meta.get("cutoff")
    baseline_nodes = meta.get("baseline_nodes")
    baseline_gap = meta.get("baseline_gap")
    baseline_integral = meta.get("baseline_integral")
    baseline_status = meta.get("baseline_status")

    try:
        env.reset(instance_path, cutoff, baseline_nodes, baseline_gap, baseline_integral, baseline_status)
        done, info, ep_reward = env.run_episode(learn=False)
        status = str(info.get("status"))
        nnodes = int(info.get("nnodes", 0))
        solve_time = float(info.get("scip_solve_time", 0.0))
        gap = float(info.get("gap", 1.0))
        pdi = float(info.get("primalDualIntegral", info.get("pdi", 0.0)))
        return EpisodeMetrics(name, seed, status, nnodes, solve_time, gap, pdi, float(ep_reward))
    except Exception:
        logger.exception("validation_job failed")
        return EpisodeMetrics(name, seed, "error", int(1e9), cfg.time_limit, 1.0, 1e9, -1.0)
    finally:
        try:
            del env, agent
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------- Scoring ----------------

def score_from_metrics(results: List[EpisodeMetrics], time_limit: float) -> float:
    if not results:
        return float("inf")
    times, nodes = [], []
    penalties = 0.0
    for r in results:
        par10 = r.solve_time if r.status == "optimal" else 10.0 * time_limit
        times.append(par10)
        nodes.append(r.nnodes)
        if r.status == "error":
            penalties += 1e6
    arr_t = np.asarray(times, dtype=float)
    arr_n = np.asarray(nodes, dtype=float)
    shift = 100.0
    sgm_time = math.exp(np.mean(np.log(arr_t + shift))) - shift
    sgm_nodes = math.exp(np.mean(np.log(arr_n + shift))) - shift
    return float(0.6 * sgm_time + 0.4 * sgm_nodes + penalties)

# ---------------- IO utils ----------------

def list_instances(dirpath: str) -> List[str]:
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if f.endswith('.mps') or f.endswith('.mps.gz') or f.endswith('.lp')]


def save_checkpoint(path: str, actor: Actor, critic: Critic, agent: Agent, it: int, score: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "iter": it,
        "score": score,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_opt": agent.actor_optimizer.state_dict() if hasattr(agent, "actor_optimizer") else None,
        "critic_opt": agent.critic_optimizer.state_dict() if hasattr(agent, "critic_optimizer") else None,
        "hparams": {k: v for k, v in agent.__dict__.items() if k in []},  # placeholder if you wish
    }
    torch.save(payload, path)

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Final training of Tree‑Gate PPO with parallel rollouts")
    parser.add_argument('--instances_dir', type=str, required=True)
    parser.add_argument('--instances_info_dict', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--best_params_json', type=str, required=True, help='JSON with best hyperparameters')
    parser.add_argument('--output_model', type=str, default='output/models/final_model.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='output/checkpoints')

    parser.add_argument('--time_limit', type=float, default=900.0)
    parser.add_argument('--per_job_timeout', type=int, default=900)
    parser.add_argument('--scip_setting', type=str, default='sandbox')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--train_iterations', type=int, default=12)
    parser.add_argument('--checkpoint_every', type=int, default=2)
    parser.add_argument('--max_workers', type=int, default=20)

    parser.add_argument('--val_instances_dir', type=str, default=None)
    parser.add_argument('--val_seeds', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    logger = setup_logging(log_file=os.path.join(args.logs_dir, "training.log"))
    logger.info("Starting final training…")

    with open(args.instances_info_dict, 'rb') as f:
        info_dict = pickle.load(f)

    with open(args.best_params_json, 'r') as f:
        bp = json.load(f)

    # Resolve train/val instances
    train_instances = list_instances(args.instances_dir)
    if not train_instances:
        logger.error("No instances found in --instances_dir")
        sys.exit(1)

    if args.val_instances_dir:
        val_instances = list_instances(args.val_instances_dir)
    else:
        val_instances = []  # optional validation

    # Build TrainArgs from best params
    targs = TrainArgs(
        logs_dir=args.logs_dir,
        scip_setting=args.scip_setting,
        time_limit=args.time_limit,
        seeds=args.seeds,
        per_job_timeout=args.per_job_timeout,
        max_workers=args.max_workers,
        train_iterations=args.train_iterations,
        checkpoint_every=args.checkpoint_every,
        actor_lr=float(bp["actor_lr"]),
        critic_lr=float(bp["critic_lr"]),
        hidden_dim=int(bp["hidden_dim"]),
        num_layers=int(bp["num_layers"]),
        num_heads=int(bp["num_heads"]),
        dropout=float(bp["dropout"]),
        policy_clip=float(bp["policy_clip"]),
        entropy_weight=float(bp["entropy_weight"]),
        gamma=float(bp["gamma"]),
        gae_lambda=float(bp["gae_lambda"]),
        batch_size=int(bp["batch_size"]),
        n_epochs=int(bp["n_epochs"]),
        reward_function=str(bp["reward_function"]),
    )

    # Central learner
    actor, critic = build_models(targs)
    agent, _ = build_agent_env_with_models(actor, critic, targs, seed=0, logger=logger)

    # Logs
    train_rows: List[Dict[str, Any]] = []
    best_score = float("inf")

    for it in range(1, targs.train_iterations + 1):
        t0 = time.time()
        weights = serialize_weights(actor, critic)
        jobs = [(inst, s, weights, targs, info_dict) for inst in train_instances for s in targs.seeds]

        metrics_list, traj_paths = run_parallel_in_batches(
            jobs,
            fn=collect_rollout_job,
            max_workers=targs.max_workers,
            per_job_timeout=targs.per_job_timeout,
            tasks_per_child=1,
            returns_path=True,
        )

        # Import trajectories
        if hasattr(agent, "memory") and hasattr(agent.memory, "clear"):
            agent.memory.clear()
        num_traj = 0
        for path in traj_paths:
            if not path:
                continue
            try:
                with open(path, "rb") as f:
                    traj = pickle.load(f)
                if traj and hasattr(agent.memory, "import_dict"):
                    agent.memory.import_dict(traj)
                    num_traj += 1
            finally:
                try: os.remove(path)
                except OSError: pass

        learn_metrics = agent.learn() or {}
        elapsed = time.time() - t0

        # Optional interim validation
        val_score = None
        if val_instances:
            final_w = serialize_weights(actor, critic)
            vjobs = [(inst, s, final_w, targs, info_dict) for inst in val_instances for s in (args.val_seeds or [0])]
            vmetrics = run_parallel_in_batches(
                vjobs,
                fn=validation_job,
                max_workers=targs.max_workers,
                per_job_timeout=targs.per_job_timeout,
                tasks_per_child=2,
                returns_path=False,
            )
            vmetrics = [m for m in vmetrics if m is not None]
            val_score = score_from_metrics(vmetrics, targs.time_limit)

        row = {
            "iter": it,
            "elapsed_sec": elapsed,
            "num_traj": num_traj,
            "learn_metrics": learn_metrics,
            "val_score": val_score,
        }
        train_rows.append(row)
        pd.DataFrame(train_rows).to_json(os.path.join(args.logs_dir, "train_progress.json"), orient="records", indent=2)

        # Checkpoint
        if (it % targs.checkpoint_every == 0) or (val_score is not None and val_score < best_score):
            score_for_ckpt = val_score if val_score is not None else float(it)
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_iter_{it}.pt")
            save_checkpoint(ckpt_path, actor, critic, agent, it, score_for_ckpt)
            if val_score is not None and val_score < best_score:
                best_score = val_score

        logger.info(f"Iter {it}/{targs.train_iterations}  traj={num_traj}  time={elapsed:.1f}s  val_score={val_score}")

    # Save final model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, args.output_model)
    logger.info(f"Saved final model to {args.output_model}")


if __name__ == "__main__":
    main()
