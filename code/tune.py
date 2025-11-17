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
from typing import List, Dict, Any, Tuple, NamedTuple
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def run_parallel_in_batches(jobs, fn, max_workers, per_job_timeout, tasks_per_child=1, returns_path=False, time_limit_pad=60):
    """
    Run `fn(job)` over `jobs` in batches. Each batch uses a fresh executor.
    If returns_path=True, fn returns (metrics, path); else just metrics.
    """
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
                    # mark a timeout; caller can decide how to penalize/cancel
                    outputs.append(None)
                except Exception:
                    outputs.append(None)
    # split metrics/paths if needed
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

# Project imports
from project.utils import (
    setup_logging, strip_extension, get_device, settings, state_dims, scip_limits, get_reward
)
from project.policy import Actor, Critic
from project.agent import Agent
from project.environment import Environment

# Thread hygiene
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
class TrialArgs:
    logs_dir: str
    scip_setting: str
    time_limit: float
    seeds: List[int]
    per_job_timeout: int
    train_iterations: int
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

@dataclass
class Fold:
    train: List[str]
    val: List[str]

@dataclass
class OuterFold:
    train20: List[str]
    test5: List[str]

# ---------------- Model / env builders ----------------

def build_models(cfg: TrialArgs) -> Tuple[Actor, Critic]:
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


def build_agent_env_with_models(actor: Actor, critic: Critic, cfg: TrialArgs, seed: int, logger: logging.Logger) -> Tuple[Agent, Environment]:
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

# ---------------- Workers ----------------

def _make_worker_logger() -> logging.Logger:
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
    return logger


def collect_rollout_job(job: Tuple[str, int, bytes, TrialArgs, Dict[str, Any]]) -> Tuple[EpisodeMetrics, str]:
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


def validation_job(job: Tuple[str, int, bytes, TrialArgs, Dict[str, Any]]) -> EpisodeMetrics:
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

# ---------------- Folds construction ----------------

def make_outer5(instances: List[str], info_dict: Dict[str, Any]) -> List[OuterFold]:
    def diff_key(p):
        name = strip_extension(os.path.basename(p)).split(".")[0]
        bn = info_dict.get(name, {}).get("baseline_nodes")
        try:
            return float("inf") if bn is None else float(bn)
        except Exception:
            return float("inf")
    sorted_all = sorted(instances, key=diff_key)
    n = len(sorted_all)
    if n < 10:
        raise ValueError("Not enough instances for outer CV; need >=10, ideally 25.")
    if n == 25:
        slices = [sorted_all[i*5:(i+1)*5] for i in range(5)]
        folds: List[OuterFold] = []
        for test_idx in range(5):
            test5 = slices[test_idx]
            train20 = [p for i,s in enumerate(slices) if i != test_idx for p in s]
            folds.append(OuterFold(train20=train20, test5=test5))
        return folds
    # For other n, build 5 folds by round‑robin
    k = 5
    buckets = [[] for _ in range(k)]
    for i, p in enumerate(sorted_all):
        buckets[i % k].append(p)
    folds = []
    for test_idx in range(k):
        test = buckets[test_idx]
        train = [p for i,b in enumerate(buckets) if i != test_idx for p in b]
        folds.append(OuterFold(train20=train, test5=test))
    return folds


def make_inner_folds(train20: List[str], info_dict: Dict[str, Any], k: int) -> List[Fold]:
    def diff_key(p):
        name = strip_extension(os.path.basename(p)).split(".")[0]
        bn = info_dict.get(name, {}).get("baseline_nodes")
        try:
            return float("inf") if bn is None else float(bn)
        except Exception:
            return float("inf")
    sorted_train = sorted(train20, key=diff_key)
    buckets = [[] for _ in range(k)]
    for i, p in enumerate(sorted_train):
        buckets[i % k].append(p)
    folds: List[Fold] = []
    for val_idx in range(k):
        val = buckets[val_idx]
        tr = [p for i,b in enumerate(buckets) if i != val_idx for p in b]
        folds.append(Fold(train=tr, val=val))
    return folds

# ---- corrected inner_objective_builder ----
def inner_objective_builder(inner_cfg: Dict[str, Any]):
    """
    Optuna objective that averages validation score across inner folds.
    Compatible with Python 3.8 (no maxtasksperchild kw); uses batched executors.
    """
    info_dict: Dict[str, Any] = inner_cfg["info_dict"]
    folds: List[Fold] = inner_cfg["folds"]
    base_args: Dict[str, Any] = inner_cfg["base_args"]

    def objective(trial: optuna.Trial) -> float:
        # ----- Hyperparameter suggestions -----
        actor_lr = trial.suggest_float("actor_lr", 1e-6, 3e-4, log=True)
        critic_lr = trial.suggest_float("critic_lr", 1e-6, 3e-4, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256, 320, 384])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        policy_clip = trial.suggest_float("policy_clip", 0.05, 0.3)
        entropy_weight = trial.suggest_float("entropy_weight", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.92, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
        batch_size = trial.suggest_categorical("batch_size", [32, 48, 64, 96])
        n_epochs = trial.suggest_int("n_epochs", 1, 6)
        reward_function = trial.suggest_categorical("reward_function", ["reward_h1", "reward_h2", "reward_h3"])

        targs = TrialArgs(
            logs_dir=base_args["logs_dir"],
            scip_setting=base_args["scip_setting"],
            time_limit=base_args["time_limit"],
            seeds=base_args["seeds"],
            per_job_timeout=base_args["per_job_timeout"],
            train_iterations=base_args["train_iterations"],
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            policy_clip=policy_clip,
            entropy_weight=entropy_weight,
            gamma=gamma,
            gae_lambda=gae_lambda,
            batch_size=batch_size,
            n_epochs=n_epochs,
            reward_function=reward_function,
        )

        fold_scores: List[float] = []
        for fold_idx, fold in enumerate(folds):
            logger = logging.getLogger(f"inner-fold-{fold_idx}")
            actor, critic = build_models(targs)
            agent, _ = build_agent_env_with_models(actor, critic, targs, seed=0, logger=logger)

            # ---------- Training iterations (collect -> import -> learn) ----------
            for it in range(targs.train_iterations):
                weights = serialize_weights(actor, critic)
                jobs = [(inst, s, weights, targs, info_dict) for inst in fold.train for s in targs.seeds]

                # Collect rollouts in batches; each worker writes a temp pickle
                metrics_list, traj_paths = run_parallel_in_batches(
                    jobs,
                    fn=collect_rollout_job,
                    max_workers=base_args["max_workers"],
                    per_job_timeout=targs.per_job_timeout,
                    tasks_per_child=1,          # strongest protection against leaks
                    returns_path=True,
                )

                # Import trajectories
                if hasattr(agent, "memory") and hasattr(agent.memory, "clear"):
                    agent.memory.clear()

                for path in traj_paths:
                    if not path:
                        continue
                    try:
                        with open(path, "rb") as f:
                            traj = pickle.load(f)
                        if traj and hasattr(agent.memory, "import_dict"):
                            agent.memory.import_dict(traj)
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass

                # PPO update
                agent.learn()

            # ---------- Validation ----------
            final_w = serialize_weights(actor, critic)
            vjobs = [(inst, s, final_w, targs, info_dict) for inst in fold.val for s in targs.seeds]

            vmetrics = run_parallel_in_batches(
                vjobs,
                fn=validation_job,
                max_workers=base_args["max_workers"],
                per_job_timeout=targs.per_job_timeout,
                tasks_per_child=2,        # validation is lighter; reuse workers a bit
                returns_path=False,
            )
            vmetrics = [m for m in vmetrics if m is not None]

            fscore = score_from_metrics(vmetrics, targs.time_limit)
            fold_scores.append(fscore)

            # Optuna pruning: report average so far
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective

# ---------------- Utilities ----------------

def list_instances(dirpath: str) -> List[str]:
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if f.endswith('.mps') or f.endswith('.mps.gz') or f.endswith('.lp')]


def aggregate_best_params(best_params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not best_params_list:
        return {}
    keys = best_params_list[0].keys()
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [bp[k] for bp in best_params_list if k in bp]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)) and all(isinstance(v, (int, float)) for v in vals):
            agg[k] = float(np.median(vals))
        else:
            cnt = Counter(vals)
            agg[k] = cnt.most_common(1)[0][0]
    return agg

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Nested 5-fold CV with inner Optuna for Tree-Gate PPO")
    parser.add_argument('--instances_dir', type=str, required=True)
    parser.add_argument('--instances_info_dict', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--time_limit', type=float, default=900.0)
    parser.add_argument('--per_job_timeout', type=int, default=900)
    parser.add_argument('--scip_setting', type=str, default='sandbox')
    parser.add_argument('--outer_seeds', type=int, nargs='+', default=[0,1])
    parser.add_argument('--inner_seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--inner_folds', type=int, default=2, choices=[2,3,4,5])
    parser.add_argument('--train_iterations_inner', type=int, default=3)
    parser.add_argument('--train_iterations_refit', type=int, default=6)
    parser.add_argument('--max_workers', type=int, default=20)
    parser.add_argument('--sampler', type=str, default='tpe', choices=['tpe','cmaes'])
    parser.add_argument('--pruner', type=str, default='median', choices=['median','sha'])
    parser.add_argument('--n_trials_inner', type=int, default=60)
    parser.add_argument('--storage_inner', type=str, default='optuna_inner.db')
    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    logger = setup_logging(log_file=os.path.join(args.logs_dir, "nested_cv.log"))
    logger.info("Starting nested 5-fold CV…")

    with open(args.instances_info_dict, 'rb') as f:
        info_dict = pickle.load(f)

    instances = list_instances(args.instances_dir)
    if len(instances) < 10:
        logger.error("Not enough instances; need at least 10 (25 recommended).")
        sys.exit(1)

    outer_folds = make_outer5(instances, info_dict)
    logger.info(f"Built {len(outer_folds)} outer folds")

    all_test_rows: List[Dict[str, Any]] = []
    outer_best_params: List[Dict[str, Any]] = []

    for of_idx, ofold in enumerate(outer_folds):
        logger.info(f"===== Outer fold {of_idx} =====")
        logger.info("Test5: " + ", ".join(os.path.basename(p) for p in ofold.test5))

        # Build inner folds on the 20 training instances
        inner_folds = make_inner_folds(ofold.train20, info_dict, k=args.inner_folds)
        logger.info(f"Inner folds: {len(inner_folds)}")

        # Inner Optuna HPO
        base_args = dict(
            logs_dir=args.logs_dir,
            scip_setting=args.scip_setting,
            time_limit=args.time_limit,
            seeds=args.inner_seeds,
            per_job_timeout=args.per_job_timeout,
            train_iterations=args.train_iterations_inner,
            max_workers=args.max_workers,
        )
        inner_cfg = dict(info_dict=info_dict, folds=inner_folds, base_args=base_args)
        objective = inner_objective_builder(inner_cfg)

        storage_url = f"sqlite:///{args.storage_inner}" if args.storage_inner else None
        if args.sampler == 'tpe':
            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42)
        else:
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        if args.pruner == 'median':
            pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=1, interval_steps=1)
        else:
            pruner = SuccessiveHalvingPruner(reduction_factor=3, min_resource=1)

        study_name = f"inner_ofold{of_idx}"
        study = optuna.create_study(
            storage=storage_url,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            direction='minimize',
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.n_trials_inner, n_jobs=1, gc_after_trial=True, show_progress_bar=False)
        best = study.best_trial
        logger.info(f"Outer {of_idx}: best inner score={best.value}")
        logger.info("Params:\n" + json.dumps(best.params, indent=2))
        outer_best_params.append(best.params)

        # Refit on all 20 training instances with best params (longer iterations)
        targs = TrialArgs(
            logs_dir=args.logs_dir,
            scip_setting=args.scip_setting,
            time_limit=args.time_limit,
            seeds=args.inner_seeds,  # training seeds; we can keep the inner seeds here
            per_job_timeout=args.per_job_timeout,
            train_iterations=args.train_iterations_refit,
            actor_lr=float(best.params["actor_lr"]),
            critic_lr=float(best.params["critic_lr"]),
            hidden_dim=int(best.params["hidden_dim"]),
            num_layers=int(best.params["num_layers"]),
            num_heads=int(best.params["num_heads"]),
            dropout=float(best.params["dropout"]),
            policy_clip=float(best.params["policy_clip"]),
            entropy_weight=float(best.params["entropy_weight"]),
            gamma=float(best.params["gamma"]),
            gae_lambda=float(best.params["gae_lambda"]),
            batch_size=int(best.params["batch_size"]),
            n_epochs=int(best.params["n_epochs"]),
            reward_function=str(best.params["reward_function"]),
        )

        logger.info(f"Refitting on 20 instances for outer fold {of_idx}…")
        actor, critic = build_models(targs)
        agent, _ = build_agent_env_with_models(actor, critic, targs, seed=0, logger=logger)
        for it in range(targs.train_iterations):
            weights = serialize_weights(actor, critic)
            jobs = [(inst, s, weights, targs, info_dict) for inst in ofold.train20 for s in targs.seeds]
            traj_paths: List[str] = []
            with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=None, maxtasksperchild=1) as ex:
                futures = [ex.submit(collect_rollout_job, job) for job in jobs]
                deadline = targs.per_job_timeout * len(jobs) + 60
                for fut in as_completed(futures, timeout=deadline):
                    try:
                        m, p = fut.result(timeout=targs.per_job_timeout)
                        traj_paths.append(p)
                    except TimeoutError:
                        for f in futures: f.cancel()
                        raise
                    except Exception:
                        pass
            if hasattr(agent, "memory") and hasattr(agent.memory, "clear"):
                agent.memory.clear()
            for path in traj_paths:
                try:
                    with open(path, "rb") as f:
                        traj = pickle.load(f)
                    if traj and hasattr(agent.memory, "import_dict"):
                        agent.memory.import_dict(traj)
                finally:
                    try: os.remove(path)
                    except OSError: pass
            agent.learn()

        # Test on outer test5 with outer seeds (potentially > inner seeds)
        logger.info(f"Testing outer fold {of_idx} on its 5 held‑out instances…")
        final_w = serialize_weights(actor, critic)
        test_jobs = [(inst, s, final_w, targs, info_dict) for inst in ofold.test5 for s in args.outer_seeds]
        test_metrics: List[EpisodeMetrics] = []
        with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=None, maxtasksperchild=1) as ex:
            futures = [ex.submit(validation_job, job) for job in test_jobs]
            deadline = args.per_job_timeout * len(test_jobs) + 60
            for fut in as_completed(futures, timeout=deadline):
                try:
                    m = fut.result(timeout=args.per_job_timeout)
                    test_metrics.append(m)
                except TimeoutError:
                    for f in futures: f.cancel()
                    test_metrics.append(EpisodeMetrics("timeout", -1, "error", int(1e9), args.time_limit, 1.0, 1e9, -1.0))
                except Exception:
                    test_metrics.append(EpisodeMetrics("error", -1, "error", int(1e9), args.time_limit, 1.0, 1e9, -1.0))

        # Save per‑fold CSV
        df = pd.DataFrame([{
            "outer_fold": of_idx,
            "instance": m.instance,
            "seed": m.seed,
            "status": m.status,
            "nnodes": m.nnodes,
            "solve_time": m.solve_time,
            "gap": m.gap,
            "pdi": m.pdi,
            "episode_return": m.episode_return,
        } for m in test_metrics])
        fold_dir = os.path.join(args.logs_dir, f"outer_fold_{of_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        csv_path = os.path.join(fold_dir, "test_metrics.csv")
        df.to_csv(csv_path, index=False)
        with open(os.path.join(fold_dir, "best_params.json"), 'w') as f:
            json.dump(best.params, f, indent=2)
        all_test_rows.extend(df.to_dict(orient="records"))

        fold_score = score_from_metrics(test_metrics, args.time_limit)
        logger.info(f"Outer fold {of_idx} test score = {fold_score:.4f}")

    # Aggregate across outer folds
    all_df = pd.DataFrame(all_test_rows)
    all_csv = os.path.join(args.logs_dir, "outer_test_all.csv")
    all_df.to_csv(all_csv, index=False)

    # Summary stats
    def safe_median(x):
        return float(np.median(x)) if len(x) else float('nan')
    summary = {
        "success_rate": float(np.mean((all_df["status"] == "optimal").values)) if len(all_df) else 0.0,
        "median_nnodes": safe_median(all_df["nnodes"].values if "nnodes" in all_df else []),
        "median_par10_time": None,
        "median_pdi": safe_median(all_df["pdi"].values if "pdi" in all_df else []),
    }
    # Compute PAR-10 per row then median
    par10 = []
    for _, r in all_df.iterrows():
        par10.append(r["solve_time"] if r["status"] == "optimal" else 10.0 * args.time_limit)
    summary["median_par10_time"] = safe_median(par10)

    with open(os.path.join(args.logs_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Aggregate hyperparams across outer folds
    agg_params = aggregate_best_params(outer_best_params)
    with open(os.path.join(args.logs_dir, "best_hparams_aggregate.json"), 'w') as f:
        json.dump(agg_params, f, indent=2)

    logger.info("=== Nested CV finished ===")
    logger.info("Summary: " + json.dumps(summary, indent=2))
    logger.info("Aggregate best hparams: " + json.dumps(agg_params, indent=2))


if __name__ == "__main__":
    main()
