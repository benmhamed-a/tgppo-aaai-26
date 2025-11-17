from __future__ import annotations
import os, sys, gc, io, re, json, math, time, argparse, logging, random, pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, NamedTuple, Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import torch

# ---------------- Project imports ----------------
from project.utils import (
    setup_logging, strip_extension, get_device, settings, state_dims,
    scip_limits, get_reward
)
from project.policy import Actor, Critic
from project.agent import Agent
from project.environment import Environment

# ---------------- Thread hygiene ----------------
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
@dataclass
class TestArgs:
    logs_dir: str
    scip_setting: str
    time_limit: float
    per_job_timeout: int
    max_workers: int
    seeds: List[int]
    shift: float

class EpisodeMetrics(NamedTuple):
    instance: str; seed: int; status: str
    nnodes: int; solve_time: float; gap: float; pdi: float; episode_return: float
# ---------------------------------------------------------------------------

# ---------- model utilities ------------------------------------------------
def build_models_from_ckpt(ckpt_path: str) -> Tuple[Actor, Critic, Dict[str, Any]]:
    """Load checkpoint -> return instantiated Actor, Critic and hparam dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    h = ckpt.get("hparams", {})            # may be empty

    hidden_dim = int(h.get("hidden_dim", 256))
    num_layers = int(h.get("num_layers", 5))
    num_heads  = int(h.get("num_heads", 8))
    dropout    = float(h.get("dropout", 0.05))

    actor = Actor(
        var_dim  = state_dims["var_dim"],
        node_dim = state_dims["node_dim"],
        mip_dim  = state_dims["mip_dim"],
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        num_heads  = num_heads,
        dropout    = dropout,
    )
    critic = Critic(
        var_dim  = state_dims["var_dim"],
        node_dim = state_dims["node_dim"],
        mip_dim  = state_dims["mip_dim"],
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        num_heads  = num_heads,
        dropout    = dropout,
    )

    # load weights (works for both final_model.pt and training checkpoints)
    if "actor" in ckpt and "critic" in ckpt:
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
    else:
        raise ValueError("Checkpoint lacks 'actor'/'critic' state_dicts")

    return actor, critic, h

# ---------- environment builder -------------------------------------------
def build_eval_env(actor: Actor, critic: Critic, cfg: TestArgs, seed: int,
                   logger: logging.Logger) -> Tuple[Agent, Environment]:

    device = get_device(device="cpu")
    # dummy optimizers (never stepped)
    actor_opt  = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)

    agent = Agent(
        actor_network   = actor,   actor_optimizer  = actor_opt,
        critic_network  = critic,  critic_optimizer = critic_opt,
        policy_clip     = 0.18,    entropy_weight   = 3e-4,
        gamma           = 0.99,    gae_lambda       = 0.95,
        batch_size      = 1024,    n_epochs         = 1,
        device=device,  state_dims=state_dims,      logger=logger,
    )

    scip_params = settings.get(cfg.scip_setting, {}).copy()
    limits      = scip_limits.copy(); limits["time_limit"] = cfg.time_limit

    env = Environment(
        device=device, agent=agent, state_dims=state_dims,
        scip_limits=limits, scip_params=scip_params,
        scip_seed=seed, reward_func=get_reward("reward_h1"), logger=logger,
    )
    return agent, env
# ---------------------------------------------------------------------------

def _batched(seq, n):
    for i in range(0, len(seq), n): yield seq[i:i+n]

def run_parallel_in_batches(jobs, fn, max_workers, per_job_timeout,
                            tasks_per_child=2, time_limit_pad=120):
    batch_size = max(1, max_workers * tasks_per_child)
    outputs = []
    for batch in _batched(jobs, batch_size):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fn, job) for job in batch]
            deadline = per_job_timeout * len(batch) + time_limit_pad
            for fut in as_completed(futs, timeout=deadline):
                try: outputs.append(fut.result(timeout=per_job_timeout))
                except Exception: outputs.append(None)
    return [o for o in outputs if o]

# ---------------- Worker ---------------------------------------------------
def _make_worker_logger() -> logging.Logger:
    lg = logging.getLogger(f"worker-{os.getpid()}"); lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        lg.addHandler(h)
    return lg

def eval_job(job: Tuple[str,int,TestArgs,Dict[str,Any],str]) -> EpisodeMetrics:
    instance_path, seed, cfg, info_dict, ckpt_path = job
    logger = _make_worker_logger()

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    actor, critic, _ = build_models_from_ckpt(ckpt_path)
    actor.eval(); critic.eval()
    agent, env = build_eval_env(actor, critic, cfg, seed, logger)

    name = strip_extension(os.path.basename(instance_path)).split(".")[0]
    cutoff_value = info_dict.get(name, {})
    try:
        env.reset( instance_path, cutoff_value)
        _, info, ep_ret = env.run_episode()
        return EpisodeMetrics(
            name, seed, str(info.get("status")),
            int(info.get("nnodes", 0)),
            float(info.get("scip_solve_time", 0.0)),
            float(info.get("gap", 1.0)),
            float(info.get("primalDualIntegral", info.get("pdi", 0.0))),
            float(ep_ret),
        )
    except Exception:
        logger.exception("eval_job failed")
        return EpisodeMetrics(name, seed, "error", 10**9, cfg.time_limit, 1.0, 10**9, -1.0)
    finally:
        del env, agent; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
# ---------------------------------------------------------------------------

def shifted_geometric_mean(vals: Iterable[float], shift: float) -> Optional[float]:
    good = [v for v in vals if np.isfinite(v)]
    if not good: return None
    arr = np.asarray(good, float); return float(math.exp(np.mean(np.log(arr+shift))) - shift)

def par10(times, statuses, cutoff):  # helper for PAR-10
    return float(np.mean([t if s=="optimal" else 10*cutoff for t,s in zip(times,statuses)]))

# ---------- aggregation helpers (unchanged) --------------------------------
def aggregate_overall(res: List[EpisodeMetrics], cutoff, shift):
    solved   = [r for r in res if r.status=="optimal"]
    unsolved = [r for r in res if r.status!="optimal"]
    by_inst  = {}
    for r in res: by_inst.setdefault(r.instance, []).append(r)

    return {
        "num_runs": len(res),
        "num_instances": len(by_inst),
        "success_rate_runs": len(solved)/len(res) if res else 0,
        "coverage_instances_solved_at_least_once":
            sum( any(r.status=="optimal" for r in vs) for vs in by_inst.values() ) / len(by_inst) if by_inst else 0,
        "sgm_time_solved": shifted_geometric_mean([r.solve_time for r in solved], shift),
        "sgm_pdi_all":     shifted_geometric_mean([r.pdi       for r in res   ], shift),
        "sgm_nodes_all":   shifted_geometric_mean([r.nnodes    for r in res   ], shift),
        "median_time_solved":
            float(np.median([r.solve_time for r in solved])) if solved else None,
        "median_pdi_unsolved":
            float(np.median([r.pdi for r in unsolved])) if unsolved else None,
        "par10_time": par10([r.solve_time for r in res],[r.status for r in res], cutoff),
        "primary_metric": "sgm_time_solved" if solved else "sgm_pdi_all",
        "primary_value":
            shifted_geometric_mean([r.solve_time for r in solved], shift)
            if solved else shifted_geometric_mean([r.pdi for r in res], shift),
    }

def aggregate_per_instance(res: List[EpisodeMetrics], cutoff, shift):
    by = {}
    for r in res: by.setdefault(r.instance, []).append(r)
    rows=[]
    for inst,lst in by.items():
        solved=[r for r in lst if r.status=="optimal"]; unsolved=[r for r in lst if r.status!="optimal"]
        rows.append({
            "instance":inst,
            "runs":len(lst),
            "success_rate":len(solved)/len(lst),
            "sgm_time_solved":shifted_geometric_mean([r.solve_time for r in solved],shift),
            "sgm_pdi_all":shifted_geometric_mean([r.pdi for r in lst],shift),
            "median_time_solved":float(np.median([r.solve_time for r in solved])) if solved else None,
            "median_pdi_unsolved":float(np.median([r.pdi for r in unsolved])) if unsolved else None,
            "best_gap":float(min(r.gap for r in lst)),
            "par10_time":par10([r.solve_time for r in lst],[r.status for r in lst],cutoff),
            "mean_nodes":float(np.mean([r.nnodes for r in lst])),
        })
    df=pd.DataFrame(rows); order=[c for c in ("instance","runs","success_rate","sgm_time_solved",
        "sgm_pdi_all","median_time_solved","median_pdi_unsolved","best_gap","par10_time","mean_nodes") if c in df]
    return df[order].sort_values("instance") if not df.empty else df
# ---------------------------------------------------------------------------

def list_instances(d): return [os.path.join(d,f) for f in os.listdir(d)
                               if f.endswith((".mps",".mps.gz",".lp"))]

# ---------------- main -----------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--instances_dir",required=True)
    ap.add_argument("--instances_info_dict",required=True)
    ap.add_argument("--model_path",required=True)
    ap.add_argument("--logs_dir",required=True)
    ap.add_argument("--time_limit",type=float,default=3600)
    ap.add_argument("--per_job_timeout",type=int,default=3900)
    ap.add_argument("--scip_setting",default="sandbox")
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4])
    ap.add_argument("--max_workers",type=int,default=40)
    ap.add_argument("--shift",type=float,default=100.0)
    args=ap.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    logger=setup_logging(os.path.join(args.logs_dir,"testing.log"))
    logger.info("Starting evaluation of final model…")

    with open(args.instances_info_dict,"rb") as f: info=pickle.load(f)
    insts=list_instances(args.instances_dir)
    if not insts:
        logger.error("No instances found."); sys.exit(1)
    logger.info(f"Testing on {len(insts)} instances, seeds={args.seeds}")

    cfg=TestArgs(args.logs_dir,args.scip_setting,args.time_limit,
                 args.per_job_timeout,args.max_workers,args.seeds,args.shift)

    jobs=[(p,s,cfg,info,args.model_path) for p in insts for s in args.seeds]
    res=run_parallel_in_batches(jobs,eval_job,args.max_workers,args.per_job_timeout)
    raw=pd.DataFrame([r._asdict() for r in res])
    raw.to_csv(os.path.join(args.logs_dir,"test_runs_raw.csv"),index=False)

    overall=aggregate_overall(res,args.time_limit,args.shift)
    with open(os.path.join(args.logs_dir,"overall_summary.json"),"w") as f: json.dump(overall,f,indent=2)
    perinst=aggregate_per_instance(res,args.time_limit,args.shift)
    perinst.to_csv(os.path.join(args.logs_dir,"per_instance_summary.csv"),index=False)

    logger.info("=== Overall summary ===\n"+json.dumps(overall,indent=2))

if __name__=="__main__":
    main()