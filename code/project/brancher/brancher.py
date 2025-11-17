import pyscipopt as scip
import torch as T
import numpy as np
import traceback


class Brancher(scip.Branchrule):
    def __init__(self, model, state_dims, device, agent, reward_func, cutoff, logger):
        super().__init__()
        self.model = model
        self.var_dim = state_dims["var_dim"]
        self.node_dim = state_dims["node_dim"]
        self.mip_dim = state_dims["mip_dim"]
        self.device = device
        self.agent = agent
        self.reward_func = reward_func
        self.logger = logger
        self.cutoff = abs(cutoff) if cutoff not in (None, 0) else 1e-6
        self.episode_rewards = []

        self.branch_count = 0
        self.branchexec_count = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

    def _is_solved(self):
        status = self.model.getStatus()
        return status in ("optimal", "infeasible", "unbounded", "timelimit")

    def branchexeclp(self, allowaddcons):
        try:
            self.branchexec_count += 1

            cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branch_count)
            if not cands or cands_state_mat is None:
                self.logger.error("No candidates available for branching.")
                return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            node_state = self.model.getNodeState(self.node_dim).astype('float32')
            mip_state = self.model.getMIPState(self.mip_dim).astype('float32')

            if isinstance(cands_state_mat, np.ndarray):
                cands_state_tensor = T.from_numpy(cands_state_mat.astype(np.float32))
            else:
                cands_state_tensor = T.tensor(cands_state_mat, dtype=T.float32)

            if cands_state_tensor.dim() == 1:
                n_candidates = len(cands)
                if cands_state_tensor.shape[0] == n_candidates * self.var_dim:
                    cands_state_tensor = cands_state_tensor.view(n_candidates, self.var_dim)
                else:
                    cands_state_tensor = cands_state_tensor.unsqueeze(0)

            cands_state_tensor = cands_state_tensor.to(self.device)
            node_state_tensor = T.from_numpy(node_state).to(self.device)
            mip_state_tensor = T.from_numpy(mip_state).to(self.device)

            if self.prev_state is not None:
                done = self._is_solved()
                step_reward = self.reward_func.compute(self.model, done)
                try:
                    self.agent.remember(
                        cands_state=self.prev_state['cands_state'],
                        mip_state=self.prev_state['mip_state'],
                        node_state=self.prev_state['node_state'],
                        action=self.prev_action,
                        reward=float(step_reward),
                        done=bool(done),
                        value=float(self.prev_value),
                        log_prob=float(self.prev_log_prob),
                    )
                except Exception:
                    self.logger.error("Failed to store transition")
                    self.logger.error(traceback.format_exc())
                self.episode_rewards.append(float(step_reward))
                self.logger.info(
                    f"Stored transition for branch {self.branch_count - 1} - "
                    f"Action: {self.prev_action}, Reward: {step_reward:.4f}, Done: {done}"
                )

                if done:
                    self.logger.info(f"Episode completed with total reward: {sum(self.episode_rewards):.4f}")
                    return {"result": scip.SCIP_RESULT.DIDNOTRUN}

            padding_mask = None  # no padding at selection time (exact candidate set)

            action, value, log_prob = self.agent.choose_action(
                cands_state_tensor.cpu().numpy(),
                mip_state_tensor.cpu().numpy(),
                node_state_tensor.cpu().numpy(),
                padding_mask=padding_mask,
                deterministic=False,
            )

            if isinstance(action, np.ndarray):
                action = int(action.item() if action.size == 1 else action[0])
            else:
                action = int(action)

            if not (0 <= action < len(cands)):
                self.logger.error(f"Invalid action selected: {action} for {len(cands)} candidates")
                action = 0
                self.logger.warning("Using fallback action: 0")

            selected_var = cands[action]
            self.model.branchVar(selected_var)
            self.branch_count += 1

            self.prev_state = {
                'cands_state': cands_state_tensor.clone(),
                'mip_state': mip_state_tensor.clone(),
                'node_state': node_state_tensor.clone(),
            }
            self.prev_action = action
            self.prev_value = float(value.item() if isinstance(value, np.ndarray) else value)
            self.prev_log_prob = float(log_prob.item() if isinstance(log_prob, np.ndarray) else log_prob)

            result = scip.SCIP_RESULT.BRANCHED

            # Optional validation
            try:
                if result == scip.SCIP_RESULT.BRANCHED:
                    children = self.model.getChildren()
                    if children:
                        branch_infos = children[0].getBranchInfos()
                        if len(branch_infos) > 1:
                            chosen_variable = branch_infos[1]
                            assert chosen_variable is not None, "Chosen variable is None"
                            assert chosen_variable.isInLP(), "Chosen variable is not in LP"
            except Exception:
                # Do not break on validation failure — logging only
                self.logger.debug("Validation check failed or unavailable.")

            return {"result": result}
        except Exception as e:
            self.logger.error(f"Exception in branching rule: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"result": scip.SCIP_RESULT.DIDNOTRUN}

    def branchfree(self):
        # Called when branchrule is freed; ensure we store final transition once
        if self.prev_state is not None:
            done = True
            terminal_reward = self.reward_func.compute(self.model, done)
            try:
                self.agent.remember(
                    cands_state=self.prev_state['cands_state'],
                    mip_state=self.prev_state['mip_state'],
                    node_state=self.prev_state['node_state'],
                    action=self.prev_action,
                    reward=float(terminal_reward),
                    done=True,
                    value=float(self.prev_value),
                    log_prob=float(self.prev_log_prob),
                )
            except Exception:
                self.logger.error("Failed to store final transition")
                self.logger.error(traceback.format_exc())
            self.episode_rewards.append(float(terminal_reward))
            self.logger.info(
                f"Final transition stored - Reward: {terminal_reward:.4f}, Total episode reward: {sum(self.episode_rewards):.4f}"
            )

        self.prev_state = None
        self.prev_action = None
        self.prev_value = None
        self.prev_log_prob = None

    def get_episode_stats(self):
        return {
            'branch_count': self.branch_count,
            'branchexec_count': self.branchexec_count,
            'episode_rewards': self.episode_rewards.copy(),
            'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0.0,
        }