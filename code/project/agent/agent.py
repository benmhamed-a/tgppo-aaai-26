import torch
from project.memory import Memory
import torch.nn.functional as F
from torch.distributions import Categorical
import traceback
import logging
import pandas as pd
import os

def save_metrics_to_csv(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([metrics])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


class Agent:
    """
    Policy Gradient Branching Agent (PPO).
    """

    def __init__(self, actor_network, actor_optimizer, critic_network, critic_optimizer,
                 policy_clip, entropy_weight, gamma, gae_lambda, batch_size, n_epochs,
                 device, state_dims, logger=None):
        self.actor_network = actor_network
        self.actor_optimizer = actor_optimizer
        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        self.policy_clip = policy_clip
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.state_dims = state_dims
        self.logger = logger or logging.getLogger(__name__)

        self.memory = Memory(self.batch_size, self.state_dims, self.device, self.logger)

        # Move networks to device
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)

        # Training statistics
        self.training_step = 0
        self.update_counter = 0

    def remember(self, cands_state, mip_state, node_state, action, reward, done, value, log_prob):
        try:
            self.memory.store(
                cands_state=cands_state,
                mip_state=mip_state,
                node_state=node_state,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )
        except Exception:
            self.logger.error("Error in storing transition in memory")
            self.logger.error(traceback.format_exc())

    def choose_action(self, cands_state, mip_state, node_state, padding_mask=None, deterministic=False):
        try:
            # Ensure inputs are tensors on correct device
            cands_state = torch.as_tensor(cands_state, dtype=torch.float32, device=self.device)
            mip_state = torch.as_tensor(mip_state, dtype=torch.float32, device=self.device)
            node_state = torch.as_tensor(node_state, dtype=torch.float32, device=self.device)

            if cands_state.dim() == 2:
                cands_state = cands_state.unsqueeze(0)
            if mip_state.dim() == 1:
                mip_state = mip_state.unsqueeze(0)
            if node_state.dim() == 1:
                node_state = node_state.unsqueeze(0)

            if padding_mask is not None:
                padding_mask = torch.as_tensor(padding_mask, dtype=torch.bool, device=self.device)
                if padding_mask.dim() == 1:
                    padding_mask = padding_mask.unsqueeze(0)

            self.actor_network.eval()
            self.critic_network.eval()

            with torch.no_grad():
                action_probs = self.actor_network(cands_state, node_state, mip_state, padding_mask)
                value = self.critic_network(cands_state, node_state, mip_state, padding_mask)
                value = value.squeeze(-1)  # [batch]

                if deterministic:
                    action = action_probs.argmax(dim=-1)
                    # max prob log
                    max_prob = action_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
                    log_prob = torch.log(max_prob.clamp_min(1e-12))
                else:
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

            action_np = action.cpu().numpy()
            value_np = value.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()

            if action_np.shape[0] == 1:
                return action_np[0], value_np[0], log_prob_np[0]
            return action_np, value_np, log_prob_np
        except Exception as e:
            self.logger.error(f"Error in choose_action: {e}")
            self.logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def _compute_gae(self, rewards, values, dones, bootstrap_value):
        """Compute GAE with scalar bootstrap value for the state after the last step.
        Args:
            rewards: [T]
            values:  [T]
            dones:   [T] float tensor (0.0 or 1.0)
            bootstrap_value: scalar tensor for V(s_{T}) if episode not done, else 0
        Returns:
            advantages [T], returns [T]
        """
        T_ = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        next_value = bootstrap_value
        gae = torch.zeros((), device=rewards.device)
        for t in reversed(range(T_)):
            nonterminal = 1.0 - dones[t]
            nv = next_value if t == T_ - 1 else values[t + 1]
            nv = nv * (1.0 - dones[t + 1]) if t < T_ - 1 else nv
            delta = rewards[t] + self.gamma * nv * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        return advantages, returns

    def learn(self):
        try:
            self.logger.debug("Started Learning...")
            if self.memory.is_empty():
                self.logger.debug("Memory is empty, cannot train")
                return {}
            if len(self.memory) < self.batch_size:
                self.logger.debug(f"Not enough samples in memory for training. Have {len(self.memory)}, need {self.batch_size}")
                return {}

            (cands_states_list, mip_states_list, node_states_list,
             actions_list, rewards_list, dones_list, log_probs_list, values_list) = self.memory.get_all_data()

            actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
            old_values = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=self.device)

            mip_states = torch.stack(mip_states_list)
            node_states = torch.stack(node_states_list)

            # Build padded last state to compute bootstrap
            max_candidates = max(cs.size(0) for cs in cands_states_list)
            last_cs = cands_states_list[-1]
            if last_cs.size(0) < max_candidates:
                pad = torch.zeros(max_candidates - last_cs.size(0), last_cs.size(1), device=self.device)
                last_cs_padded = torch.cat([last_cs, pad], dim=0).unsqueeze(0)
                last_mask = torch.cat([
                    torch.zeros(last_cs.size(0), dtype=torch.bool, device=self.device),
                    torch.ones(max_candidates - last_cs.size(0), dtype=torch.bool, device=self.device),
                ], dim=0).unsqueeze(0)
            else:
                last_cs_padded = last_cs.unsqueeze(0)
                last_mask = torch.zeros(last_cs.size(0), dtype=torch.bool, device=self.device).unsqueeze(0)

            if dones[-1] > 0.5:
                bootstrap_value = torch.tensor(0.0, device=self.device)
            else:
                bootstrap_value = self.critic_network(
                    last_cs_padded, node_states[-1:].unsqueeze(0).squeeze(0), mip_states[-1:].unsqueeze(0).squeeze(0), last_mask
                ).squeeze(-1)[0]

            advantages, returns = self._compute_gae(rewards, old_values, dones, bootstrap_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            num_updates = 0

            self.actor_network.train()
            self.critic_network.train()

            for epoch in range(self.n_epochs):
                self.logger.debug(f"Epoch {epoch+1}/{self.n_epochs}")
                for batch in self.memory.get_batch_generator(self.batch_size):
                    (mb_cands_states, mb_cands_masks, mb_mip_states, mb_node_states,
                     mb_actions, _mb_rewards, _mb_dones, mb_old_log_probs, mb_old_values, batch_indices) = batch

                    mb_advantages = advantages[batch_indices]
                    mb_returns = returns[batch_indices]
                    mb_padding_masks = mb_cands_masks

                    action_probs = self.actor_network(mb_cands_states, mb_node_states, mb_mip_states, mb_padding_masks)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss -= self.entropy_weight * entropy.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
                    self.actor_optimizer.step()

                    values = self.critic_network(mb_cands_states, mb_node_states, mb_mip_states, mb_padding_masks)
                    values = values.view(-1)
                    mb_returns = mb_returns.view(-1)
                    critic_loss = F.mse_loss(values, mb_returns)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
                    self.critic_optimizer.step()

                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

            # Clear memory after updates
            self.memory.clear()
            self.update_counter += 1

            advantage_std = advantages.std().item()

            approx_kl = 0.0
            if num_updates > 0:
                with torch.no_grad():
                    last_action_probs = self.actor_network(mb_cands_states, mb_node_states, mb_mip_states, mb_padding_masks)
                    last_dist = Categorical(last_action_probs)
                    last_log_probs = last_dist.log_prob(mb_actions)
                    approx_kl = (mb_old_log_probs - last_log_probs).mean().item()

            episode_return = rewards.sum().item()

            metrics = {
                'update_step': self.update_counter,
                'actor_loss': total_actor_loss / max(num_updates, 1),
                'critic_loss': total_critic_loss / max(num_updates, 1),
                'entropy': total_entropy / max(num_updates, 1),
                'advantage_std': advantage_std,
                'approx_kl_divergence': approx_kl,
                'episode_return': episode_return,
                'num_updates': num_updates,
            }

            self.logger.info(
                f"PPO Update {self.update_counter} | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Critic Loss: {metrics['critic_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"KL Div: {metrics['approx_kl_divergence']:.6f} | "
                f"Return: {metrics['episode_return']:.2f}"
            )

            save_metrics_to_csv(metrics, filepath="output/training_metrics.csv")
            return metrics
        except Exception as e:
            self.logger.error(f"Error in learn: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def save_models(self, path):
        try:
            torch.save({
                'actor_state_dict': self.actor_network.state_dict(),
                'critic_state_dict': self.critic_network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'update_counter': self.update_counter,
            }, f"{path}_checkpoint.pth")
            self.logger.info(f"Models saved to {path}_checkpoint.pth")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            self.logger.error(traceback.format_exc())

    def load_models(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.update_counter = checkpoint.get('update_counter', 0)
            self.logger.info(f"Models loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.logger.error(traceback.format_exc())