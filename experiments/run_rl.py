"""
Train and evaluate the RL-based context selector.

Usage:
    python experiments/run_rl.py [--config config.yaml] [--episodes 200]

Outputs:
    results/rl_results.json
    results/rl_agent.pkl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_dataset
from src.data.qa_generator import build_synthetic_dataset
from src.evaluation.evaluator import Evaluator
from src.models.embeddings import EmbeddingModel
from src.models.tinyllama import TinyLlamaModel
from src.rl.agent import EpsilonGreedyBandit, PolicyGradientAgent
from src.rl.environment import ContextSelectionEnv
from src.selectors.rl_selector import RLSelector
from src.utils.logging import ResultsLogger, get_logger

logger = get_logger("run_rl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--agent", choices=["bandit", "pg"], default="bandit",
                   help="bandit=EpsilonGreedyBandit, pg=PolicyGradient")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def train_bandit(
    dataset: list,
    emb_model: EmbeddingModel,
    env: ContextSelectionEnv,
    rl_cfg: dict,
    num_episodes: int,
) -> EpsilonGreedyBandit:
    """Train the epsilon-greedy bandit agent."""
    agent = EpsilonGreedyBandit(
        embedding_dim=emb_model._model.get_sentence_embedding_dimension(),
        epsilon=rl_cfg["epsilon"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        learning_rate=rl_cfg["learning_rate"],
    )

    flat_items = []
    for item in dataset:
        for qa in item["qa_pairs"]:
            flat_items.append((item["story"], qa["question"], qa["answer"]))

    episode_rewards = []
    logger.info(f"Training bandit for {num_episodes} episodes…")

    for ep in tqdm(range(num_episodes)):
        item = flat_items[ep % len(flat_items)]
        story, question, answer = item

        state = env.reset(story, question, answer)
        q_vec = env._q_vec
        c_vecs = env._c_vecs

        # Select chunks using bandit
        selected = agent.select(q_vec, c_vecs, rl_cfg["max_chunks"])

        # Compute reward from environment
        env._selected = selected
        reward = env._compute_reward()

        agent.update(selected, reward)
        episode_rewards.append(reward)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            logger.info(f"  Episode {ep+1}/{num_episodes} | avg_reward={avg_r:.4f} | ε={agent.epsilon:.3f}")

    return agent


def train_policy_gradient(
    dataset: list,
    emb_model: EmbeddingModel,
    env: ContextSelectionEnv,
    rl_cfg: dict,
    num_episodes: int,
) -> PolicyGradientAgent:
    """Train the policy gradient agent."""
    state_dim = env.state_dim()
    agent = PolicyGradientAgent(
        state_dim=state_dim,
        hidden_dim=64,
        learning_rate=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        epsilon=rl_cfg["epsilon"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        max_chunks=rl_cfg["max_chunks"],
    )

    flat_items = []
    for item in dataset:
        for qa in item["qa_pairs"]:
            flat_items.append((item["story"], qa["question"], qa["answer"]))

    episode_rewards = []
    logger.info(f"Training policy gradient agent for {num_episodes} episodes…")

    for ep in tqdm(range(num_episodes)):
        item = flat_items[ep % len(flat_items)]
        story, question, answer = item

        state = env.reset(story, question, answer)
        episode_log_probs = []
        episode_rewards_step = []
        done = False
        step = 0

        while not done and step < rl_cfg["max_chunks"] + 1:
            n = len(env._chunks)
            stop_action = env.n_actions() - 1

            # Build valid actions (unchosen chunk indices + stop)
            # Map chunk indices to [0, max_chunks-1] positions
            valid_positions = []
            for i in range(min(n, agent.max_chunks)):
                if i not in env._selected:
                    valid_positions.append(i)
            valid_positions.append(agent.max_chunks)  # stop

            action, log_prob = agent.act(state, valid_positions)

            # Map position back to actual action
            if action == agent.max_chunks:
                actual_action = stop_action
            else:
                actual_action = action if action < n else stop_action

            state, reward, done = env.step(actual_action)
            episode_log_probs.append(log_prob)
            episode_rewards_step.append(reward)
            step += 1

        # Store final reward for the episode
        final_reward = env._compute_reward()
        for lp in episode_log_probs:
            agent.remember(lp, final_reward)

        loss = agent.update()
        episode_rewards.append(final_reward)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            logger.info(f"  Episode {ep+1}/{num_episodes} | avg_reward={avg_r:.4f} | ε={agent.epsilon:.3f}")

    return agent


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rl_cfg = cfg["rl"]
    num_episodes = args.episodes or rl_cfg["num_episodes"]

    # Dataset
    dataset_path = cfg["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        build_synthetic_dataset(dataset_path)
    dataset = load_dataset(dataset_path)

    # Models
    logger.info("Loading embedding model…")
    emb_model = EmbeddingModel(
        model_id=cfg["embeddings"]["model_id"],
        cache_dir=cfg["embeddings"]["cache_dir"],
    )

    logger.info("Loading TinyLlama…")
    llm = TinyLlamaModel(
        model_id=cfg["model"]["tinyllama_model_id"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
        use_fp16=cfg["model"]["use_fp16"],
    )

    # RL Environment
    env = ContextSelectionEnv(
        embedding_model=emb_model,
        lambda_penalty=rl_cfg["lambda_penalty"],
        max_chunks=rl_cfg["max_chunks"],
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"],
    )

    # Train agent
    if args.agent == "bandit":
        agent = train_bandit(dataset, emb_model, env, rl_cfg, num_episodes)
        agent_name = "rl_bandit"
    else:
        agent = train_policy_gradient(dataset, emb_model, env, rl_cfg, num_episodes)
        agent_name = "rl_pg"

    # Save agent
    agent_path = f"results/{agent_name}.pkl"
    agent.save(agent_path)

    # Evaluate
    results_logger = ResultsLogger(cfg["evaluation"]["results_dir"])
    evaluator = Evaluator(
        llm=llm,
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"],
        results_logger=results_logger,
    )

    rl_selector = RLSelector(
        embedding_model=emb_model,
        agent=agent,
        max_chunks=rl_cfg["max_chunks"],
    )
    rl_selector.name = agent_name

    metrics = evaluator.evaluate_selector(
        selector=rl_selector,
        dataset=dataset,
        hyperparams={
            "episodes": num_episodes,
            "lambda": rl_cfg["lambda_penalty"],
            "max_chunks": rl_cfg["max_chunks"],
        },
        verbose=args.verbose,
    )

    saved_path = results_logger.save(f"{agent_name}_results.json")
    logger.info(f"\nResults saved → {saved_path}")
    print(f"\n[{agent_name}] sub_match={metrics['substring_match']:.3f} "
          f"f1={metrics['f1']:.3f} avg_tokens={metrics['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
