import numpy as np
import math
from typing import List, Tuple

# For features
from src.evaluation.metrics import substring_match

# For selectors
from src.selectors.topk_selector import TopKSelector
from src.selectors.keyword_selector import KeywordSelector
from src.selectors.sliding_window import SlidingWindowSelector
from src.evaluation.evaluator import TruncatedSelector
from src.utils.chunking import chunk_by_tokens


def compute_dataset_features(dataset: list, emb_model, token_budget: int) -> np.ndarray:
    """Computes a 10-dimensional feature vector summarizing the dataset characteristics."""
    if not dataset:
        return np.zeros(10, dtype=np.float32)

    story_lengths = []
    question_lengths = []
    answer_lengths = []
    unique_words = set()
    total_words = 0
    keyword_densities = []
    chunk_counts = []
    answer_positions = []
    semantic_spreads = []
    question_types = 0
    total_questions = 0

    qa_words = {"who", "what", "when", "where", "why", "how"}

    for item in dataset:
        story = item.get("story", "")
        # Very rough token approximation by splitting on space
        story_words = story.split()
        story_len = len(story_words) * 1.3
        story_lengths.append(story_len)
        
        for w in story_words:
            unique_words.add(w.lower())
            total_words += 1
            
        chunks = chunk_by_tokens(story, 100, 10)  # assume 100 token chunks
        chunk_counts.append(len(chunks))
        
        if len(chunks) > 1 and emb_model is not None:
            try:
                emb_vecs = emb_model.encode(chunks)
                sims = []
                for i in range(len(emb_vecs)):
                    for j in range(i+1, len(emb_vecs)):
                        norm1 = np.linalg.norm(emb_vecs[i])
                        norm2 = np.linalg.norm(emb_vecs[j])
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(emb_vecs[i], emb_vecs[j]) / (norm1 * norm2)
                            sims.append(sim)
                if sims:
                    semantic_spreads.append(np.std(sims))
                else:
                    semantic_spreads.append(0.0)
            except:
                semantic_spreads.append(0.0)
        else:
            semantic_spreads.append(0.0)

        qa_pairs = item.get("qa_pairs", [])
        for qa in qa_pairs:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            
            q_words = q.split()
            a_words = a.split()
            
            question_lengths.append(len(q_words) * 1.3)
            answer_lengths.append(len(a_words) * 1.3)
            
            q_lower = q.lower()
            if q_words and q_words[0].lower() in qa_words:
                question_types += 1
            total_questions += 1
            
            q_set = set([w.lower() for w in q_words])
            s_set = set([w.lower() for w in story_words])
            if len(q_set) > 0:
                overlap = len(q_set.intersection(s_set)) / len(q_set)
                keyword_densities.append(overlap)
            
            try:
                idx = story.lower().find(a.lower())
                if idx >= 0 and len(story) > 0:
                    answer_positions.append(idx / len(story))
                else:
                    answer_positions.append(0.5)
            except:
                answer_positions.append(0.5)

    if total_words == 0:
        total_words = 1
    if total_questions == 0:
        total_questions = 1

    features = np.zeros(10, dtype=np.float32)
    features[0] = np.mean(story_lengths) if story_lengths else 0
    features[1] = np.mean(question_lengths) if question_lengths else 0
    features[2] = len(unique_words) / total_words
    features[3] = np.mean(answer_lengths) if answer_lengths else 0
    features[4] = np.mean(keyword_densities) if keyword_densities else 0
    features[5] = np.mean(chunk_counts) if chunk_counts else 0
    features[6] = np.mean(answer_positions) if answer_positions else 0.5
    features[7] = np.mean(semantic_spreads) if semantic_spreads else 0
    features[8] = question_types / total_questions
    features[9] = features[0] / token_budget if token_budget > 0 else 0

    return features


class SelectorBandit:
    """A Contextual Multi-Armed Bandit using UCB1 to select the best context selector."""
    
    ARM_NAMES = [
        "topk_semantic",
        "topk_hybrid",
        "keyword",
        "sliding_window",
        "truncated_head",
        "truncated_tail"
    ]

    def __init__(self, n_arms=6, state_dim=10, C=1.4, alpha=1.0, beta=0.3, gamma=0.2):
        self.n_arms = n_arms
        self.state_dim = state_dim
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.q_values = np.zeros(n_arms, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int32)
        self.total_steps = 0
        
        self.reward_history: List[Tuple[int, float, str]] = []
        self.best_arm_history: List[str] = []

    def select_arm(self, state: np.ndarray) -> int:
        """Select an arm using the UCB1 algorithm."""
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            exploitation = self.q_values[i]
            exploration = self.C * np.sqrt(np.log(self.total_steps) / (self.counts[i] + 1e-8))
            ucb_values[i] = exploitation + exploration
            
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float, state: np.ndarray) -> None:
        """Update Q-values and counts for the selected arm."""
        self.counts[arm] += 1
        self.total_steps += 1
        
        n = self.counts[arm]
        value = self.q_values[arm]
        self.q_values[arm] = value + (reward - value) / n
        
        arm_name = self.ARM_NAMES[arm]
        self.reward_history.append((self.total_steps, reward, arm_name))
        best_arm_idx = int(np.argmax(self.q_values))
        self.best_arm_history.append(self.ARM_NAMES[best_arm_idx])

    def get_arm_stats(self) -> dict:
        """Returns current Q-values, counts, and UCB scores."""
        stats = {}
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb = float('inf')
            else:
                ucb = self.q_values[i] + self.C * np.sqrt(np.log(max(1, self.total_steps)) / self.counts[i])
            stats[self.ARM_NAMES[i]] = {
                "q_value": float(self.q_values[i]),
                "count": int(self.counts[i]),
                "ucb": float(ucb)
            }
        return stats

    def save(self, path: str) -> None:
        """Save the agent state to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                "q_values": self.q_values,
                "counts": self.counts,
                "total_steps": self.total_steps,
                "reward_history": self.reward_history,
                "best_arm_history": self.best_arm_history,
                "params": {"C": self.C, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
            }, f)

    def load(self, path: str) -> None:
        """Load the agent state from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_values = data["q_values"]
            self.counts = data["counts"]
            self.total_steps = data["total_steps"]
            self.reward_history = data["reward_history"]
            self.best_arm_history = data["best_arm_history"]
            params = data.get("params", {})
            if params:
                self.C = params.get("C", self.C)
                self.alpha = params.get("alpha", self.alpha)
                self.beta = params.get("beta", self.beta)
                self.gamma = params.get("gamma", self.gamma)


class SelectorRLTrainer:
    """Orchestrates the training loop for the SelectorBandit on a dataset."""
    
    def __init__(self, dataset, emb_model, llm, token_budget, chunk_size, overlap,
                 n_episodes=30, C=1.4, alpha=1.0, beta=0.3, gamma=0.2):
        self.dataset = dataset
        self.emb_model = emb_model
        self.llm = llm
        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_episodes = n_episodes
        
        self.agent = SelectorBandit(n_arms=6, state_dim=10, C=C, alpha=alpha, beta=beta, gamma=gamma)
        self.selectors = self.build_selectors()
        
        self.cached_features = None

    def build_selectors(self) -> dict:
        """Instantiates all 6 selector methods to be used as arms."""
        from src.selectors.topk_selector import TopKSelector
        from src.selectors.keyword_selector import KeywordSelector
        from src.selectors.sliding_window import SlidingWindowSelector
        from src.evaluation.evaluator import TruncatedSelector

        return {
            "topk_semantic": TopKSelector(self.emb_model, k=3, alpha=1.0),
            "topk_hybrid": TopKSelector(self.emb_model, k=3, alpha=0.5),
            "keyword": KeywordSelector(num_keywords=10),
            "sliding_window": SlidingWindowSelector(self.emb_model, window_size=3, stride=1, top_n=2),
            "truncated_head": TruncatedSelector(mode='head', max_chunks=3),
            "truncated_tail": TruncatedSelector(mode='tail', max_chunks=3)
        }

    def compute_reward_for_arm(self, arm_idx: int, dataset_subset=None) -> tuple:
        """Evaluates a single selector method on the dataset computing its reward."""
        if dataset_subset is None:
            dataset_subset = self.dataset
            
        arm_name = self.agent.ARM_NAMES[arm_idx]
        
        # Check cache if available via streamlit session state
        try:
            import streamlit as st
            if 'eval_cache' in st.session_state:
                dataset_repr = str([(d.get('story')[:50], len(d.get('qa_pairs', []))) for d in dataset_subset])
                cache_key = f"{arm_name}_{hash(dataset_repr)}"
                if cache_key in st.session_state['eval_cache']:
                    cached_reward, cached_details = st.session_state['eval_cache'][cache_key]
                    
                    quality_score = cached_details['quality_score']
                    compression_ratio = cached_details['compression_ratio']
                    consistency_bonus = cached_details['consistency_bonus']
                    
                    new_reward = (self.agent.alpha * quality_score 
                              - self.agent.beta * compression_ratio 
                              + self.agent.gamma * consistency_bonus)
                    
                    cached_details['reward'] = float(new_reward)
                    return new_reward, cached_details
        except ImportError:
            st = None
            cache_key = None
            
        selector = self.selectors[arm_name]
        
        all_matches = []
        all_tokens = []
        full_context_tokens = []
        sample_answers = []
        
        for item in dataset_subset:
            story = item.get("story", "")
            chunks = chunk_by_tokens(story, self.chunk_size, self.overlap)
            
            f_tokens = len(self.llm.tokenizer.encode(story)) if hasattr(self.llm, 'tokenizer') else len(story.split()) * 1.3
            full_context_tokens.append(f_tokens)
            
            qa_pairs = item.get("qa_pairs", [])
            for qa in qa_pairs:
                q = qa.get("question", "")
                gold = qa.get("answer", "")
                
                try:
                    out = selector.select(chunks, q, getattr(self.llm, 'tokenizer', None))
                    if isinstance(out, (list, tuple)):
                        selected_ctx = out[0]
                        tokens_used = out[1] if len(out) >= 2 else len(selected_ctx.split()) * 1.3
                    else:
                        selected_ctx = out
                        tokens_used = len(selected_ctx.split()) * 1.3
                except Exception as e:
                    print(f"Error in selector {arm_name}: {e}")
                    selected_ctx = " ".join(chunks[:3]) if chunks else ""
                    tokens_used = len(selected_ctx.split()) * 1.3
                
                all_tokens.append(tokens_used)
                
                prompt = f"Context: {selected_ctx}\n\nQuestion: {q}\nAnswer:"
                
                try:
                    pred = self.llm.generate(prompt)
                except Exception:
                    pred = ""
                    
                match = substring_match(gold, pred)
                all_matches.append(match)
                
                if len(sample_answers) < 3:
                    sample_answers.append({
                        "question": q,
                        "gold": gold,
                        "predicted": pred,
                        "match": match
                    })
                    
        quality_score = np.mean(all_matches) if all_matches else 0.0
        
        mean_used = np.mean(all_tokens) if all_tokens else 1.0
        mean_full = np.mean(full_context_tokens) if full_context_tokens else 1.0
        compression_ratio = mean_used / mean_full if mean_full > 0 else 1.0
        
        consistency_bonus = 1.0 - np.std(all_matches) if all_matches else 0.0
        
        reward = (self.agent.alpha * quality_score 
                  - self.agent.beta * compression_ratio 
                  + self.agent.gamma * consistency_bonus)
                  
        details = {
            "arm_name": arm_name,
            "quality_score": float(quality_score),
            "compression_ratio": float(compression_ratio),
            "consistency_bonus": float(consistency_bonus),
            "reward": float(reward),
            "avg_tokens": float(mean_used),
            "per_story_scores": [float(m) for m in all_matches],
            "sample_answers": sample_answers
        }
        
        try:
            import streamlit as st
            if 'eval_cache' in st.session_state and cache_key is not None:
                # Add to cache to avoid refetching
                st.session_state['eval_cache'][cache_key] = (float(reward), details)
        except (ImportError, NameError):
            pass
        
        return reward, details

    def train(self, progress_callback=None) -> dict:
        """Main training loop."""
        import time
        if self.cached_features is None:
            self.cached_features = compute_dataset_features(self.dataset, self.emb_model, self.token_budget)
            
        all_arm_details = {}
            
        for ep in range(1, self.n_episodes + 1):
            arm_idx = self.agent.select_arm(self.cached_features)
            arm_name = self.agent.ARM_NAMES[arm_idx]
            
            reward, details = self.compute_reward_for_arm(arm_idx, self.dataset)
            self.agent.update(arm_idx, reward, self.cached_features)
            
            all_arm_details[arm_name] = details
            
            if progress_callback:
                progress_callback(ep, arm_name, reward, details)
                time.sleep(0)  # yield
                
        best_arm_idx = int(np.argmax(self.agent.q_values))
        best_arm = self.agent.ARM_NAMES[best_arm_idx]
        best_reward = float(self.agent.q_values[best_arm_idx])
        
        q_sum = max(1e-8, np.sum(np.maximum(0, self.agent.q_values)))
        confidence = float(np.clip(max(0, best_reward) / q_sum, 0, 1)) if q_sum > 1e-8 else 0.0
        
        results = {
            "best_arm": best_arm,
            "best_arm_idx": best_arm_idx,
            "best_reward": best_reward,
            "q_values": {self.agent.ARM_NAMES[i]: float(self.agent.q_values[i]) for i in range(self.agent.n_arms)},
            "counts": {self.agent.ARM_NAMES[i]: int(self.agent.counts[i]) for i in range(self.agent.n_arms)},
            "reward_history": self.agent.reward_history,
            "best_arm_history": self.agent.best_arm_history,
            "feature_vector": self.cached_features.tolist(),
            "feature_names": [
                "avg_story_length", "avg_question_length", "vocab_richness",
                "avg_answer_length", "keyword_density", "avg_chunk_count",
                "answer_position_bias", "semantic_spread", "question_type_ratio",
                "compression_pressure"
            ],
            "all_arm_details": all_arm_details
        }
        
        results["recommendation"] = self.get_recommendation_text(results)
        results["confidence"] = confidence
        
        return results

    def get_recommendation_text(self, results: dict) -> str:
        """Generates a plain English explanation of the agent's recommendation."""
        best = results["best_arm"]
        q_vals = results["q_values"]
        sorted_arms = sorted(q_vals.keys(), key=lambda x: q_vals[x], reverse=True)
        
        gap_info = ""
        if len(sorted_arms) > 1:
            best_q = q_vals[sorted_arms[0]]
            second_q = q_vals[sorted_arms[1]]
            gap = best_q - second_q
            gap_info = f" The margin over the second-best selector ({sorted_arms[1]}) was {gap:.3f} Q-value points."
        
        text = (f"The recommended selector is **{best}**. Throughout training, this method provided the best "
                f"balance of answer quality, context compression, and consistency across stories.{gap_info} "
                "For practical deployment, consider adopting this selector as your default for datasets with similar characteristics.")
        return text
