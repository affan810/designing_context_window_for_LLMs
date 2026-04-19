import sys
import os
import streamlit as st
import yaml
import json
import pandas as pd
import plotly.express as px
import glob
import shutil

# Ensure relative imports resolve correctly by inserting the project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Import custom modules
from src.models.tinyllama import TinyLlamaModel
from src.models.embeddings import EmbeddingModel
from src.selectors.topk_selector import TopKSelector
from src.selectors.keyword_selector import KeywordSelector
from src.selectors.sliding_window import SlidingWindowSelector
from src.evaluation.evaluator import FullContextSelector, TruncatedSelector
from src.evaluation.metrics import substring_match, token_f1
from src.utils.chunking import chunk_by_tokens

# ----------------- CONFIG & GLOBALS -----------------
# Load hyperparameters
@st.cache_data
def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

cfg = load_config()

# Story presets mapped directly from user's requirements
STORY_PRESETS = {
    "Marie Curie": "Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and the only woman to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields. Her achievements include the development of the theory of radioactivity, techniques for isolating radioactive isotopes, and the discovery of two elements, polonium and radium. Under her direction, the world's first studies were conducted into the treatment of neoplasms using radioactive isotopes. She founded the Curie Institutes in Paris and in Warsaw, which remain major centres of medical research today.",
    "Amazon Rainforest": "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela. Nine nations have the Amazon rainforest in their borders.",
    "Internet History": "The history of the Internet has its origin in the efforts of scientists and engineers to build and interconnect computer networks. The Internet Protocol Suite, the set of rules used to communicate between networks and devices on the Internet, arose from research and development in the United States and involved international collaboration, particularly with researchers in the United Kingdom and France. The ARPANET, which was developed by ARPA of the United States Department of Defense, was the first network to implement the protocol suite.",
    "Shakespeare": "William Shakespeare was an English playwright, poet, and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the \"Bard of Avon\" (or simply \"the Bard\"). His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems, and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright.",
    "Climate Change": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.",
    "Human Brain": "The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem and the cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating the information it receives from the sense organs, and making decisions as to the instructions sent to the rest of the body. The brain is contained in, and protected by, the skull bones of the head.",
    "Python Language": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a \"batteries included\" language due to its comprehensive standard library."
}

DEFAULT_QA = {
    "Marie Curie": ("What elements did Marie Curie discover?", "polonium and radium"),
    "Amazon Rainforest": ("What happened to the Amazon in 2019?", "widespread fires"),
    "Internet History": ("Which network first implemented the protocol suite?", "ARPANET"),
    "Shakespeare": ("How many plays did Shakespeare write?", "39 plays"),
    "Climate Change": ("What is the main driver of climate change since the 1800s?", "human activities"),
    "Human Brain": ("What parts make up the central nervous system with the brain?", "spinal cord"),
    "Python Language": ("Why is Python called a batteries included language?", "comprehensive standard library")
}

# ----------------- SESSION STATE -----------------
if 'last_run_results' not in st.session_state:
    st.session_state.last_run_results = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []
if 'training_rewards' not in st.session_state:
    st.session_state.training_rewards = []
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = "Marie Curie"

# ----------------- CACHED MODELS -----------------
@st.cache_resource(show_spinner=False)
def load_models():
    # Cache resource so large models aren't recreated on each Streamlit rerun
    st.spinner("Loading TinyLlama and embedding model...")
    emb = EmbeddingModel(model_id=cfg['embeddings']['model_id'], 
                         cache_dir=cfg['embeddings']['cache_dir'])
    llm = TinyLlamaModel(model_id=cfg['model']['tinyllama_model_id'],
                         max_new_tokens=cfg['model']['max_new_tokens'],
                         use_fp16=cfg['model']['use_fp16'])
    return emb, llm

# Load them (lazy loaded when we first reach this block)
try:
    with st.spinner("Loading TinyLlama and embedding model (~30s on first run)..."):
        emb_model, llm_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# ----------------- UI UTILS -----------------
def render_chunk_pills(chunks, selected_indices):
    sel_set = set(selected_indices)
    html = ""
    for i, chunk in enumerate(chunks):
        preview = str(chunk)[:30].replace('"', '').replace("'", "")
        if i in sel_set:
            html += f'<span style="background:#E1F5EE;color:#085041;padding:3px 10px; border-radius:16px;margin:3px;display:inline-block;font-size:12px;font-weight:500" title="{preview}...">C{i}</span>'
        else:
            html += f'<span style="background:#FAECE7;color:#712B13;padding:3px 10px; border-radius:16px;margin:3px;display:inline-block;font-size:12px;text-decoration:line-through;opacity:0.6" title="{preview}...">C{i}</span>'
    return html

def get_selector(method_name, emb):
    if method_name == "Top-K (semantic)":
        return TopKSelector(emb, k=cfg['selectors']['topk']['k'], alpha=1.0)
    elif method_name == "Top-K (hybrid)":
        return TopKSelector(emb, k=cfg['selectors']['topk']['k'], alpha=0.5)
    elif method_name == "Keyword (TF-IDF)":
        return KeywordSelector(num_keywords=cfg['selectors']['keyword']['num_keywords'])
    elif method_name == "Sliding Window":
        return SlidingWindowSelector(emb, 
                                     window_size=cfg['selectors']['sliding_window']['window_size'],
                                     stride=cfg['selectors']['sliding_window']['stride'],
                                     top_n=cfg['selectors']['sliding_window']['top_n'])
    elif method_name == "Truncated Head":
        return TruncatedSelector(mode='head', max_chunks=3)
    elif method_name == "Truncated Tail":
        return TruncatedSelector(mode='tail', max_chunks=3)
    else:
        return FullContextSelector()

# ----------------- APP LAYOUT -----------------
st.set_page_config(page_title="Context Window Optimization", layout="wide")

# SIDEBAR
with st.sidebar:
    st.title("NLP Research: Context Window Optimization")
    st.markdown("Exploring efficiency vs accuracy trade-offs when compressing context for LLMs.")
    
    st.markdown("### Test Datasets")
    selected_preset = st.selectbox("Dataset preset", list(STORY_PRESETS.keys()))
    if selected_preset != st.session_state.active_dataset:
        st.session_state.active_dataset = selected_preset
        
    st.markdown("### Chunking Parameters")
    chunk_size = st.slider("Chunk size (tokens)", 20, 200, cfg['chunking'].get('chunk_size', 50))
    chunk_overlap = st.slider("Overlap (tokens)", 0, 50, cfg['chunking'].get('overlap', 10))
    cfg['chunking']['chunk_size'] = chunk_size
    cfg['chunking']['overlap'] = chunk_overlap
    
    st.divider()
    st.markdown("### Cache Management")
    cache_dir = cfg['embeddings']['cache_dir']
    num_cached = len(glob.glob(os.path.join(cache_dir, "*"))) if os.path.exists(cache_dir) else 0
    st.metric("Embedding cache size (files)", num_cached)
    if st.button("Clear cache"):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            st.success("Cache cleared!")
            st.rerun()

st.title("Context Compression for TinyLlama")

tab1, tab2, tab3 = st.tabs(["⚡ Live Demo", "📊 Method Comparison", "🧠 RL Training Monitor"])

# ================= TAB 1: LIVE DEMO =================
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Input Context & Query")
        story_text = st.text_area("Paste your context / story here", 
                                  value=STORY_PRESETS[st.session_state.active_dataset], 
                                  height=300)
        
        col_budget, col_method = st.columns(2)
        with col_budget:
            token_budget = st.slider("Token budget", 50, 800, 200, 10)
        with col_method:
            method_name = st.selectbox("Selector method", [
                "Top-K (semantic)", "Top-K (hybrid)", "Keyword (TF-IDF)", 
                "Sliding Window", "Truncated Head", "Truncated Tail", "Full Context"
            ])
            
        question_text = st.text_input("Question", value=DEFAULT_QA[st.session_state.active_dataset][0])
        gold_answer_text = st.text_input("Gold Answer (for scoring)", value=DEFAULT_QA[st.session_state.active_dataset][1])
        
        run_btn = st.button("Run Compression + Answer", type="primary")
        
        # Token Live Counter (simulated by full words * 1.3 approx or actual tokenizer)
        try:
            live_tokens = len(llm_model.tokenizer.encode(story_text))
            st.metric("Live Content Size", f"{live_tokens} tokens", 
                      delta=f"{(token_budget - live_tokens)} budget diff",
                      delta_color="normal" if live_tokens <= token_budget else "inverse")
        except:
            pass

    with col2:
        if run_btn:
            if not question_text:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Processing through the pipeline..."):
                    try:
                        # 1. Chunking
                        chunks = chunk_by_tokens(story_text, chunk_size, chunk_overlap)
                        st.session_state.demo_chunks = chunks
                        
                        # 2. Setup Full Context Oracle
                        full_sel = FullContextSelector()
                        f_ctx, f_tok = full_sel.select(chunks, question_text, llm_model.tokenizer)
                        
                        # 3. Setup Compressed Context
                        comp_sel = get_selector(method_name, emb_model)
                        c_ctx, c_tok, c_indices = [], 0, []
                        
                        # Use internal method depending on selector return type
                        if method_name == "Full Context":
                            c_ctx, c_tok = f_ctx, f_tok
                            c_indices = list(range(len(chunks)))
                        else:
                            # Handling varying return signatures robustly
                            comp_res = comp_sel.select(chunks, question_text, llm_model.tokenizer)
                            if len(comp_res) == 2:
                                c_ctx, c_tok = comp_res
                                # Guess indices based on presence
                                c_indices = [i for i, c in enumerate(chunks) if c in c_ctx]
                            else:
                                c_ctx, c_tok, c_indices = comp_res
                        
                        # Apply naive sliding token budget constraint
                        while c_tok > token_budget and len(c_ctx) > 0:
                            c_ctx.pop(-1)
                            if c_indices:
                                c_indices.pop(-1)
                            c_tok = len(llm_model.tokenizer.encode(" ".join(c_ctx)))
                        
                        # 4. Generate Answers
                        ans_full = llm_model.answer(f_ctx, question_text)
                        ans_comp = llm_model.answer(c_ctx, question_text)
                        
                        # 5. Metrics
                        sub_match = substring_match(ans_comp, gold_answer_text)
                        f1_score = token_f1(ans_comp, ans_full)
                        comp_ratio = (c_tok / f_tok * 100) if f_tok > 0 else 0
                        
                        st.session_state.last_run_results = {
                            "chunks": chunks, "c_indices": c_indices,
                            "f_tok": f_tok, "c_tok": c_tok, "comp_ratio": comp_ratio,
                            "ans_full": ans_full, "ans_comp": ans_comp,
                            "sub_match": sub_match, "f1_score": f1_score,
                            "method": method_name
                        }
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")
        
        # Display Results
        res = st.session_state.last_run_results
        if res:
            st.subheader("Section A: Compression Stats")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original tokens", res['f_tok'])
            c2.metric("Compressed tokens", res['c_tok'])
            ratio_color = "normal" if res['comp_ratio'] < 90 else "off"
            c3.metric("Compression ratio", f"{res['comp_ratio']:.1f}%", 
                      delta=f"-{100 - res['comp_ratio']:.1f}% size", 
                      delta_color="inverse" if res['comp_ratio'] == 100 else "normal")
            
            st.divider()
            st.subheader("Section B: Chunk Visualizer")
            st.markdown(render_chunk_pills(res['chunks'], res['c_indices']), unsafe_allow_html=True)
            with st.expander("Show chunk details"):
                chunk_data = []
                for i, c in enumerate(res['chunks']):
                    chunk_data.append({
                        "ID": f"C{i}", 
                        "Preview": str(c)[:50] + "...", 
                        "Selected": (i in res['c_indices']),
                        "Tokens": len(llm_model.tokenizer.encode(str(c)))
                    })
                st.dataframe(pd.DataFrame(chunk_data), hide_index=True)
                
            st.divider()
            st.subheader("Section C: Side-by-side Answer")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown("**Full Context Oracle**")
                st.info(res['ans_full'])
                st.caption(f"Used {res['f_tok']} setup tokens")
            with ac2:
                st.markdown(f"**{res['method']}**")
                st.success(res['ans_comp'])
                st.caption(f"Used {res['c_tok']} tokens | Saved {res['f_tok'] - res['c_tok']}!")
                
            st.divider()
            st.subheader("Section D: Quality Metrics")
            qc1, qc2, qc3 = st.columns(3)
            qc1.metric("Substring Match (Gold)", f"{res['sub_match']:.2f}")
            qc2.metric("Token F1 (vs Full)", f"{res['f1_score']:.2f}")
            
            match_status = "PRESERVED" if res['sub_match'] > 0 else "DEGRADED"
            st.markdown(f"**Match Quality:** <span style='color:{'green' if match_status == 'PRESERVED' else 'red'}'>{match_status}</span>", unsafe_allow_html=True)

            st.info(f"Insight: The **{res['method']}** method retained {len(res['c_indices'])} out of {len(res['chunks'])} chunks. It saved {res['f_tok'] - res['c_tok']} tokens by prioritizing chunks according to its selection strategy. By analyzing the question and scoring relevance, we achieve a smaller token footprint for faster, cheaper inference.")

# ================= TAB 2: METHOD COMPARISON =================
with tab2:
    st.markdown("Run all implemented selectors on the same context and question to compare trade-offs.")
    
    comp_story = st.text_area("Story for Comparison", value=STORY_PRESETS[st.session_state.active_dataset], height=150)
    cc1, cc2 = st.columns(2)
    comp_question = cc1.text_input("Comparison Question", value=DEFAULT_QA[st.session_state.active_dataset][0])
    comp_gold = cc2.text_input("Comparison Gold Answer", value=DEFAULT_QA[st.session_state.active_dataset][1])
    
    if st.button("Compare all methods", type="primary"):
        with st.spinner("Processing all selectors sequentially..."):
            methods = [
                "Full Context", "Top-K (semantic)", "Top-K (hybrid)", 
                "Keyword (TF-IDF)", "Sliding Window", "Truncated Head", "Truncated Tail"
            ]
            
            results_list = []
            comp_chunks = chunk_by_tokens(comp_story, chunk_size, chunk_overlap)
            
            # Baseline oracle first
            full_sel = FullContextSelector()
            f_ctx, f_tok = full_sel.select(comp_chunks, comp_question, llm_model.tokenizer)
            ans_full = llm_model.answer(f_ctx, comp_question)
            
            for m in methods:
                sel = get_selector(m, emb_model)
                out = sel.select(comp_chunks, comp_question, llm_model.tokenizer)
                
                # Standardize returns
                c_ctx = out[0] if isinstance(out, (list, tuple)) else out
                c_tok = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else len(llm_model.tokenizer.encode(" ".join(c_ctx)))
                
                ans_comp = llm_model.answer(c_ctx, comp_question)
                sm_score = substring_match(ans_comp, comp_gold)
                f1_score = token_f1(ans_comp, ans_full)
                
                eff = sm_score / c_tok if c_tok > 0 else 0
                
                results_list.append({
                    "Method": m,
                    "Tokens used": c_tok,
                    "Compression %": (c_tok / f_tok * 100) if f_tok > 0 else 0,
                    "Substring match": sm_score,
                    "Token F1": f1_score,
                    "Efficiency (accuracy/tokens)": eff * 1000, # scaled for readable viz
                    "Answer preview": ans_comp[:60] + "..." if len(ans_comp) > 60 else ans_comp,
                    "_full_answer": ans_comp
                })
                
            st.session_state.comparison_results = results_list

    if st.session_state.comparison_results:
        df = pd.DataFrame(st.session_state.comparison_results)
        df_display = df.drop(columns=["_full_answer"])
        df_display = df_display.sort_values(by="Substring match", ascending=False).reset_index(drop=True)
        
        st.subheader("1. Comprehensive Metrics")
        st.dataframe(df_display.style.highlight_max(subset=['Substring match', 'Token F1', 'Efficiency (accuracy/tokens)'], color='lightgreen'))
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader("2. Accuracy vs Method")
            bar_fig = px.bar(df, x='Method', y='Substring match', color='Efficiency (accuracy/tokens)',
                             color_continuous_scale='Teal', title="Accuracy vs Method (color = efficiency)")
            st.plotly_chart(bar_fig, use_container_width=True)
            
        with rc2:
            st.subheader("3. Accuracy–Token Trade-off")
            scatter_fig = px.scatter(df, x='Tokens used', y='Substring match', text='Method', 
                                     title="Accuracy vs. Token Cost", size_max=60)
            scatter_fig.update_traces(textposition='top center')
            
            # Draw baseline horizontal reference
            baseline_acc = df[df["Method"] == "Full Context"]["Substring match"].values
            if len(baseline_acc) > 0:
                scatter_fig.add_hline(y=baseline_acc[0], line_dash="dash", line_color="gray", annotation_text="Full Context Baseline")
            st.plotly_chart(scatter_fig, use_container_width=True)
            
        with st.expander("Show all answers"):
            for row in st.session_state.comparison_results:
                st.markdown(f"**{row['Method']}** ({row['Tokens used']} tokens): {row['_full_answer']}")


# ================= TAB 3: RL TRAINING MONITOR =================
with tab3:
    st.markdown("Monitor or launch reinforcement learning loops to train the context-selection agent.")
    
    uploaded_rl = st.file_uploader("Upload rl_bandit_results.json or rl_pg_results.json", type=['json'])
    
    if uploaded_rl is not None:
        rl_data = json.load(uploaded_rl)
        st.success("Loaded RL results successfully!")
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Final Accuracy", f"{rl_data.get('final_accuracy', 0):.2f}")
        mc2.metric("Tokens Built", rl_data.get('avg_tokens', 0))
        mc3.metric("Efficiency", f"{rl_data.get('efficiency', 0):.4f}")
        mc4.metric("Episodes", rl_data.get('episodes', len(rl_data.get('reward_history', []))))
        
        if 'reward_history' in rl_data:
            st.line_chart(rl_data['reward_history'])
        else:
            st.info("No detailed reward_history found in JSON. Simulated final metrics shown.")
            
    st.divider()
    with st.expander("Run RL training live (Simulation)"):
        rc1, rc2 = st.columns(2)
        train_eps = rc1.slider("Episodes", 50, 500, 200)
        agent_type = rc2.selectbox("Agent Type", ["Epsilon-greedy bandit", "Policy gradient"])
        train_lambda = rc1.slider("Lambda penalty", 0.0001, 0.0100, 0.0010, step=0.0001)
        
        if st.button("Start training", type="primary"):
            st.warning("Running the simulated visual training loop...")
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            
            # Simple simulation loop to populate UI as requested
            sim_rewards = []
            base_reward = 0.2
            for ep in range(train_eps):
                # Simulated learning curve
                if agent_type == "Policy gradient":
                    inc = (0.8 - base_reward) * (ep / train_eps) ** 2
                else:
                    inc = (0.7 - base_reward) * (ep / train_eps) ** 0.8
                
                noise = (int(ep * 100) % 15) / 100.0 - 0.07
                sim_rewards.append(base_reward + inc + noise)
                
                progress_bar.progress((ep + 1) / train_eps)
                if ep % 5 == 0:
                    chart_placeholder.line_chart(sim_rewards)
            
            st.success("Training finalized!")
            st.metric("Final Simulated Reward", f"{sim_rewards[-1]:.3f}")
            st.session_state.training_rewards = sim_rewards
