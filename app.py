import os
import re
import subprocess
import sys
from datetime import datetime
import numpy as np

import pandas as pd
import streamlit as st
import torch
from peft import PeftModel

from utils import (BASE_DIR, DATA_DIR, LOG_DIR, MODEL_ID, OUTPUT_DIR, USE_4BIT, build_prompt,
                   load_base_model, load_tokenizer)

TRAIN_FILE = DATA_DIR / "train.jsonl"
SAMPLE_FILE = DATA_DIR / "sample.jsonl"
LOG_FILE = LOG_DIR / "train.log"

ALIASES = {
    "instruction": ["instruction", "prompt", "question"],
    "input": ["input", "context"],
    "output": ["output", "response", "answer"],
}

st.set_page_config(page_title="DeepSeek-R1 LoRA Studio", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- helpers
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for key, names in ALIASES.items():
        col = next((c for c in names if c in df.columns), None)
        if col is None:
            if key == "input":
                out[key] = ""
                continue
            raise ValueError(f"Missing a column for '{key}'. Accepted names: {names}")
        out[key] = df[col].fillna("").astype(str)
    return out[out["instruction"].str.strip().ne("") & out["output"].str.strip().ne("")]


def read_upload(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_json(f, lines=name.endswith(".jsonl"))


def save_train(df: pd.DataFrame):
    df.to_json(TRAIN_FILE, orient="records", lines=True, force_ascii=False)


@st.cache_resource(show_spinner="Loading model (first time takes a while)...")
def get_model(adapter_path: str):
    tok = load_tokenizer()
    model = load_base_model()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def generate(model, tok, prompt, max_new_tokens, temperature, top_p) -> str:
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def show_answer(text: str):
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        if think.strip():
            with st.expander("Reasoning"):
                st.write(think.strip())
        st.write(answer.strip())
    else:
        st.write(text.strip())


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Environment")
    st.caption(f"Base model\n\n`{MODEL_ID}`")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        st.success(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        st.warning("No CUDA GPU found - training on CPU will be very slow.")

st.title("🧠 DeepSeek-R1-Distill-Qwen-1.5B · LoRA Studio")
tab_data, tab_train, tab_test = st.tabs(["1 · Data", "2 · Train", "3 · Test"])

# ---------------------------------------------------------------- data
with tab_data:
    st.caption("Columns: `instruction` (or prompt/question), optional `input`, `output` (or response/answer).")
    up = st.file_uploader("Upload dataset", type=["jsonl", "json", "csv"])
    c1, c2, _ = st.columns([1, 1, 3])
    if up is not None and c1.button("Use uploaded file"):
        try:
            save_train(normalize(read_upload(up)))
            st.success("Saved as training set.")
        except Exception as e:
            st.error(str(e))
    if SAMPLE_FILE.exists() and c2.button("Load sample data"):
        save_train(normalize(pd.read_json(SAMPLE_FILE, lines=True)))
        st.success("Sample data loaded.")
    if TRAIN_FILE.exists():
        df = pd.read_json(TRAIN_FILE, lines=True)
        st.write(f"**{len(df)} training examples**")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("No training data yet - upload a file or load the sample.")


# ---------------------------------------------------------------- train
def training_monitor():
    proc = st.session_state.get("proc")
    running = proc is not None and proc.poll() is None
    if not running and st.session_state.get("train_active"):
        st.session_state.train_active = False
        st.rerun()  # refresh the whole page once training ends

    text = LOG_FILE.read_text(encoding="utf-8", errors="ignore") if LOG_FILE.exists() else ""
    if running:
        st.info("Training in progress...")
    elif "TRAINING_COMPLETE" in text:
        st.success("Training finished - adapter saved. Go to the Test tab.")
    elif proc is not None and proc.returncode not in (0, None):
        st.error(f"Training exited with code {proc.returncode}. Check the log below.")

    losses = [float(x) for x in re.findall(r"'loss': ([0-9.]+)", text)]
    if losses:
        st.line_chart(pd.DataFrame({"loss": losses}))
    st.code("\n".join(text.splitlines()[-25:]) or "Log is empty.", language="text")


with tab_train:
    proc = st.session_state.get("proc")
    running = proc is not None and proc.poll() is None

    r1 = st.columns(3)
    epochs = r1[0].number_input("Epochs", 1, 50, 3)
    lr = r1[1].select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=2e-4, format_func=lambda x: f"{x:.0e}")
    max_len = r1[2].select_slider("Max sequence length", [256, 512, 1024, 2048], value=512)
    r2 = st.columns(3)
    rank = r2[0].select_slider("LoRA rank (r)", [4, 8, 16, 32, 64], value=16)
    alpha = r2[1].select_slider("LoRA alpha", [8, 16, 32, 64, 128], value=32)
    dropout = r2[2].slider("LoRA dropout", 0.0, 0.3, 0.05, 0.01)
    r3 = st.columns(3)
    batch = r3[0].number_input("Batch size", 1, 32, 2)
    accum = r3[1].number_input("Gradient accumulation", 1, 64, 8)
    run_name = r3[2].text_input("Run name", f"lora-{datetime.now():%Y%m%d-%H%M}")
    use_4bit = st.checkbox("4-bit QLoRA (NVIDIA GPU + bitsandbytes; saves VRAM)", value=USE_4BIT)

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("🚀 Start training", type="primary", disabled=running or not TRAIN_FILE.exists()):
        cmd = [sys.executable, "train.py", "--data", str(TRAIN_FILE), "--run_name", run_name,
               "--epochs", str(epochs), "--lr", str(lr), "--rank", str(rank), "--alpha", str(alpha),
               "--dropout", str(dropout), "--batch_size", str(batch), "--grad_accum", str(accum),
               "--max_len", str(max_len)]
        if use_4bit:
            cmd.append("--use_4bit")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        log = open(LOG_FILE, "w", encoding="utf-8")
        st.session_state.proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        st.session_state.train_active = True
        st.rerun()
    if b2.button("⏹ Stop", disabled=not running):
        proc.terminate()
        st.rerun()

    st.fragment(run_every=3 if running else None)(training_monitor)()

# ---------------------------------------------------------------- test
with tab_test:
    if running:
        st.warning("Training is using the GPU - testing now may run out of memory.")
    adapters = sorted(p.name for p in OUTPUT_DIR.iterdir() if (p / "adapter_config.json").exists())
    choice = st.selectbox("Adapter", ["(base model only)"] + adapters, index=len(adapters) if adapters else 0)
    adapter_path = "" if choice.startswith("(") else str(OUTPUT_DIR / choice)

    prompt_text = st.text_area("Prompt", "Explain what LoRA fine-tuning is in two sentences.", height=120)
    g = st.columns(4)
    max_new = g[0].number_input("Max new tokens", 32, 2048, 256, step=32)
    temp = g[1].slider("Temperature", 0.0, 1.5, 0.6, 0.05)
    top_p = g[2].slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    compare = g[3].checkbox("Compare with base", value=bool(adapter_path), disabled=not adapter_path)

    if st.button("Generate", type="primary", disabled=not prompt_text.strip()):
        tok, model = get_model(adapter_path)
        prompt = build_prompt(tok, prompt_text.strip())
        with st.spinner("Generating..."):
            tuned = generate(model, tok, prompt, max_new, temp, top_p)
            base = None
            if compare and isinstance(model, PeftModel):
                with model.disable_adapter():
                    base = generate(model, tok, prompt, max_new, temp, top_p)
        if base is None:
            show_answer(tuned)
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Fine-tuned")
                show_answer(tuned)
            with right:
                st.subheader("Base")
                show_answer(base)