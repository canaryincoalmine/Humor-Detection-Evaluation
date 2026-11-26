import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TASK_INSTRUCTIONS = {
    "sarcasm": "Is the following forum post sarcastic?",
    "humor": "Is the following text meant to be humorous?",
}


# Data loading
def load_sarcasm(sarcasm_dir, subsets=("GEN", "HYP", "RQ")):
    """Loads the Sarcasm Corpus V2 CSVs (columns: Corpus, Label, ID, Text)."""
    files = glob.glob(os.path.join(sarcasm_dir, "**", "*.csv"), recursive=True)
    frames = []
    for f in files:
        name = os.path.basename(f).upper()
        if not any(name.startswith(s) for s in subsets):
            continue
        df = pd.read_csv(f, encoding="utf-8-sig", encoding_errors="replace")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={"class": "label", "response": "text", "post": "text"})
        if not {"text", "label"} <= set(df.columns):
            raise KeyError(
                f"{f}: expected 'text' and 'label' columns, found {list(df.columns)}. "
                f"First row: {df.iloc[0].to_dict() if len(df) else 'empty'}"
            )
        df["source"] = name.split("-")[0]
        frames.append(df[["text", "label", "source"]])
    if not frames:
        raise FileNotFoundError(
            f"No matching CSVs found under {sarcasm_dir}. "
            "Check --sarcasm_dir and that the files start with GEN/HYP/RQ."
        )
    df = pd.concat(frames, ignore_index=True)
    # HYP/RQ posts partly overlap with GEN, so drop duplicate texts
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = (df["label"].astype(str).str.strip().str.lower() == "sarc").astype(int)
    return df[["text", "label"]]


def load_humor(max_rows=None, seed=42):
    from datasets import load_dataset

    ds = load_dataset("CreativeLang/ColBERT_Humor_Detection", split="train")
    df = ds.to_pandas()[["text", "humor"]].rename(columns={"humor": "label"})
    df["label"] = df["label"].astype(int)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df


def split(df, seed=42, test_size=0.2):
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


# Metrics
def compute_metrics(y_true, y_pred, scores=None):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": int(len(y_true)),
    }
    if scores is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, scores)
    return out


# Logistic regression
def train_lr(train_df, seed=42):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )
    model.fit(train_df["text"], train_df["label"])
    return model


def eval_lr(model, df):
    proba = model.predict_proba(df["text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return compute_metrics(df["label"].values, pred, proba)


# Llama 3.1 8B zero-shot (Yes/No logit scoring)
class LlamaClassifier:
    def __init__(self, model_name=LLM_NAME, load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"  # needed for last-token logits with batching
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

        self.yes_id = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("No", add_special_tokens=False)[0]

    def _prompt(self, instruction, text):
        messages = [
            {
                "role": "system",
                "content": "You are a careful text classifier. Answer with exactly one word: Yes or No.",
            },
            {"role": "user", "content": f"{instruction}\n\nText: {text[:2000]}\n\nAnswer Yes or No."},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @property
    def _device(self):
        return next(self.model.parameters()).device

    def score(self, texts, instruction, batch_size=16):
        """Returns logit(Yes) - logit(No) per text. >0 means predicted positive."""
        torch = self.torch
        scores = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Llama"):
            batch = [self._prompt(instruction, t) for t in texts[i : i + batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            scores.extend((logits[:, self.yes_id] - logits[:, self.no_id]).cpu().tolist())
        return np.array(scores)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcasm_dir", required=True, help="Path to cloned soraby/sarcasm2 repo")
    ap.add_argument("--sarcasm_subsets", nargs="+", default=["GEN", "HYP", "RQ"])
    ap.add_argument("--humor_max_rows", type=int, default=None, help="Subsample humor data (200k total)")
    ap.add_argument("--max_llm_samples", type=int, default=1000, help="Test examples per dataset for Llama")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--skip_llm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = {
        "sarcasm": load_sarcasm(args.sarcasm_dir, tuple(args.sarcasm_subsets)),
        "humor": load_humor(args.humor_max_rows, args.seed),
    }

    splits, lr_models, rows = {}, {}, []
    for name, df in datasets.items():
        train_df, test_df = split(df, args.seed)
        splits[name] = (train_df, test_df)
        print(f"{name}: {len(df)} rows | train {len(train_df)} | test {len(test_df)} | "
              f"pos rate {df['label'].mean():.2f}")

        lr_models[name] = train_lr(train_df, args.seed)
        rows.append({"dataset": name, "model": "LogReg (full test)", **eval_lr(lr_models[name], test_df)})

    # Cross-dataset transfer for LR (train on A, test on B). Labels mean different
    # things (sarcastic vs. humorous), so treat this as a rough domain-shift probe.
    for a in datasets:
        for b in datasets:
            if a != b:
                rows.append({"dataset": f"{a}->{b}", "model": "LogReg (transfer)",
                             **eval_lr(lr_models[a], splits[b][1])})

    if not args.skip_llm:
        llm = LlamaClassifier(load_in_4bit=args.load_in_4bit)
        for name, (_, test_df) in splits.items():
            sub = test_df
            if len(sub) > args.max_llm_samples:
                sub, _ = train_test_split(
                    sub, train_size=args.max_llm_samples, stratify=sub["label"], random_state=args.seed
                )
            sub = sub.reset_index(drop=True)

            # LR on the same subset
            rows.append({"dataset": name, "model": "LogReg (LLM subset)", **eval_lr(lr_models[name], sub)})

            scores = llm.score(sub["text"].tolist(), TASK_INSTRUCTIONS[name], args.batch_size)
            preds = (scores > 0).astype(int)
            rows.append({"dataset": name, "model": "Llama-3.1-8B zero-shot",
                         **compute_metrics(sub["label"].values, preds, scores)})

            pd.DataFrame({"text": sub["text"], "label": sub["label"], "llm_score": scores,
                          "llm_pred": preds}).to_csv(
                os.path.join(args.out_dir, f"llama_preds_{name}.csv"), index=False)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)

    pd.set_option("display.width", 200)
    print("\n", results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()