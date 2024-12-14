import re

import numpy as np
import pandas as pd
import vllm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from transformers import AutoTokenizer

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_text(x):
    x = re.sub("http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


PROMPT = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
Pick the correct misconception number from the below:

{Retrival}
"""


def apply_template(row, tokenizer):
    messages = [
        {
            "role": "user",
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row["incorrect_answer"],
                    CorrectAnswer=row["correct_answer"],
                    Retrival=row["retrieval"],
                )
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


misconception_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

df = pd.read_parquet("df_target.parquet")
indices = np.stack(df["MisconceptionId"].apply(lambda x: np.array(list(map(int, x.split())))))

llm = vllm.LLM(
    model_path,
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True,
)
tokenizer = llm.get_tokenizer()


def get_candidates(c_indices):
    candidates = []

    mis_names = misconception_df["MisconceptionName"].values
    for ix in c_indices:
        c_names = []
        for i, name in enumerate(mis_names[ix]):
            c_names.append(f"{i+1}. {name}")

        candidates.append("\n".join(c_names))

    return candidates


survivors = indices[:, -1:]

for i in range(3):
    c_indices = np.concatenate([indices[:, -8 * (i + 1) - 1 : -8 * i - 1], survivors], axis=1)

    df["retrieval"] = get_candidates(c_indices)
    df["text"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)

    responses = llm.generate(
        df["text"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777,  # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=1,  # Maximum number of tokens to generate per output sequence.
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])],
        ),
        use_tqdm=True,
    )

    responses = [x.outputs[0].text for x in responses]
    df["response"] = responses

    llm_choices = df["response"].astype(int).values - 1

    survivors = np.array([cix[best] for best, cix in zip(llm_choices, c_indices, strict=False)]).reshape(-1, 1)
    df[f"s{i}"] = survivors


def create_reranker_result(row):
    originals = row.MisconceptionId.split()
    rerank_result = [str(row.s2)] + originals[:8] + [str(row.s1)] + originals[8:16] + [str(row.s0)] + originals[16:]
    rerank_result = list(dict.fromkeys(rerank_result))[:25]
    return " ".join(rerank_result)


df["reranker_results"] = df.apply(create_reranker_result, axis=1)

###########################
# 2,3位もLLMに抽出させる
###########################


def extract_candidates(row, target_rank=2):
    target_ids = list(map(int, row.reranker_results.split()))[1:]
    if target_rank == 2:
        target_ids = target_ids[:9]
    if target_rank == 3:
        target_ids = [id for id in target_ids if id != row.f2][:9]
    return target_ids


for i in range(2):
    target_rank = i + 2
    df["candidates"] = df.apply(lambda row: extract_candidates(row, target_rank=target_rank), axis=1)
    print("candidates ", df["candidates"].values)

    df["retrieval"] = get_candidates(df["candidates"].values)
    df["text"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)
    responses = llm.generate(
        df["text"].values,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=777,  # Seed for reprodicibility
            skip_special_tokens=False,  # Whether to skip special tokens in the output.
            max_tokens=1,  # Maximum number of tokens to generate per output sequence.
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])],
        ),
        use_tqdm=True,
    )

    responses = [x.outputs[0].text for x in responses]
    df["response"] = responses

    llm_choices = df["response"].astype(int).values - 1

    survivors = np.array([cix[best] for best, cix in zip(llm_choices, df["candidates"].values, strict=False)]).reshape(-1, 1)
    df[f"f{target_rank}"] = survivors


def create_reranker_result_v2(row):
    originals = row.reranker_results.split()
    rerank_result = [originals[0]] + [str(row.f2), str(row.f3)] + originals[1:]
    rerank_result = list(dict.fromkeys(rerank_result))[:25]
    return " ".join(rerank_result)


df["reranker_results_v2"] = df.apply(create_reranker_result_v2, axis=1)

df.to_parquet("df_target.parquet", index=False)
df_sub = df[["QuestionId_Answer", "reranker_results_v2"]].copy()
df_sub.columns = ["QuestionId_Answer", "MisconceptionId"]
df_sub.to_csv("submission.csv", index=False)
