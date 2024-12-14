import re

import pandas as pd
import vllm
from transformers import AutoTokenizer

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
# model_path = "/kaggle/input/kirillr-qwq-32b-preview-awq"
tokenizer = AutoTokenizer.from_pretrained(model_path)

misconception_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
mapping_dict = misconception_df.set_index("MisconceptionId")["MisconceptionName"].to_dict()

PROMPT = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
From the options below, please select the three that you consider most appropriate, in order of preference.
Answer by listing the option numbers in a comma-separated format. No additional output is required.

{Retrival}
"""


def preprocess_text(x):
    x = re.sub("http\w+", "", x)  # Delete URL
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()  # Remove empty characters at the beginning and end
    return x


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


def get_candidates(row):
    target_ids = list(map(int, row.MisconceptionId.split()))[22:40]
    misconceptions_text = [mapping_dict.get(m, "error") for m in target_ids]

    res_text = ""
    for i, (_, text) in enumerate(zip(target_ids, misconceptions_text, strict=False)):
        if text == "error":
            continue
        res_text += f"{i}. {text}\n"
    return res_text, target_ids


def postprocess_llm_output(row, length=3):
    x = row.response
    target_ids = row.target_ids

    try:
        res = x.split("\n")[0].replace(",", " ")
        res_lst = list(map(int, res.split()))
        res_lst = [str(target_ids[idx]) for idx in res_lst][:length]
        res = " ".join(res_lst)
        assert len(res_lst) == length
    except:  # noqa
        res = " ".join(row["MisconceptionId"].split()[:length])
    return res


def merge_ranking(row):
    original_list = list(map(int, row.original_ids.split()))
    llm_multi_list = list(map(int, row.llm_multi.split()))
    res_ids = list(dict.fromkeys(original_list[:22] + llm_multi_list + list[22:]))[:25]
    return " ".join(map(str, res_ids))


df = pd.read_parquet("df_target.parquet")

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

df[["retrieval", "target_ids"]] = df.apply(lambda x: get_candidates(x), axis=1, result_type="expand")
df["text"] = df.apply(lambda row: apply_template(row, tokenizer), axis=1)

responses = llm.generate(
    df["llm_input"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=1,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777,  # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=16,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm=True,
)

df["response"] = [x.outputs[0].text for x in responses]
df["llm_multi"] = df.apply(lambda x: postprocess_llm_output(x), axis=1)
df = df.rename(columns={"MisconceptionId": "original_ids"})
df["MisconceptionId"] = df.apply(lambda x: merge_ranking(x), axis=1)
df.to_parquet("df_target.parquet", index=False)
