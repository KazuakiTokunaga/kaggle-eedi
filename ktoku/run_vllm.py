import re

import pandas as pd
import vllm

df = pd.read_parquet("df_target.parquet")

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

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


responses = llm.generate(
    df["llm_input"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.8,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777,  # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=512,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm=True,
)

responses = [x.outputs[0].text for x in responses]
df["fullLLMText"] = responses


def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()


df["first_llmMisconception"] = responses
df.to_parquet("df_target.parquet", index=False)
