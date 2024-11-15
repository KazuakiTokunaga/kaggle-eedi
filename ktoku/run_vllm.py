import argparse

import pandas as pd
import vllm


def main(filename, model_path, quantization=None):
    df = pd.read_parquet(filename)

    if quantization != "awq":
        quantization = None

    llm = vllm.LLM(
        model_path,
        quantization=quantization,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=5120,
        disable_log_stats=True,
    )

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
    df.to_parquet("df_target.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")  # Qwen/Qwen2.5-32B-Instruct-AWQ
    parser.add_argument("--file_path", type=str, default="df_target.parquet")
    parser.add_argument("--quantization", type=str, default=None)  # awq
    args = parser.parse_args()

    print("Run vllm args:", args)
    main(filename=args.file_path, model_path=args.model_path, quantization=args.quantization)
