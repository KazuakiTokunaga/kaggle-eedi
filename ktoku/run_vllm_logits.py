import argparse

import pandas as pd
import vllm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from transformers import AutoTokenizer


def main(filename, model_path, quantization=None, suffix="r0"):
    df = pd.read_parquet(filename)

    if quantization != "awq":
        quantization = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
            n=1,
            top_k=1,
            temperature=0,
            seed=777,
            skip_special_tokens=False,
            max_tokens=1,
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])],
        ),
        use_tqdm=True,
    )

    responses = [x.outputs[0].text for x in responses]
    df[f"fullLLMText_{suffix}"] = responses
    df.to_parquet("df_target.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")  # Qwen/Qwen2.5-32B-Instruct-AWQ
    parser.add_argument("--file_path", type=str, default="df_target.parquet")
    parser.add_argument("--quantization", type=str, default=None)  # awq
    parser.add_argument("--suffix", type=str, default="v1")
    args = parser.parse_args()

    print("Run vllm args:", args)
    main(filename=args.file_path, model_path=args.model_path, quantization=args.quantization, suffix=args.suffix)
