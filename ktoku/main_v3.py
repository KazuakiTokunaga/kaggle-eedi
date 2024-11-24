# https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from utils import Logger, WriteSheet, class_vars_to_dict, create_random_id, set_random_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RCFG:
    """実行に関連する設定"""

    SUBMIT = False
    FILE_NAME = __file__.split("/")[-1]
    RUN_NAME = "test2"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEBUG = False
    DEBUG_SIZE = 30
    COMMIT_HASH = ""
    MODEL_LLM_PATH = "Qwen/Qwen2.5-3B-Instruct"
    AWQ = "none"
    DROP_NA = False
    SAVE_TO_SHEET = True
    SHEET_KEY = "1LTgeCmbwwbF3bdt6x2J00GtpEEG1ArI3xr52IXZZmtQ"
    SHEET_NAME = "cv_ktoku"


RCFG.RUN_NAME = create_random_id()


def apk(actual, predicted, k=25):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted, strict=False)])


def preprocess_text(x, ver="v1"):
    if ver == "v1":
        x = x.lower()  # Convert words to lowercase
        x = re.sub("@\w+", "", x)  # Delete strings starting with @
    x = re.sub("http\w+", "", x)  # Delete URL
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\.+", ".", x)  # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = x.strip()  # Remove empty characters at the beginning and end

    return x


def get_val_score(df_target, target_col="MisconceptionId", k=25):
    apks = []
    for _, row in df_target.iterrows():
        if pd.isna(row.true):
            apks.append(np.nan)
        else:
            actual = [int(row.true)]
            pred = list(map(int, row[target_col].split()))
            apk_score = apk(actual, pred, k)
            apks.append(apk_score)

    df_target[f"apk_{target_col}"] = apks
    val_score = df_target[f"apk_{target_col}"].mean()
    return df_target, val_score


def create_retrieval_text(row, mapping: dict):
    misconceptions_ids = list(map(int, row.MisconceptionId.split()))[31:60]
    misconceptions_text = [mapping.get(m, "error") for m in misconceptions_ids]

    res_text = ""
    notfound_occurred = 0
    index_mapping = dict()
    for i, (id, text) in enumerate(zip(misconceptions_ids, misconceptions_text, strict=False)):
        if text == "error":
            notfound_occurred += 1
            continue
        index_mapping[i] = id
        res_text += f"{i}. {text}\n"
    return res_text, notfound_occurred, index_mapping


def create_retrieval_text_r(row, mapping: dict, r=0):
    misconceptions_ids = list(map(int, row.MisconceptionId.split()))[:30]
    llm_ids = list(map(int, row.llm_id_v1.split()))
    total_ids = list(dict.fromkeys(misconceptions_ids + llm_ids))
    if len(total_ids) != 36:
        total_ids = list(map(int, row.MisconceptionId.split()))[:36]
    idx = [i for i in range(36) if i % 4 == r]
    target_ids = [total_ids[i] for i in idx]
    misconceptions_text = [mapping.get(m, "error") for m in target_ids]

    res_text = ""
    notfound_occurred = 0
    index_mapping = []
    for i, (id, text) in enumerate(zip(target_ids, misconceptions_text, strict=False)):
        if text == "error":
            notfound_occurred += 1
            continue
        index_mapping.append(id)
        res_text += f"{i}. {text}\n"
    return res_text, notfound_occurred, index_mapping


def create_retrieval_text_final(row, mapping: dict):
    target_ids = list(set([row.llm_id_r0, row.llm_id_r1, row.llm_id_r2, row.llm_id_r3]))
    if len(target_ids) != 4:
        target_ids = list(map(int, row.MisconceptionId.split()))[:4]
    misconceptions_text = [mapping.get(m, "error") for m in target_ids]

    res_text = ""
    notfound_occurred = 0
    index_mapping = []
    for i, (id, text) in enumerate(zip(target_ids, misconceptions_text, strict=False)):
        if text == "error":
            notfound_occurred += 1
            continue
        index_mapping.append(id)
        res_text += f"{i}. {text}\n"
    return res_text, notfound_occurred, index_mapping


def apply_template(row, tokenizer, number="ten"):
    PROMPT = """Here is a question about {ConstructName}({SubjectName}).
    Question: {Question}
    Correct Answer: {CorrectAnswer}
    Incorrect Answer: {IncorrectAnswer}

    You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
    From the options below, please select the {Number} that you consider most appropriate, in order of preference.
    Answer by listing the option numbers in a comma-separated format. No additional output is required.

    Options:

    {Retrival}
    """

    messages = [
        {
            "role": "user",
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"Answer{row.answer_name}Text"],
                    CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                    Number=number,
                    Retrival=row.retrieval_text,
                ),
                ver="v2",
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def apply_template_single(row, tokenizer):
    PROMPT = """Here is a question about {ConstructName}({SubjectName}).
    Question: {Question}
    Correct Answer: {CorrectAnswer}
    Incorrect Answer: {IncorrectAnswer}

    You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
    Answer concisely what misconception it is to lead to getting the incorrect answer.
    Pick the correct misconception number from the below:

    {Retrival}
    """

    messages = [
        {
            "role": "user",
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"Answer{row.answer_name}Text"],
                    CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                    Retrival=row.retrieval_text,
                ),
                ver="v2",
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def postprocess_llm_output(row, length=10):
    x = row["fullLLMText"]
    mapping = row["mapping_dict"]
    exception_occurred = 0
    try:
        res = x.split("\n")[0].replace(",", " ")
        res_lst = list(map(int, res.split()))
        res_lst = [mapping[idx] for idx in res_lst if idx in mapping][:length]
        res = " ".join(res_lst)
        assert len(res_lst) == length
    except:  # noqa
        res = " ".join(row["MisconceptionId"].split()[:length])
        exception_occurred += 1
    return res, exception_occurred


def postprocess_llm_output_single(row, suffix="r0", fallback=0):
    x = row[f"fullLLMText_{suffix}"]
    mapping = row["mapping_dict"]
    exception_occurred = 0
    try:
        res = mapping[x]
    except:  # noqa
        res = int(row["MisconceptionId"].split()[fallback])
        exception_occurred += 1
    return res, exception_occurred


def merge_ranking(r1, r2, w1=0.5, w2=0.5):
    ranking_dict = {idx: rank for rank, idx in enumerate(r1)}
    reranking_dict = {idx: rank for rank, idx in enumerate(r2)}

    # スコアを計算
    scores = []
    for idx in r1:
        rank_score = ranking_dict.get(idx, len(r1))
        rerank_score = reranking_dict.get(idx, len(r2))
        score = w1 * rank_score + w2 * rerank_score
        scores.append((idx, score))

    # スコアでソートしてランキングを改良
    refined_ranking = [idx for idx, score in sorted(scores, key=lambda x: x[1])]
    return refined_ranking


def create_merge_ranking_columns(row):
    res_list = [row.llm_id_final] + [row.llm_id_r0, row.llm_id_r1, row.llm_id_r2, row.llm_id_r3] + row.MisconceptionId.split()
    res_list = list(dict.fromkeys(res_list))[:25]

    return " ".join(map(str, res_list))


######################################################
# Runner
######################################################


class Runner:
    def __init__(self, env="colab", commit_hash=""):
        global ENV, ROOT_PATH, OUTPUT_PATH
        ENV = env
        ROOT_PATH = "/content/drive/MyDrive/eedi" if ENV == "colab" else "/kaggle"
        if ENV == "colab":
            OUTPUT_PATH = Path(ROOT_PATH) / "output" / RCFG.RUN_NAME
            OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
        else:
            OUTPUT_PATH = "/kaggle/working"

        set_random_seed()
        global logger
        logger = Logger(log_path=OUTPUT_PATH, filename_suffix=RCFG.RUN_NAME)
        logger.info(f"Initializing Runner.　Run Name: {RCFG.RUN_NAME}")
        start_dt_jst = str(datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"))
        self.info = {"start_dt_jst": start_dt_jst}
        self.info["scores"] = []

        logger.info(f"commit_hash: {commit_hash}")
        RCFG.COMMIT_HASH = commit_hash

        if ENV == "kaggle":
            from kaggle_secrets import UserSecretsClient

            self.user_secrets = UserSecretsClient()

        # MODEL_LLM_PATHがQwen2.5-32B-Instruct-AWQを部分文字列で含むかどうか
        if "Qwen2.5-32B-Instruct-AWQ" in RCFG.MODEL_LLM_PATH:
            logger.info("Use Large Model with AWQ.")
            RCFG.AWQ = "awq"

        if RCFG.SAVE_TO_SHEET:
            sheet_json_key = ROOT_PATH + "/input/ktokunagautils/ktokunaga-4094cf694f5c.json"
            logger.info("Initializing Google Sheet.")
            self.sheet = WriteSheet(sheet_json_key=sheet_json_key, sheet_key=RCFG.SHEET_KEY, sheet_name=RCFG.SHEET_NAME)
            self._update_sheet()

    def load_dataset(
        self,
    ):
        self.df_train = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/train.csv")
        self.df_test = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/test.csv")
        self.df_mapping = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
        self.mapping_dict = self.df_mapping.set_index("MisconceptionId")["MisconceptionName"].to_dict()

        if RCFG.DEBUG:
            logger.info(f"DEBUG MODE. Reduce the size of the dataset: {RCFG.DEBUG_SIZE}.")
            self.df_train = self.df_train.sample(RCFG.DEBUG_SIZE, random_state=42).reset_index(drop=True)

        # self.model_llm_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
        self.model_llm_path = RCFG.MODEL_LLM_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_llm_path)

    def retrieve(
        self,
    ):
        pass

    def prepare_llm_reranker(
        self,
    ):
        if RCFG.SUBMIT:
            df_target = pd.read_parquet("df_target.parquet")
        else:
            df_target = pd.read_csv(f"{ROOT_PATH}/input/baseline/train_df.csv")

        if RCFG.DEBUG:
            logger.info(f"DEBUG MODE. Reduce the size of the dataset: {RCFG.DEBUG_SIZE}.")
            df_target = df_target.sample(RCFG.DEBUG_SIZE, random_state=42).reset_index(drop=True)

        logger.info("Create LLM input for llmreranker.")
        if not RCFG.SUBMIT:
            df_target["true"] = df_target.apply(lambda x: x[f"Misconception{x.answer_name}Id"], axis=1)
        else:
            df_target["true"] = 1
        if RCFG.DROP_NA:
            df_target = df_target.dropna(subset=["true"]).reset_index(drop=True)
        df_target[["retrieval_text", "notfound_count", "mapping_dict"]] = df_target.apply(lambda x: create_retrieval_text(x, self.mapping_dict), axis=1, result_type="expand")
        logger.info(f"NOTFOUND_COUNT: {df_target['notfound_count'].sum()}")
        df_target["llm_input"] = df_target.apply(lambda x: apply_template(x, self.tokenizer, number="six"), axis=1)
        df_target.to_parquet("df_target.parquet", index=False)

        logger.info("Prepare LLM reranker done.")

    def prepare_llm_reranker_r0(
        self,
    ):
        df_target = pd.read_parquet("df_target.parquet")
        logger.info("Create llm_id_v1 with prostprocess.")
        df_target[["llm_id_v1", "exception_flag"]] = df_target.apply(lambda x: postprocess_llm_output(x, 10), axis=1, result_type="expand")
        logger.info(f"EXCEPTION_COUNT: {df_target['exception_flag'].sum()}")

        logger.info("Create LLM input for llmreranker_r0.")
        df_target[["retrieval_text", "notfound_count", "mapping_dict"]] = df_target.apply(
            lambda x: create_retrieval_text_r(x, self.mapping_dict, r=0), axis=1, result_type="expand"
        )
        logger.info(f"NOTFOUND_COUNT: {df_target['notfound_count'].sum()}")

        df_target["llm_input"] = df_target.apply(lambda x: apply_template(x, self.tokenizer), axis=1)
        df_target.to_parquet("df_target.parquet", index=False)

        logger.info("Prepare LLM reranker done.")

    def prepare_llm_reranker_r123(self, r=1):
        df_target = pd.read_parquet("df_target.parquet")

        logger.info(f"Create LLM input for llmreranker for r={r}.")
        df_target[["retrieval_text", "notfound_count", "mapping_dict"]] = df_target.apply(
            lambda x: create_retrieval_text_r(x, self.mapping_dict, r=r), axis=1, result_type="expand"
        )
        logger.info(f"NOTFOUND_COUNT: {df_target['notfound_count'].sum()}")

        df_target["llm_input"] = df_target.apply(lambda x: apply_template(x, self.tokenizer), axis=1)
        df_target.to_parquet("df_target.parquet", index=False)

        logger.info(f"Prepare LLM reranker for r={r} done.")

    def prepare_llm_reranker_final(
        self,
    ):
        df_target = pd.read_parquet("df_target.parquet")

        for i in range(4):
            logger.info(f"Create llm_id_r{i} with prostprocess.")
            df_target[[f"llm_id_r{i}", "exception_flag"]] = df_target.apply(lambda x: postprocess_llm_output_single(x, fallback=i, suffix=f"r{i}"), axis=1, result_type="expand")
            logger.info(f"EXCEPTION_COUNT: {df_target['exception_flag'].sum()}")

        df_target[["retrieval_text", "notfound_count", "mapping_dict"]] = df_target.apply(lambda x: create_retrieval_text_final(x, self.mapping_dict), axis=1, result_type="expand")
        logger.info(f"NOTFOUND_COUNT: {df_target['notfound_count'].sum()}")

        df_target["llm_input"] = df_target.apply(lambda x: apply_template_single(x, self.tokenizer), axis=1)
        df_target.to_parquet("df_target.parquet", index=False)

        logger.info("Prepare LLM reranker final done.")

    def merge_ranking(
        self,
    ):
        df_target = pd.read_parquet("df_target.parquet")
        df_target, val_score = get_val_score(df_target, target_col="MisconceptionId")
        logger.info(f"MisconceptionId_score: {val_score}")
        self.info["scores"].append(val_score)

        logger.info("Create llm_id final with prostprocess.")
        df_target[["llm_id_final", "exception_flag"]] = df_target.apply(lambda x: postprocess_llm_output_single(x, fallback=0, suffix="final"), axis=1, result_type="expand")
        logger.info(f"EXCEPTION_COUNT: {df_target['exception_flag'].sum()}")
        self.info["scores"].append(0)

        df_target["merged_ranking"] = df_target.apply(lambda x: create_merge_ranking_columns(x), axis=1)
        df_target, val_score = get_val_score(df_target, target_col="merged_ranking")
        logger.info(f"merged_ranking_score: {val_score}")
        self.info["scores"].append(val_score)

        df_target.to_parquet("df_target.parquet", index=False)
        df_target.to_parquet(Path(OUTPUT_PATH) / "df_target.parquet", index=False)
        self._update_sheet()

    def create_submission_file(
        self,
    ):
        df_target = pd.read_parquet("df_target.parquet")
        sub = []
        for _, row in df_target.iterrows():
            sub.append({"QuestionId_Answer": f"{row['QuestionId']}_{row['answer_name']}", "MisconceptionId": row["merged_ranking"]})
        submission_df = pd.DataFrame(sub)
        submission_df.to_csv("submission.csv", index=False)
        logger.info("Submission file created successfully!")

    def _update_sheet(
        self,
    ):
        if not RCFG.SAVE_TO_SHEET:
            return None

        logger.info("Update info to google sheet.")
        write_dt_jst = str(datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"))

        data = [
            RCFG.RUN_NAME,
            self.info["start_dt_jst"],
            write_dt_jst,
            RCFG.COMMIT_HASH,
            RCFG.FILE_NAME,
            class_vars_to_dict(RCFG),
            *self.info["scores"],
        ]

        self.sheet.update(data)
