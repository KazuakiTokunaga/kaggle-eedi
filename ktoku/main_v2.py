# https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from eedi_score import apk
from transformers import AutoTokenizer
from utils import Logger, WriteSheet, set_random_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RCFG:
    """実行に関連する設定"""

    FILE_NAME = ""  # __file__.split("/")[-1]
    RUN_NAME = "test2"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEBUG = False
    DEBUG_SIZE = 30
    COMMIT_HASH = ""
    USE_FOLD = []  # 空のときは全fold、0-4で指定したfoldのみを使う
    MODEL_LLM_PATH = "Qwen/Qwen2.5-3B-Instruct"
    DROP_NA = False
    SAVE_TO_SHEET = True
    SHEET_KEY = "1D-8FAIA4mj7LxaUkiQb5L1dS_aZ2pdSB2uK8Hd_DTfI"
    SHEET_NAME = "cv_ktoku"


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
    df_target["true"] = df_target.apply(lambda x: x[f"Misconception{x.answer_name}Id"], axis=1)

    apks = []
    for _, row in df_target.iterrows():
        if pd.isna(row.true):
            apks.append(np.nan)
        else:
            actual = [int(row.true)]
            pred = list(map(int, row[target_col].split()))
            apk_score = apk(actual, pred, k)
            apks.append(apk_score)

    df_target["apk"] = apks
    val_score = df_target["apk"].mean()
    return df_target, val_score


def create_retrieval_text(row, mapping, k=25):
    misconceptions_ids = list(map(int, row.MisconceptionId.split()))
    misconceptions_text = [mapping[mapping.MisconceptionId == m].MisconceptionName.values[0] for m in misconceptions_ids]

    res_text = ""
    for i, (id, text) in enumerate(zip(misconceptions_ids, misconceptions_text)):
        res_text += f"{id}. {text}\n"
        if i == k - 1:
            break
    return res_text


def apply_template(row, tokenizer):
    PROMPT = """Here is a question about {ConstructName}({SubjectName}).
    Question: {Question}
    Correct Answer: {CorrectAnswer}
    Incorrect Answer: {IncorrectAnswer}

    You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
    From the options below, please select the ten that you consider most appropriate, in order of preference.
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
                    Retrival=row.retrieval_text,
                ),
                ver="v2",
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def postprocess_llm_output(row):
    x = row["fullLLMText"]
    try:
        res = x.split("\n")[0].replace(",", " ")
        res_lst = list(map(int, res.split()))
        assert len(res_lst) == 10
    except:  # noqa
        res = " ".join(row["MisconceptionId"].split()[:10])
    return res


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


def create_merge_ranking_columns(row, w1=0.5, w2=0.5):
    r1 = list(map(int, row["MisconceptionId"].split()))
    r2 = list(map(int, row["llm_misconceptionId"].split()))
    refined_ranking = merge_ranking(r1, r2, w1=w1, w2=w2)
    return " ".join(map(str, refined_ranking))


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

        logger.info(f"commit_hash: {commit_hash}")
        RCFG.COMMIT_HASH = commit_hash

        if ENV == "kaggle":
            from kaggle_secrets import UserSecretsClient

            self.user_secrets = UserSecretsClient()

        if RCFG.SAVE_TO_SHEET:
            sheet_json_key = ROOT_PATH + "/input/ktokunagautils/ktokunaga-4094cf694f5c.json"
            logger.info("Initializing Google Sheet.")
            self.sheet = WriteSheet(sheet_json_key=sheet_json_key, sheet_key=RCFG.SHEET_KEY, sheet_name=RCFG.SHEET_NAME)

    def load_dataset(
        self,
    ):
        self.df_train = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/train.csv")
        self.df_test = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/test.csv")
        self.df_mapping = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

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

    def prepare_llm_reranker(self, df_target):
        logger.info("Create LLM input for llmreranker.")
        df_target["true"] = df_target.apply(lambda x: x[f"Misconception{x.answer_name}Id"], axis=1)
        if RCFG.DROP_NA:
            df_target = df_target.dropna(subset=["true"]).reset_index(drop=True)
        df_target["retrieval_text"] = df_target.apply(lambda x: create_retrieval_text(x, self.df_mapping), axis=1)
        df_target["llm_input"] = df_target.apply(lambda x: apply_template(x, self.tokenizer), axis=1)

        logger.info("Save df_target.")
        df_target.to_parquet(OUTPUT_PATH / "df_target.parquet", index=False)

        return df_target

    def merge_ranking(self, df_target):
        df_target["llm_misconceptionId"] = df_target.apply(lambda x: postprocess_llm_output(x), axis=1)
        df_target, val_score = get_val_score(df_target, target_col="llm_misconceptionId")
        logger.info(f"llm_misconceptionId_score: {val_score}")

        df_target.to_parquet(OUTPUT_PATH / "df_target.parquet", index=False)
        df_target.to_parquet("df_target.parquet", index=False)

        df_target["merged_ranking"] = df_target.apply(lambda x: create_merge_ranking_columns(x), axis=1)
        df_target, val_score = get_val_score(df_target, target_col="merged_ranking")
        logger.info(f"merged_ranking_score: {val_score}")

        df_target.to_parquet(OUTPUT_PATH / "df_target.parquet", index=False)

        return df_target

    def create_submission_file(self, df_target):
        sub = []
        for _, row in df_target.iterrows():
            sub.append({"QuestionId_Answer": f"{row['QuestionId']}_{row['answer_name']}", "MisconceptionId": row["MisconceptionId"]})
        submission_df = pd.DataFrame(sub)
        submission_df.to_csv("submission.csv", index=False)
        logger.info("Submission file created successfully!")
