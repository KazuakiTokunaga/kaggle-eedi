# https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval

import datetime
import os
from pathlib import Path

import numpy as np
import polars as pl
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.metrics.pairwise import cosine_similarity
from utils import Logger, WriteSheet, set_random_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
NUM_PROC = os.cpu_count()


class RCFG:
    """実行に関連する設定"""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    RETRIEVE_NUM = 25
    EPOCH = 2
    LR = 2e-05
    BS = 8
    GRAD_ACC_STEP = 128 // BS
    TRAINING = True
    DEBUG = False
    WANDB = True


######################################################
# Runner
######################################################


class Runner:
    def __init__(self, env="colab", commit_hash=""):
        global ENV, ROOT_PATH, OUTPUT_PATH
        ENV = env
        ROOT_PATH = "/content/drive/MyDrive/eedi" if ENV == "colab" else "/kaggle"
        OUTPUT_PATH = ROOT_PATH if ENV == "colab" else "/kaggle/working"
        if ENV == "kaggle":
            (Path(OUTPUT_PATH) / "log").mkdir(exist_ok=True)
            (Path(OUTPUT_PATH) / "data").mkdir(exist_ok=True)
            (Path(OUTPUT_PATH) / "model").mkdir(exist_ok=True)

        set_random_seed()
        global logger
        logger = Logger(log_path=f"{OUTPUT_PATH}/log/", filename_suffix=RCFG.RUN_NAME)
        logger.info(f"Initializing Runner.　Run Name: {RCFG.RUN_NAME}")
        start_dt_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"))

        # Initialize info
        self.info = {"start_dt_jst": start_dt_jst}

        logger.info(f"commit_hash: {commit_hash}")
        RCFG.COMMIT_HASH = commit_hash

        if ENV == "kaggle":
            from kaggle_secrets import UserSecretsClient

            self.user_secrets = UserSecretsClient()

        if RCFG.SAVE_TO_SHEET:
            sheet_json_key = ROOT_PATH + "/input/ktokunagautils/ktokunaga-4094cf694f5c.json"
            logger.info("Initializing Google Sheet.")
            self.sheet = WriteSheet(sheet_json_key=sheet_json_key, sheet_key=RCFG.SHEET_KEY, sheet_name=RCFG.SHEET_NAME)

    def run(
        self,
    ):
        train = pl.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/train.csv")
        misconception_mapping = pl.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

        common_col = [
            "QuestionId",
            "ConstructName",
            "SubjectName",
            "QuestionText",
            "CorrectAnswer",
        ]

        train_long = (
            train.select(pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]]))
            .unpivot(
                index=common_col,
                variable_name="AnswerType",
                value_name="AnswerText",
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("ConstructName"),
                        pl.col("SubjectName"),
                        pl.col("QuestionText"),
                        pl.col("AnswerText"),
                    ],
                    separator=" ",
                ).alias("AllText"),
                pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
            )
            .with_columns(
                pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
            )
            .sort("QuestionId_Answer")
        )

        train_misconception_long = (
            train.select(pl.col(common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]))
            .unpivot(
                index=common_col,
                variable_name="MisconceptionType",
                value_name="MisconceptionId",
            )
            .with_columns(
                pl.col("MisconceptionType").str.extract(r"Misconception([A-D])Id$").alias("AnswerAlphabet"),
            )
            .with_columns(
                pl.concat_str([pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_").alias("QuestionId_Answer"),
            )
            .sort("QuestionId_Answer")
            .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
            .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
        )

        # join MisconceptionId
        train_long = train_long.join(train_misconception_long, on="QuestionId_Answer")

        model = SentenceTransformer(RCFG.MODEL_NAME)
        train_long_vec = model.encode(train_long["AllText"].to_list(), normalize_embeddings=True)
        misconception_mapping_vec = model.encode(misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True)

        train_cos_sim_arr = cosine_similarity(train_long_vec, misconception_mapping_vec)
        train_sorted_indices = np.argsort(-train_cos_sim_arr, axis=1)

        train_long = train_long.with_columns(pl.Series(train_sorted_indices[:, : RCFG.RETRIEVE_NUM].tolist()).alias("PredictMisconceptionId"))

        train_retrieved = (
            train_long.filter(pl.col("MisconceptionId").is_not_null())
            .explode("PredictMisconceptionId")
            .join(
                misconception_mapping,
                on="MisconceptionId",
            )
            .join(
                misconception_mapping.rename(lambda x: "Predict" + x),
                on="PredictMisconceptionId",
            )
        )

        train = Dataset.from_polars(train_retrieved).filter(  # To create an anchor, positive, and negative structure, delete rows where the positive and negative are identical.
            lambda example: example["MisconceptionId"] != example["PredictMisconceptionId"],
            num_proc=NUM_PROC,
        )

        logger.info(f"train.shape: {train.shape}")
        if RCFG.DEBUG:
            train = train.select(range(1000))
            EPOCH = 1

        model = SentenceTransformer(RCFG.MODEL_NAME)
        loss = MultipleNegativesRankingLoss(model)
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=OUTPUT_PATH,
            # Optional training parameters:
            num_train_epochs=EPOCH,
            per_device_train_batch_size=RCFG.BS,
            gradient_accumulation_steps=RCFG.GRAD_ACC_STEP,
            per_device_eval_batch_size=RCFG.BS,
            eval_accumulation_steps=RCFG.GRAD_ACC_STEP,
            learning_rate=RCFG.LR,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            lr_scheduler_type="cosine_with_restarts",
            save_strategy="steps",
            save_steps=0.1,
            save_total_limit=2,
            logging_steps=100,
            report_to=RCFG.REPORT_TO,  # Will be used in W&B if `wandb` is installed
            run_name=RCFG.EXP_NAME,
            do_eval=False,
        )

        trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train.select_columns(["AllText", "MisconceptionName", "PredictMisconceptionName"]), loss=loss)

        logger.info("Start training.")
        trainer.train()
        model.save_pretrained(f"{OUTPUT_PATH}/model/test")
