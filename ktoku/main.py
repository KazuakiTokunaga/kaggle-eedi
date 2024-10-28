# https://www.kaggle.com/code/takanashihumbert/eedi-qwen-2-5-32b-awq-two-time-retrieval

import datetime
import os
import re
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import Logger, WriteSheet, set_random_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RCFG:
    """実行に関連する設定"""

    FILE_NAME = ""  # __file__.split("/")[-1]
    RUN_NAME = ""
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEBUG = False
    DEBUG_SIZE = 30
    COMMIT_HASH = ""
    USE_FOLD = []  # 空のときは全fold、0-4で指定したfoldのみを使う
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
        self.info["fold_auroc"] = [0 for _ in range(5)]
        self.info["mean_auroc"] = 0
        self.info["oof_auroc"] = 0
        self.info["fold_pauc"] = [0 for _ in range(5)]
        self.info["mean_pauc"] = 0
        self.info["oof_pauc"] = 0
        self.info["fold_validloss"] = [0 for _ in range(5)]
        self.info["mean_validloss"] = 0

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
        self.IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
        self.df_train = (
            pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/train.csv").fillna(-1).sample(RCFG.DEBUG_SIZE, random_state=42).reset_index(drop=True)
        )
        self.df_test = pd.read_csv(f"{ROOT_PATH}/input/eedi-mining-misconceptions-in-mathematics/test.csv")

        if not self.IS_SUBMISSION:
            self.df_ret = self.df_train.copy()
        else:
            self.df_ret = self.df_test.copy()
        self.df_misconception_mapping = pd.read_csv(f"{ROOT_PATH}/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

        self.model_llm_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_llm_path)
        self.model_retriever = SentenceTransformer(f"{ROOT_PATH}/eedi-finetuned-bge-public/Eedi-finetuned-bge")

        self.df_ret.to_parquet("df_target.parquet", index=False)

    def prepare_first_llm_generate(
        self,
    ):
        PROMPT = """Here is a question about {ConstructName}({SubjectName}).
        Question: {Question}
        Correct Answer: {CorrectAnswer}

        You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
        Answer concisely what misconception students likely have when solving this question incorrectly.
        No need to give the reasoning process and do not use "The misconception is" to start your answers.
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
                            CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                        ),
                        ver="v2",
                    ),
                }
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text

        df = pd.read_parquet("df_target.parquet")
        llm_inputs = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            llm_inputs.append(apply_template(row, self.tokenizer))
        df["llm_input"] = llm_inputs
        df.to_parquet("df_target.parquet", index=False)

    def retrieval_first(
        self,
    ):
        self.df_ret = pd.read_parquet("df_target.parquet")
        self.df_ret["input_features"] = self.df_ret["ConstructName"] + ". " + self.df_ret["SubjectName"] + ". " + self.df_ret["first_llmMisconception"]
        self.df_ret["input_features"] = self.df_ret["input_features"].apply(lambda x: preprocess_text(x))

        logger.info("Retrieval starts.")
        embedding_query = self.model_retriever.encode(self.df_ret["input_features"], convert_to_tensor=True)
        misconceptions = self.df_misconception_mapping.MisconceptionName.values
        embedding_Misconception = self.model_retriever.encode(misconceptions, convert_to_tensor=True)
        Ret_topNids = util.semantic_search(embedding_query, embedding_Misconception, top_k=100)
        logger.info("Retrieval ends.")

        retrivals = []
        self.dicts = {}
        for idx, row in tqdm(self.df_ret.iterrows(), total=len(self.df_ret)):
            top_ids = Ret_topNids[idx]
            retrival = ""
            self.dicts[str(row["QuestionId"])] = {}
            for i, ids in enumerate(top_ids):
                retrival += f"{i+1}. " + misconceptions[ids["corpus_id"]] + "\n"
                self.dicts[str(row["QuestionId"])][str(i + 1)] = misconceptions[ids["corpus_id"]]
            retrivals.append(retrival)
        self.df_ret["Retrival"] = retrivals

    def prepare_llm_generate(
        self,
    ):
        PROMPT = """Here is a question about {ConstructName}({SubjectName}).
        Question: {Question}
        Correct Answer: {CorrectAnswer}
        Incorrect Answer: {IncorrectAnswer}

        You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
        Answer concisely what misconception it is to lead to getting the incorrect answer.
        No need to give the reasoning process and do not use "The misconception is" to start your answers.
        There are some relative and possible misconceptions below to help you make the decision:

        {Retrival}
        """

        def apply_template(row, tokenizer, targetCol):
            messages = [
                {
                    "role": "user",
                    "content": preprocess_text(
                        PROMPT.format(
                            ConstructName=row["ConstructName"],
                            SubjectName=row["SubjectName"],
                            Question=row["QuestionText"],
                            IncorrectAnswer=row[f"Answer{targetCol}Text"],
                            CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                            Retrival=row["Retrival"],
                        ),
                        ver="v2",
                    ),
                }
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text

        self.df_llm = {}
        if not self.IS_SUBMISSION:
            df_label = {}
            for _, row in tqdm(self.df_ret.iterrows(), total=len(self.df_ret)):
                for option in ["A", "B", "C", "D"]:
                    if (row.CorrectAnswer != option) & (row[f"Misconception{option}Id"] != -1):
                        self.df_llm[f"{row.QuestionId}_{option}"] = apply_template(row, self.tokenizer, option)
                        df_label[f"{row.QuestionId}_{option}"] = [row[f"Misconception{option}Id"]]

            df_label = pd.DataFrame([df_label]).T.reset_index()
            df_label.columns = ["QuestionId_Answer", "MisconceptionId"]
            df_label.to_parquet("label.parquet", index=False)
        else:
            for _, row in tqdm(self.df_ret.iterrows(), total=len(self.df_ret)):
                for option in ["A", "B", "C", "D"]:
                    if row.CorrectAnswer != option:
                        self.df_llm[f"{row.QuestionId}_{option}"] = apply_template(row, self.tokenizer, option)

        self.df_llm = pd.DataFrame([self.df_llm]).T.reset_index()
        self.df_llm.columns = ["QuestionId_Answer", "text"]
        self.df_llm.to_parquet("submission.parquet", index=False)

    def retrieval_second(
        self,
    ):
        self.df_sub = pd.read_parquet("submission.parquet")

        def number2sentence(row):
            """
            This is used for post-processing of LLM's output.
            Since we give top-N retrieval to the LLM with serial number,
            Sometimes the LLM will only output the serial number without any sentence.
            We use the 'dicts' generated at the beginning to map the serial number with corresponding misconceptions.
            """
            text = row["llmMisconception"].strip()
            # potential is the most possible serial number in LLM output.
            potential = re.search(r"^\w+\.{0,1}", text).group()
            if "." in potential:
                sentence = text.replace(potential, "").strip()
            # if the LLM output is only a serial number, we map it with corresponding misconceptions saved in the dict.
            elif len(potential) == len(text):
                qid_retrieval = self.dicts[row["QuestionId"]]
                try:
                    # qid_retrieval is the top-N misconceptions for an QuestionId,
                    # qid_retrieval[potential] is the most possible misconception.
                    sentence = qid_retrieval[potential]
                except:  # noqa
                    # If the mapping fails, we use the first one(the most possible one in the first retrieval).
                    sentence = qid_retrieval["1"]
            else:
                sentence = text

            return sentence

        self.df_sub["QuestionId"] = self.df_sub["QuestionId_Answer"].apply(lambda x: x.split("_")[0])
        self.df_sub["llmMisconception_clean"] = self.df_sub.apply(number2sentence, axis=1)

        PREFIX = "<|im_start|>user"
        self.df_sub["input_features"] = self.df_sub["text"].apply(
            lambda x: x.split(PREFIX)[1].split("You are a Mathematics teacher.")[0].strip("\n").split("Here is a question about")[-1].strip()
        )

        self.df_sub["input_features"] = self.df_sub["input_features"].apply(lambda x: preprocess_text(x))
        self.df_sub["input_features"] = self.df_sub["llmMisconception_clean"] + "\n\n" + self.df_sub["input_features"]

        logger.info("Retrieval starts.")
        embedding_query = self.model_retriever.encode(self.df_sub["input_features"], convert_to_tensor=True)
        embedding_Misconception = self.model_retriever.encode(self.df_misconception_mapping.MisconceptionName.values, convert_to_tensor=True)
        top25ids = util.semantic_search(embedding_query, embedding_Misconception, top_k=25)
        logger.info("Retrieval ends.")

        self.df_sub["MisconceptionId"] = [" ".join([str(x["corpus_id"]) for x in top25id]) for top25id in top25ids]
        self.df_sub[["QuestionId_Answer", "MisconceptionId"]].to_csv("submission.csv", index=False)
        self.df_sub.head()
