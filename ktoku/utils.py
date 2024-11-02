import datetime
import logging
import os
import random
import subprocess
import typing as tp

import numpy as np
import torch
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class Logger:
    def __init__(self, log_path="", filename_suffix="exp"):
        self.general_logger = logging.getLogger("general")
        stream_handler = logging.StreamHandler()
        self.general_logger.propagate = False
        file_general_handler = logging.FileHandler(f"{log_path}/general_{filename_suffix}.log")
        for h in self.general_logger.handlers[:]:
            self.general_logger.removeHandler(h)
            h.close()
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def now_string(self):
        return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"))


def convert_value(value):
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, np.floating):
        if np.isnan(value):
            return ""
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    else:
        return str(value)


def get_column_letter(column_number):
    result = ""
    while column_number > 0:
        column_number -= 1
        result = chr(65 + (column_number % 26)) + result
        column_number //= 26
    return result or "A"  # 0や負の数が入力された場合は'A'を返す


class WriteSheet:
    def __init__(self, sheet_json_key, sheet_key, sheet_name):
        import gspread

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(sheet_json_key, scope)
        gs = gspread.authorize(credentials)
        self.worksheet = gs.open_by_key(sheet_key)
        self.sheet = self.worksheet.worksheet(sheet_name)

        # 入力する行番号を取得
        data = self.sheet.get_all_values()
        self.row_number = len(data) + 1

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(60),
        retry=retry_if_exception_type(HttpError),
    )
    def write(self, data, table_range="A1"):
        data_str = [convert_value(d) for d in data]
        self.sheet.append_row(data_str, table_range=table_range)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(60),
        retry=retry_if_exception_type(HttpError),
    )
    def update(self, data):
        data_str = [convert_value(d) for d in data]
        last_column = get_column_letter(len(data))
        self.sheet.update(f"A{self.row_number}:{last_column}{self.row_number}", [data_str])


def get_commit_hash(repo_path="./kaggle-isic2024"):
    wd = os.getcwd()
    os.chdir(repo_path)

    cmd = "git show --format='%H' --no-patch"
    hash_value = subprocess.check_output(cmd.split()).decode("utf-8")[1:-3]

    os.chdir(wd)

    return hash_value


def class_vars_to_dict(cls):
    return {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)}


def print_gpu_utilization(logger):
    from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")


def set_random_seed(seed: int = 23, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    torch.backends.cudnn.benchmark = False


def to_device(tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]], device: torch.device, *args, **kwargs):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def create_random_id(length=7):
    # randomのseedをリセットしてください
    random.seed()
    m = str(datetime.datetime.now().microsecond)
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length)) + m
