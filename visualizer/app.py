from pathlib import Path

import polars as pl
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

INPUT_DIR = Path("data/")


def replace_latex(text: str) -> str:
    return text.replace(r"\[", "$").replace(r"\]", "$").replace(r"\(", "$").replace(r"\)", "$")


def plot_answer(
    col: DeltaGenerator,
    answer: str,
    misconception_id: float | None,
    alphabet: str,
    is_correct_answer: bool,
    misconception_mapping: dict[int, str],
) -> None:
    if misconception_id is not None:
        misconception_id = int(misconception_id)
        misconception_name = misconception_mapping[misconception_id]
    else:
        misconception_name = ""

    with col:
        if is_correct_answer:
            st.write(f"{alphabet} (Correct Answer)")
            st.success(f"Answer: {replace_latex(answer)}")
            st.success(f"Misconception: {misconception_name} (id={misconception_id})")
        else:
            st.write(alphabet)
            st.error(f"Answer: {replace_latex(answer)}")
            st.error(f"Misconception: {misconception_name} (id={misconception_id})")


def main() -> None:
    st.set_page_config(page_title="Eedi Visualizer", layout="wide")
    st.title("Eedi visualizer")

    df = pl.read_csv(INPUT_DIR / "train.csv")
    misconception_mapping = pl.read_csv(INPUT_DIR / "misconception_mapping.csv")
    misconception_mapping_dict = dict(zip(misconception_mapping["MisconceptionId"].to_list(), misconception_mapping["MisconceptionName"].to_list()))

    st.markdown("## Question")
    col1, _, _, _ = st.columns(4)
    with col1:
        id = st.number_input(label="QuestionId", min_value=0, max_value=df["QuestionId"].max())
    _data = df[id].to_dict(as_series=False)
    data = {k: v[0] for k, v in _data.items()}

    st.info(f"ConstructName: {data['ConstructName']}")
    st.info(f"SubjectName: {data['SubjectName']}")
    st.info(f"QuestionText: {replace_latex(data['QuestionText'])}")

    col1, col2, col3, col4 = st.columns(4)
    plot_answer(col1, data["AnswerAText"], data["MisconceptionAId"], "A", data["CorrectAnswer"] == "A", misconception_mapping_dict)
    plot_answer(col2, data["AnswerBText"], data["MisconceptionBId"], "B", data["CorrectAnswer"] == "B", misconception_mapping_dict)
    plot_answer(col3, data["AnswerCText"], data["MisconceptionCId"], "C", data["CorrectAnswer"] == "C", misconception_mapping_dict)
    plot_answer(col4, data["AnswerDText"], data["MisconceptionDId"], "D", data["CorrectAnswer"] == "D", misconception_mapping_dict)

    st.markdown("## Question Search")
    col1, col2, col3, _ = st.columns(4)
    with col1:
        ConstructName = st.text_input(label="ConstructName")
    with col2:
        SubjectName = st.text_input(label="SubjectName")
    with col3:
        QuestionText = st.text_input(label="QuestionText")
    filtered_df = (
        df.filter(pl.col("ConstructName").str.contains(f"(?i){ConstructName}"))
        .filter(pl.col("SubjectName").str.contains(f"(?i){SubjectName}"))
        .filter(pl.col("QuestionText").str.contains(f"(?i){QuestionText}"))
    )
    filtered_df = filtered_df.select(["QuestionId", "ConstructName", "SubjectName", "QuestionText"])[:100]
    st.dataframe(
        filtered_df,
        column_config={
            "QuestionText": st.column_config.Column("QuestionText", width=1000),
        },
        height=600,
    )

    st.markdown("## Misconception Mapping")
    col1, col2, _, _ = st.columns(4)
    with col1:
        misconception_substr_filter = st.text_input(label="Misconception 部分文字列")
    with col2:
        misconception_number_filter = st.number_input(label="Misconception ID", min_value=0)

    # misconception_filterの文字列を含む行を抽出
    filtered_df = misconception_mapping.filter(pl.col("MisconceptionName").str.contains(f"(?i){misconception_substr_filter}"))
    if misconception_number_filter:
        filtered_df = filtered_df.filter(pl.col("MisconceptionId") >= misconception_number_filter).filter(pl.col("MisconceptionId") <= misconception_number_filter + 30)
    filtered_df = filtered_df[:100]
    st.dataframe(
        filtered_df,
        column_config={
            "MisconceptionName": st.column_config.Column("MisconceptionName", width=1000),
        },
    )


if __name__ == "__main__":
    main()
