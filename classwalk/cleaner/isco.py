import pandas as pd


def isco08(raw_table: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_table
        .rename(
            columns={
                "ISCO 08 Code": "Code",
                "Title EN": "Description"
            }
        )
        .loc[:, ["Code", "Description", "Level"]]
        .sort_values("Code")
        .reset_index(drop=True)
    )


def isco88(raw_table: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_table
        .rename(
            columns={
                "ISCO 88 Code": "Code",
                "Title EN": "Description"
            }
        )
        .loc[:, ["Code", "Description", "Level"]]
        .sort_values("Code")
        .reset_index(drop=True)
    )


def isco08_ir(raw_table: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_table
        .set_axis(["Code", "Description"], axis="columns")
        .iloc[1:]
        .drop_duplicates("Code", keep="first")
        .dropna()
        .assign(
            Level = lambda df: df["Code"].str.len()
        )
        .reset_index(drop=True)
    )


def isco88_ir(raw_table: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_table
        .set_axis(["Code", "Description"], axis="columns")
        .iloc[1:]
        .drop_duplicates("Code", keep="first")
        .dropna()
        .assign(
            Level = lambda df: df["Code"].str.len()
        )
        .reset_index(drop=True)
    )

def isco88_to_isco08_ir(raw_table: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_table
        .set_axis(["Description", "ISCO88_Code", "ISCO08_Code", "Partial"], axis="columns")
        .assign(Partial=lambda df: df["Partial"].notna())
        .assign(ISCO88_Code=lambda df: df["ISCO88_Code"].replace({"111": "0111"}))
        .assign(ISCO88_Code=lambda df: df["ISCO88_Code"].replace({"112": "0112"}))
        .assign(ISCO08_Code=lambda df: df["ISCO08_Code"].replace({"410": "0410"}))
        .assign(ISCO08_Code=lambda df: df["ISCO08_Code"].replace({"510": "0510"}))
        .drop(columns="Partial")
        .ffill()
        .drop_duplicates()
    )
