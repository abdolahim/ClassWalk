import pandas as pd

from classwalk.handler import open_cleaned_table


def isic31_to_isic4_calculate_transition_weights(
    number_of_years: int = 3,
    constant_weight: float = 1,
) -> pd.Series:
    table = open_cleaned_table("isic31_ir_to_isic4_ir")
    transition_matrix = (
        table
        .dropna(thresh=2)
        .assign(value=1)
        .set_index(["ISIC31_Code", "ISIC4_Code"])
        .loc[:, "value"]
        .unstack(-1, 0)
    )

    isic31_weight = (
        pd.read_csv(
            "classwalk/internal_data/isic31_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.index)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1391 - y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    isic4_weight = (
        pd.read_csv(
            "classwalk/internal_data/isic4_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.columns)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1392 + y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    transition_matrix_iterations: list[pd.DataFrame] = [
        transition_matrix
        .div(transition_matrix.sum(axis="columns"), axis="index")
        .mul(100)
    ]

    residual = 1
    while residual > 0.001:
        last_tm = transition_matrix_iterations[-1]
        new_tm = (
            last_tm
            .mul(isic31_weight.div(last_tm.sum("columns")), axis="index")
            .mul(isic4_weight.div(last_tm.sum("index")), axis="columns")
        )
        new_tm = new_tm.div(new_tm.sum(axis="columns"), axis="index").mul(100)
        transition_matrix_iterations.append(new_tm)
        residual = new_tm.sub(last_tm).abs().sum().sum()
        print(residual, end="\r")

    transition_weights = (
        transition_matrix_iterations[-1]
        .stack()
        .round(2)
        .loc[lambda s: s.gt(0)]
        .rename("Weight") # type: ignore
    )
    assert isinstance(transition_weights, pd.Series)

    transition_weights = transition_weights.rename("Weight")

    return transition_weights


def isic3_to_isic31_calculate_transition_weights(
    number_of_years: int = 3,
    constant_weight: float = 1,
) -> pd.Series:
    table = open_cleaned_table("isic3_ir_to_isic31_ir")
    transition_matrix = (
        table
        .dropna(thresh=2)
        .assign(value=1)
        .set_index(["ISIC31_Code", "ISIC4_Code"])
        .loc[:, "value"]
        .unstack(-1, 0)
    )

    isic31_weight = (
        pd.read_csv(
            "classwalk/internal_data/isic31_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.index)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1391 - y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    isic4_weight = (
        pd.read_csv(
            "classwalk/internal_data/isic4_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.columns)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1392 + y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    transition_matrix_iterations: list[pd.DataFrame] = [
        transition_matrix
        .div(transition_matrix.sum(axis="columns"), axis="index")
        .mul(100)
    ]

    residual = 1
    while residual > 0.001:
        last_tm = transition_matrix_iterations[-1]
        new_tm = (
            last_tm
            .mul(isic31_weight.div(last_tm.sum("columns")), axis="index")
            .mul(isic4_weight.div(last_tm.sum("index")), axis="columns")
        )
        new_tm = new_tm.div(new_tm.sum(axis="columns"), axis="index").mul(100)
        transition_matrix_iterations.append(new_tm)
        residual = new_tm.sub(last_tm).abs().sum().sum()
        print(residual, end="\r")

    transition_weights = (
        transition_matrix_iterations[-1]
        .stack()
        .round(2)
        .loc[lambda s: s.gt(0)]
        .rename("Weight") # type: ignore
    )
    assert isinstance(transition_weights, pd.Series)

    transition_weights = transition_weights.rename("Weight")

    return transition_weights


def isco88_to_isco08_calculate_transition_weights(
    number_of_years: int = 3,
    constant_weight: float = 1,
) -> pd.Series:
    table = open_cleaned_table("isco88_to_isco08_ir")
    transition_matrix = (
        table
        .drop_duplicates()
        .assign(value=1)
        .set_index(["ISCO88_Code", "ISCO08_Code"])
        .loc[:, "value"]
        .unstack(-1, 0)
    )

    isco88_weight = (
        pd.read_csv(
            "classwalk/internal_data/isco88_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.index)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1395 - y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    isco08_weight = (
        pd.read_csv(
            "classwalk/internal_data/isco08_employment_share_lfs.csv",
            low_memory=False,
            dtype={"Code": str}
        )
        .set_index("Code")
        .reindex(index=transition_matrix.columns)
        .fillna(0)
        .assign(
            Weight=lambda df:
            df[[str(1396 + y) for y in range(number_of_years)]]
            .mean(axis="columns")
        )
        .assign(
            Weight=lambda df:
            df["Weight"]
            .div(df["Weight"].sum()).mul(100 - constant_weight)
            .add(df["Weight"].div(len(df.index)).div(100).mul(constant_weight).sum())
        )
        .loc[:, "Weight"]
    )


    transition_matrix_iterations: list[pd.DataFrame] = [
        transition_matrix
        .div(transition_matrix.sum(axis="columns"), axis="index")
        .mul(100)
    ]

    residual = 1
    while residual > 0.001:
        last_tm = transition_matrix_iterations[-1]
        new_tm = (
            last_tm
            .mul(isco88_weight.div(last_tm.sum("columns")), axis="index")
            .mul(isco08_weight.div(last_tm.sum("index")), axis="columns")
        )
        new_tm = new_tm.div(new_tm.sum(axis="columns"), axis="index").mul(100)
        transition_matrix_iterations.append(new_tm)
        residual = new_tm.sub(last_tm).abs().sum().sum()
        print(residual, end="\r")

    transition_weights = (
        transition_matrix_iterations[-1]
        .stack()
        .round(2)
        .loc[lambda s: s.gt(0)]
        .rename("Weight") # type: ignore
    )
    assert isinstance(transition_weights, pd.Series)

    transition_weights = transition_weights.rename("Weight")

    return transition_weights