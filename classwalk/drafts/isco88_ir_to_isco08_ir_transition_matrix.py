import pandas as pd

from classwalk.handler import open_cleaned_table
from classwalk.drafts.transition_matrix_weight_calculation import isco88_to_isco08_calculate_transition_weights


def main() -> pd.DataFrame:
    table = (
        open_cleaned_table("isco88_to_isco08_ir")
        .join(
            open_cleaned_table("isco88_ir")
            .set_index("Code")
            .loc[:, "Description"]
            .rename("ISCO88_Description"),
            on="ISCO88_Code"
        )
        .join(
            open_cleaned_table("isco08_ir")
            .set_index("Code")
            .loc[:, "Description"]
            .rename("ISCO08_Description"),
            on="ISCO08_Code"
        )
        .loc[:, ["Description", "ISCO88_Code", "ISCO88_Description", "ISCO08_Code", "ISCO08_Description"]]
    )
    table = (
        table
        .join(
            table["ISCO88_Code"].value_counts().astype("Int16").rename("ISCO88_Count"),
            on="ISCO88_Code"
        )
        .join(
            table["ISCO08_Code"].value_counts().astype("Int16").rename("ISCO08_Count"),
            on="ISCO08_Code"
        )
        .assign(
            Even_Weight=lambda df: df["ISCO88_Count"].astype("Float64").pow(-1).mul(100).round(2)
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(1, 1).rename("Weight_1y_1c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(2, 1).rename("Weight_2y_1c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(3, 1).rename("Weight_3y_1c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(1, 5).rename("Weight_1y_5c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(2, 5).rename("Weight_2y_5c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
        .join(
            isco88_to_isco08_calculate_transition_weights(3, 5).rename("Weight_3y_5c"),
            on=["ISCO88_Code", "ISCO08_Code"],
        )
    )
    for col in [
        "Weight_1y_1c",
        "Weight_2y_1c",
        "Weight_3y_1c",
        "Weight_1y_5c",
        "Weight_2y_5c",
        "Weight_3y_5c",
    ]:
        table[col] = table[col].fillna(0).where(table["ISCO88_Code"].notna(), None)

    return table


if __name__ == "__main__":
    result = main()
    result.to_excel("ISCO88_to_ISCO08_Transition_Weights.xlsx", index=False)
    (
        result
        .drop(
            columns=[
                "Description",
                "ISCO88_Description",
                "ISCO08_Description",
            ]
        )
        .dropna(how="all")
        .to_csv("ISCO88_to_ISCO08_Transition_Weights.csv", index=False)
    )
