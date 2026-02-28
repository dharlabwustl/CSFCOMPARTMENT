import pandas as pd

def add_matched_column_from_large_csv(
        small_csv_path: str,
        large_csv_path: str,
        output_csv_path: str,
        *,
        small_match_col: str,
        large_match_col: str,
        large_value_col: str,
        new_col_name: str = "matched_value",
        case_sensitive: bool = True,
):
    """
    From small CSV and large CSV:
      - Match small_df[small_match_col] with large_df[large_match_col]
      - Copy large_df[large_value_col] into a new column in small named new_col_name
      - Save to output_csv_path

    Returns output_csv_path.
    """

    small_df = pd.read_csv(small_csv_path)
    large_df = pd.read_csv(large_csv_path)

    # Validate columns
    for c in (small_match_col,):
        if c not in small_df.columns:
            raise ValueError(f"Column not found in small CSV: {c}")
    for c in (large_match_col, large_value_col):
        if c not in large_df.columns:
            raise ValueError(f"Column not found in large CSV: {c}")

    # Work on minimal subset of large for speed/memory
    large_subset = large_df[[large_match_col, large_value_col]].copy()

    # Optional: case-insensitive matching
    if not case_sensitive:
        small_df[small_match_col] = small_df[small_match_col].astype(str).str.lower()
        large_subset[large_match_col] = large_subset[large_match_col].astype(str).str.lower()

    # Left join
    merged = small_df.merge(
        large_subset,
        how="left",
        left_on=small_match_col,
        right_on=large_match_col,
    )

    # Rename copied column and drop the extra join column from large
    merged.rename(columns={large_value_col: new_col_name}, inplace=True)
    merged.drop(columns=[large_match_col], inplace=True)

    merged.to_csv(output_csv_path, index=False)
    return output_csv_path

add_matched_column_from_large_csv(
    small_csv_path='VNS_study_to_fix_02282026.csv',
    large_csv_path="input.csv",
    output_csv_path="small_with_matches.csv",
    small_match_col="snipr.session",
    large_match_col="label",
    large_value_col="ID",
    new_col_name="matched_session_id",
)
# add_matched_column_from_large_csv('VNS_study_to_fix_02282026.csv','input.csv','output_csv_path.csv',small_match_col='snipr.session')