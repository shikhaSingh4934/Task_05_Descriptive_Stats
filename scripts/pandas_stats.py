import pandas as pd
import ast
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import chardet

# ====== Encoding Detection ======
def detect_encoding(filepath, fallback='utf-8'):
    with open(filepath, 'rb') as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    enc = result['encoding'] or fallback
    print(f"🔍 Detected encoding: {enc}")
    if enc.lower() == 'ascii':
        enc = 'utf-8'
    return enc

# ====== Unpacking Logic ======
def detect_unpackable_columns_df(df, sample_size=5):
    unpackable = []
    for col in df.columns:
        samples = df[col].dropna().astype(str).head(sample_size)
        for val in samples:
            try:
                parsed = ast.literal_eval(val.strip())
                if isinstance(parsed, dict) and all(isinstance(v, dict) for v in parsed.values()):
                    unpackable.append(col)
                    break
            except:
                continue
    return unpackable

def unpack_nested_column_df(df, col, prefix):
    unpacked_rows = []
    for _, row in df.iterrows():
        raw_value = row[col]
        try:
            nested_dict = ast.literal_eval(str(raw_value).strip())
        except:
            continue
        for subkey, subvals in nested_dict.items():
            if not isinstance(subvals, dict):
                continue
            new_row = row.to_dict()
            new_row[f"{prefix}_key"] = subkey
            for k, v in subvals.items():
                new_row[f"{prefix}_{k}"] = v
            unpacked_rows.append(new_row)
    return pd.DataFrame(unpacked_rows)

# ====== Dataset Setup ======
def get_dataset_info_pandas():
    print("\n📂 Dataset Setup")
    path = input("Enter dataset path (e.g., ./datasets/Superstore.csv): ").strip()
    group_input = input("Enter group keys (comma-separated, or leave blank for none): ").strip()
    group_keys = [k.strip() for k in group_input.split(",") if k.strip()]

    print("⚡ Loading and preprocessing data...")
    encoding = detect_encoding(path)
    df = pd.read_csv(path, encoding=encoding).dropna(how="all")
    # Detect and unpack nested columns
    unpack_cols = detect_unpackable_columns_df(df)
    for col in unpack_cols:
        prefix = col.split('_')[0]
        print(f"🔓 Unpacking column {col} with prefix '{prefix}'")
        df = unpack_nested_column_df(df, col, prefix)
        df = df.drop(columns=[col])

    dataset_name = os.path.splitext(os.path.basename(path))[0]
    return {
        "path": path,
        "df": df,
        "group_keys": group_keys,
        "dataset_name": dataset_name
    }

# ====== Visualization Setup ======
def get_visualization_info_pandas(df):
    wants_viz = input("\n🎨 Do you want to generate visualizations for this dataset? (y/n): ").strip().lower()
    numeric_cols, categorical_cols = [], []

    if wants_viz != 'y':
        return numeric_cols, categorical_cols

    print("🔍 Sampling data to suggest column types...")
    sample_df = df.head(50)
    numeric_suggestions = [col for col in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[col])]
    categorical_suggestions = [col for col in sample_df.columns if col not in numeric_suggestions]

    print(f"Suggested numeric columns: {numeric_suggestions}")
    print(f"Suggested categorical columns: {categorical_suggestions}")

    num_input = input("Enter numeric columns to plot (comma-separated or leave blank to use suggested): ").strip()
    cat_input = input("Enter categorical columns to plot (comma-separated or leave blank to use suggested): ").strip()

    numeric_cols = [c.strip() for c in num_input.split(",") if c.strip()] if num_input else numeric_suggestions
    categorical_cols = [c.strip() for c in cat_input.split(",") if c.strip()] if cat_input else categorical_suggestions

    return numeric_cols, categorical_cols

# ====== Summary Logic ======
def summarize_dataframe(df, group_label):
    summary = []
    for col in df.columns:
        s = df[col].dropna()
        count = s.count()
        unique = s.nunique()
        mean_val = min_val = max_val = std_val = "NA"
        most_freq = "NA"
        if pd.api.types.is_numeric_dtype(s):
            mean_val = round(s.mean(), 4)
            min_val = s.min()
            max_val = s.max()
            std_val = round(s.std(), 4)
        else:
            if not s.empty:
                mode = s.mode()
                if not mode.empty:
                    mf_val = mode.iloc[0]
                    mf_count = s.value_counts().iloc[0]
                    most_freq = f"{mf_val} ({mf_count})"
        summary.append({
            "column": col,
            "count": count,
            "unique": unique,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std_dev": std_val,
            "most_freq": most_freq,
            "group": group_label,
        })
    return summary

# ====== Plotting ======
def generate_visualizations(df, numeric_cols, categorical_cols, output_prefix):
    os.makedirs("pandas_plots", exist_ok=True)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        try:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if values.empty:
                continue
            plt.figure(figsize=(8, 4))
            plt.hist(values, bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"pandas_plots/{output_prefix}_hist_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Histogram failed for {col}: {e}")

    for col in categorical_cols:
        if col not in df.columns:
            continue
        try:
            freq = df[col].value_counts().head(10)
            if freq.empty:
                continue
            plt.figure(figsize=(8, 4))
            freq.plot(kind='bar', color='lightgreen', edgecolor='black')
            plt.title(f"Top Categories: {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"pandas_plots/{output_prefix}_bar_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Bar chart failed for {col}: {e}")

# ====== Main ======
if __name__ == "__main__":
    config = get_dataset_info_pandas()
    df = config["df"]
    group_keys = config["group_keys"]
    dataset_name = config["dataset_name"]

    numeric_cols, categorical_cols = get_visualization_info_pandas(df)

    print(f"\n🔍 Analyzing: {dataset_name} with grouping by {group_keys or 'None'}...")
    start = time.perf_counter()

    all_summaries = []
    full_summary = summarize_dataframe(df, group_label="full_dataset")
    all_summaries.extend(full_summary)

    if not group_keys:
        print("ℹ️ No grouping keys specified, skipping group summaries.")
    else:
        grouped = df.groupby(group_keys)
        for group_vals, group_df in grouped:
            label = ", ".join(f"{k}={v}" for k, v in zip(group_keys, group_vals)) if isinstance(group_vals, tuple) else f"{group_keys[0]}={group_vals}"
            group_summary = summarize_dataframe(group_df, group_label=label)
            all_summaries.extend(group_summary)

    os.makedirs("pandas_summaries", exist_ok=True)
    summary_df = pd.DataFrame(all_summaries)
    output_file = f"pandas_summaries/pandas_summary_{dataset_name}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"✅ Summary saved to {output_file}")

    if numeric_cols or categorical_cols:
        print("🎨 Generating visualizations...")
        generate_visualizations(df, numeric_cols, categorical_cols, output_prefix=dataset_name)
        print("✅ Plots saved in /pandas_plots folder.")

    print(f"⏱️ Time elapsed: {time.perf_counter() - start:.2f} seconds")
