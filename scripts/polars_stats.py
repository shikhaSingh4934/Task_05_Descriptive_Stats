import polars as pl
import pandas as pd
import ast
import time
import os
import chardet
import matplotlib.pyplot as plt

# ====== Encoding Detection ======
def detect_encoding(filepath, n_lines=1000):
    with open(filepath, 'rb') as f:
        raw_data = b''.join([f.readline() for _ in range(n_lines)])
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    confidence = result['confidence'] * 100
    print(f"🔍 Detected encoding: {encoding} (Confidence: {confidence:.1f}%)")
    return encoding

# ====== Unpacking Logic ======
def detect_unpackable_columns(df, sample_size=5):
    unpackable_cols = []
    for col in df.columns:
        col_vals = df[col].limit(sample_size).to_list()
        for val in col_vals:
            try:
                if isinstance(val, str):
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, dict) and all(isinstance(v, dict) for v in parsed.values()):
                        unpackable_cols.append(col)
                        break
            except:
                continue
    return unpackable_cols

def unpack_nested_column(df, column_name, prefix):
    unpacked = []
    for row in df.iter_rows(named=True):
        try:
            if row[column_name] is not None and row[column_name] != "":
                nested = ast.literal_eval(row[column_name])
            else:
                nested = {}
            for key, val in nested.items():
                new_row = dict(row)
                new_row[f"{prefix}_key"] = key
                # Unpack all keys from val dict, dynamically like pandas
                if isinstance(val, dict):
                    for k, v in val.items():
                        new_row[f"{prefix}_{k}"] = v
                else:
                    # If val not dict, maybe save as is or ignore
                    new_row[f"{prefix}_value"] = val
                unpacked.append(new_row)
        except:
            continue
    return pl.DataFrame(unpacked)

# ========== User Interaction ==========
def get_dataset_info_polars():
    print("\n📂 Dataset Setup")
    path = input("Enter dataset path (e.g., ./datasets/data.csv): ").strip()
    group_input = input("Enter group keys (comma-separated, or leave blank for none): ").strip()
    group_keys = [k.strip() for k in group_input.split(",") if k.strip()]

    print("⚡ Loading data and checking for nested structures...")
    df = read_csv_with_encoding(path)
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))  # Drop all-null rows
    unpack_cols = detect_unpackable_columns(df)
    for col in unpack_cols:
        prefix = col.split('_')[0]
        print(f"🔓 Unpacking column '{col}' with prefix '{prefix}'")
        df = unpack_nested_column(df, col, prefix)
        df = df.drop(col)

    dataset_name = os.path.splitext(os.path.basename(path))[0]
    return {
        "df": df,
        "dataset_name": dataset_name,
        "group_keys": group_keys
    }

def get_visualization_info_polars(df: pl.DataFrame):
    wants_viz = input("\n🎨 Do you want to generate visualizations for this dataset? (y/n): ").strip().lower()
    numeric_cols = []
    categorical_cols = []

    if wants_viz != 'y':
        return numeric_cols, categorical_cols

    print("🔍 Analyzing column types for suggestions...")
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col].dtype)]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    print(f"Suggested numeric columns: {numeric_cols}")
    print(f"Suggested categorical columns: {categorical_cols}")

    num_input = input("Enter numeric columns to plot (comma-separated or leave blank to use suggested): ").strip()
    cat_input = input("Enter categorical columns to plot (comma-separated or leave blank to use suggested): ").strip()

    numeric_cols = [c.strip() for c in num_input.split(",") if c.strip()] if num_input else numeric_cols
    categorical_cols = [c.strip() for c in cat_input.split(",") if c.strip()] if cat_input else categorical_cols

    return numeric_cols, categorical_cols

# ====== Summary Logic ======
def summarize_polars_df(df: pl.DataFrame, group_label):
    summary = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        count = series.len() - series.null_count()
        unique = series.n_unique()
        mean_val = min_val = max_val = std_dev = most_freq = "NA"

        if is_numeric_dtype(dtype):
            mean_raw = series.mean()
            if mean_raw is not None:
                mean_val = round(mean_raw, 4)
            min_val = series.min()
            max_val = series.max()
            std_raw = series.std()
            if std_raw is not None:
                std_dev = round(std_raw, 4)
        else:
            try:
                vc_df = (df
                         .select(pl.col(col).value_counts().alias("vc"))
                         .unnest("vc")
                         .sort("count", descending=True))
                if vc_df.height > 0:
                    top_val = vc_df[col][0]
                    top_count = vc_df["count"][0]
                    most_freq = f"{top_val} ({top_count})"
            except Exception:
                most_freq = "NA"

        summary.append({
            "column": col,
            "count": count,
            "unique": unique,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std_dev": std_dev,
            "most_freq": most_freq,
            "group": group_label
        })
    return summary

# ====== Plotting ======
def generate_visualizations_pandas(df, numeric_cols, categorical_cols, output_prefix):
    os.makedirs("polars_plots", exist_ok=True)

    # Histogram for numeric columns
    for col in numeric_cols:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found for histogram.")
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
            plt.savefig(f"polars_plots/{output_prefix}_hist_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to plot histogram for {col}: {e}")

    # Bar chart for top categorical values
    for col in categorical_cols:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found for bar chart.")
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
            plt.savefig(f"polars_plots/{output_prefix}_bar_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to plot bar chart for {col}: {e}")

# ========== Core Helpers ==========
NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
}

def is_numeric_dtype(dtype):
    return dtype in NUMERIC_DTYPES


def read_csv_with_encoding(file_path):
    try:
        encoding = detect_encoding(file_path)

        # 🔁 Force fallback if ascii is wrongly detected
        if encoding.lower() == 'ascii':
            print("⚠️ Detected ASCII, but this may be incorrect. Forcing UTF-8 fallback...")
            encoding = 'utf-8'

        return pl.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"❌ Failed to read '{file_path}' with encoding '{encoding}'. Error: {e}")
        raise



# ========== Main Execution ==========

if __name__ == "__main__":
    config = get_dataset_info_polars()
    df = config["df"]
    dataset_name = config["dataset_name"]
    group_keys = config["group_keys"]

    numeric_cols, categorical_cols = get_visualization_info_polars(df)

    print(f"\n🔍 Summarizing: {dataset_name} (Grouped by {group_keys or 'None'})...")
    start = time.perf_counter()
    all_summary_rows = []

    # Full dataset summary
    full_summary = summarize_polars_df(df, group_label="full_dataset")
    all_summary_rows.extend(full_summary)

    # Grouped summaries
    if not group_keys:
        print("ℹ️ No grouping keys provided. Skipping grouped summaries.")
    else:
        try:
            grouped_df = df.groupby(group_keys)
            for group_vals, group_df in grouped_df:
                label = ", ".join(f"{k}={v}" for k, v in zip(group_keys, group_vals)) if isinstance(group_vals, tuple) else f"{group_keys[0]}={group_vals}"
                group_summary = summarize_polars_df(group_df, group_label=label)
                all_summary_rows.extend(group_summary)
        except Exception as e:
            print(f"⚠️ Error grouping by keys {group_keys}: {e}")

  
    # Ensure output folder exists
    os.makedirs("polars_summaries", exist_ok=True)

# Save summary CSV for all datasets processed
    if all_summary_rows:
        output_df = pl.DataFrame(all_summary_rows)
        output_file = f"polars_summaries/polars_summary_{dataset_name}.csv"
        output_df.write_csv(output_file)
        print(f"\n✅ Summary saved to '{output_file}'")
    else:
        print("⚠️ No summary rows generated.")

        # Generate visualizations if requested
    if numeric_cols or categorical_cols:
        print("🎨 Generating visualizations...")
        df_pandas = df.to_pandas()
        generate_visualizations_pandas(df_pandas, numeric_cols, categorical_cols, output_prefix=dataset_name)
        print("✅ Plots saved in /polars_plots folder.")

    print(f"⏱️ Time elapsed: {time.perf_counter() - start:.2f} seconds")
