# generalized_stats.py

import csv
import os
import ast
import time
from collections import defaultdict
from statistics import mean, stdev
from typing import List
import pandas as pd
import chardet
import matplotlib.pyplot as plt


# ========== Encoding Detection ==========

def detect_encoding(filepath, n_lines=1000):
    with open(filepath, 'rb') as f:
        raw_data = b''.join([f.readline() for _ in range(n_lines)])
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    confidence = result['confidence'] * 100
    print(f"🔍 Detected encoding: {encoding} (Confidence: {confidence:.1f}%)")
    return encoding

# ========== Unpacking Logic ==========

def detect_unpackable_columns(data, sample_size=5):
    if not data:
        return []
    columns = data[0].keys()
    unpackable = []
    for col in columns:
        samples = [row.get(col, '') for row in data if col in row][:sample_size]
        for val in samples:
            try:
                val_str = str(val).strip()
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, dict) and all(isinstance(v, dict) for v in parsed.values()):
                    unpackable.append(col)
                    break
            except:
                continue
    return unpackable

def unpack_nested_rows(data, key, prefix):
    unpacked_rows = []
    for row in data:
        raw_value = row.get(key, '')
        try:
            nested_dict = ast.literal_eval(str(raw_value).strip())
        except:
            continue
        for subkey, subvals in nested_dict.items():
            if not isinstance(subvals, dict):
                continue  # ensure value is a dict

            new_row = row.copy()
            new_row[f'{prefix}_key'] = subkey
            for k, v in subvals.items():
                new_row[f'{prefix}_{k}'] = v
            unpacked_rows.append(new_row)
    return unpacked_rows


# ========== Core Helpers ==========
def generate_visualizations(data, numeric_cols, categorical_cols, output_prefix):
    os.makedirs("python_plots", exist_ok=True)

    # Histogram for numeric columns
    for col in numeric_cols:
        try:
            values = [float(row[col]) for row in data if col in row and isinstance(row[col], (int, float, float)) or is_number(str(row[col]))]
            plt.figure(figsize=(8, 4))
            plt.hist(values, bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"python_plots/{output_prefix}_hist_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to plot histogram for {col}: {e}")

    # Bar chart for top categorical values
    for col in categorical_cols:
        try:
            freq = defaultdict(int)
            for row in data:
                val = row.get(col, "NA")
                freq[val] += 1
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            labels, counts = zip(*sorted_freq)

            plt.figure(figsize=(8, 4))
            plt.bar(labels, counts, color='lightgreen', edgecolor='black')
            plt.title(f"Top Categories: {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"python_plots/{output_prefix}_bar_{col}.png")
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to plot bar chart for {col}: {e}")


def is_number(s):
    try:
        float(s.replace(',', '').strip())
        return True
    except:
        return False

def to_number(s):
    return float(s.replace(',', '').strip())

def clean_row(row):
    cleaned = {}
    for key, value in row.items():
        value = value.strip()
        if value == '':
            cleaned[key] = 'NA'
        elif is_number(value):
            cleaned[key] = to_number(value)
        else:
            cleaned[key] = value
    return cleaned

def load_csv(filepath: str) -> List[dict]:
    encoding = detect_encoding(filepath)
    with open(filepath, 'r', encoding=encoding, errors='replace') as f:
        reader = csv.DictReader(f)
        data = [clean_row(row) for row in reader]
    return data


# ====== Summary Logic ======

def summarize_columns(data):
    if not data:
        return []

    columns = data[0].keys()
    summary_rows = []

    for col in columns:
        values = [row[col] for row in data if col in row and row[col] != 'NA']
        count = len(values)
        unique = len(set(values))
        numeric_vals = []

        for val in values:
            try:
                numeric_vals.append(float(val))
            except:
                continue

        n = len(numeric_vals)
        if n > 0:
            mean_val = round(sum(numeric_vals) / n, 4)
            min_val = min(numeric_vals)
            max_val = max(numeric_vals)

            if n > 1:
                mean_val_float = sum(numeric_vals) / n
                variance = sum((x - mean_val_float) ** 2 for x in numeric_vals) / (n - 1)
                std_dev = round(variance ** 0.5, 4)
            else:
                std_dev = 'NA'
        else:
            mean_val = min_val = max_val = std_dev = 'NA'

        row_summary = {
            "column": col,
            "count": count,
            "unique": unique,
            "mean": mean_val,
            "min": min_val,
            "max": max_val,
            "std_dev": std_dev,
        }

        if not numeric_vals:
            freq = defaultdict(int)
            for v in values:
                freq[v] += 1
            if freq:
                most_common = max(freq.items(), key=lambda x: x[1])
                row_summary["most_freq"] = f"{most_common[0]} ({most_common[1]})"
            else:
                row_summary["most_freq"] = "NA"
        else:
            row_summary["most_freq"] = "NA"

        summary_rows.append(row_summary)

    return summary_rows


def group_by_keys(data, keys):
    grouped = defaultdict(list)
    for row in data:
        k = tuple(row.get(key, 'NA') for key in keys)
        grouped[k].append(row)
    return grouped

# ========== User Interaction ==========

def get_dataset_info():
    print("\n📂 Dataset Setup")
    path = input("Enter dataset path (e.g., ./datasets/Superstore.csv): ").strip()
    group_input = input("Enter group keys (comma-separated, or leave blank for none): ").strip()
    group_keys = [k.strip() for k in group_input.split(",") if k.strip()]
    
    # Load and unpack data
    print("⚡ Loading and preprocessing data...")
    data = load_csv(path)
    data = [row for row in data if any(v != 'NA' for v in row.values())]
    # Detect and unpack nested columns
    unpack_cols = detect_unpackable_columns(data)
    for col in unpack_cols:
        prefix = col.split('_')[0]
        data = unpack_nested_rows(data, col, prefix)

    for row in data:
        for col in unpack_cols:
            row.pop(col, None)

    dataset_name = os.path.splitext(os.path.basename(path))[0]

    return {
        "path": path,
        "data": data,
        "group_keys": group_keys,
        "dataset_name": dataset_name
    }

def get_visualization_info(data):
    wants_viz = input("\n🎨 Do you want to generate visualizations for this dataset? (y/n): ").strip().lower()
    numeric_cols, categorical_cols = [], []

    if wants_viz != 'y':
        return numeric_cols, categorical_cols

    print("🔍 Sampling data to suggest column types...")
    sample_data = data[:50]
    sample_keys = sample_data[0].keys()
    numeric_suggestions = []
    cat_suggestions = []

    for k in sample_keys:
        values = [row[k] for row in sample_data if k in row and row[k] != 'NA']
        numeric_vals = [v for v in values if isinstance(v, (int, float))]
        if values and len(numeric_vals) / len(values) >= 0.8:
            numeric_suggestions.append(k)
        else:
            cat_suggestions.append(k)

    print(f"Suggested numeric columns: {numeric_suggestions}")
    print(f"Suggested categorical columns: {cat_suggestions}")

    num_input = input("Enter numeric columns to plot (comma-separated or leave blank to use suggested): ").strip()
    cat_input = input("Enter categorical columns to plot (comma-separated or leave blank to use suggested): ").strip()

    numeric_cols = [c.strip() for c in num_input.split(",") if c.strip()] if num_input else numeric_suggestions
    categorical_cols = [c.strip() for c in cat_input.split(",") if c.strip()] if cat_input else cat_suggestions

    return numeric_cols, categorical_cols

# ========== Main Execution ==========

if __name__ == "__main__":
    dataset_config = get_dataset_info()
    data = dataset_config["data"]
    group_keys = dataset_config["group_keys"]
    dataset_name = dataset_config["dataset_name"]

    numeric_cols, categorical_cols = get_visualization_info(data)

    print(f"\n🔍 Analyzing: {dataset_name} with grouping by {group_keys or 'None'}...")
    start = time.perf_counter()

    all_summaries = []
    full_summary = summarize_columns(data)
    for row in full_summary:
        row["group"] = "full_dataset"
        all_summaries.append(row)

    if group_keys:
        grouped = group_by_keys(data, group_keys)
        for group_key, rows in grouped.items():
            label = ", ".join(f"{k}={v}" for k, v in zip(group_keys, group_key))
            group_summary = summarize_columns(rows)
            for row in group_summary:
                row["group"] = label
                all_summaries.append(row)

    # Create output directory
    os.makedirs("python_summaries", exist_ok=True)

    df_summary = pd.DataFrame(all_summaries)
    output_path = f"python_summaries/summary_{dataset_name}.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")


    if numeric_cols or categorical_cols:
        print("🖼️ Generating visualizations...")
        generate_visualizations(data, numeric_cols, categorical_cols, output_prefix=dataset_name)
        print(f"✅ Plots saved in /python_plots folder.")

    print(f"⏱️ Time elapsed: {time.perf_counter() - start:.2f} sec")
