import os
import re
import pandas as pd

# === Configure your experiment roots here ===
# Each of these dirs is expected to have subfolders "1", "2", "3"
base_dirs = [
    "/Users/shanewang/Documents/Rice_project/final/efficient_reasoning/MUR/thinking_step_prefix/time/use_teacher_model-per_step_scale/gpqa_diamond",
    "/Users/shanewang/Documents/Rice_project/final/efficient_reasoning/MUR/thinking_step_prefix/time/use_teacher_model-per_step_scale/aime2025",
]
folders = ["1", "2", "3"]

# --- Robust regexes (case-insensitive, allow spaces/underscores, optional "sec") ---
re_time = re.compile(r"(?i)\btime\s*:\s*([0-9]*\.?[0-9]+)\s*(?:s|sec|secs|seconds)?\b")
re_policy = re.compile(r"(?i)\ball[\s_]*policy[\s_]*output[\s_]*tokens\s*:\s*([\d,]+)")
re_critic = re.compile(r"(?i)\ball[\s_]*critic[\s_]*output[\s_]*tokens\s*:\s*([\d,]+)")

def _to_float(num_str):
    if num_str is None:
        return None
    return float(num_str.replace(",", ""))

def parse_metrics(text):
    time_m = re_time.search(text)
    policy_m = re_policy.search(text)
    critic_m = re_critic.search(text)
    return {
        "time": _to_float(time_m.group(1)) if time_m else None,
        "all_policy_output_tokens": _to_float(policy_m.group(1)) if policy_m else None,
        "all_critic_output_tokens": _to_float(critic_m.group(1)) if critic_m else None,
    }

for base_dir in base_dirs:
    records = []
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            # Skip silently if a run folder doesn't exist
            continue
        for filename in os.listdir(folder_path):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, "r") as f:
                    text = f.read()
            except Exception:
                continue  # skip unreadable files

            metrics = parse_metrics(text)
            row = {"filename": filename, "folder": folder}
            row.update(metrics)
            records.append(row)

    if not records:
        print(f"⚠️ No .txt records found under {base_dir}/(1|2|3). Skipping.")
        continue

    df = pd.DataFrame(records)

    # Average per filename across available folders
    avg_df = (
        df.groupby("filename")[["time", "all_policy_output_tokens", "all_critic_output_tokens"]]
          .mean()
          .reset_index()
    )

    # Ensure exact column order as requested
    avg_df = avg_df[["filename", "time", "all_policy_output_tokens", "all_critic_output_tokens"]]

    # Write CSV next to the runs
    output_path = os.path.join(base_dir, "averages.csv")
    avg_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
