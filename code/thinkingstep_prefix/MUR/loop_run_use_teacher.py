import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    #candidates = [4,8,12]
    #verify_nums = [1]
    scaling_rates = [0.9]
    thinking_step_prefix_len = [2,4,8,16,32,64]  # 0 for vanilla CoT
    iterations = [1,2,3]

    # /home/pw58/efficient_reasoning/MUR/res/{file_name}.json
    RES_ROOT = Path("thinking_step_prefix")

    SCRIPT = "code/thinkingstep_prefix/MUR/use_teacher_model_thinkingstep_prefix.py"

    failures = []

    for i in iterations:
        for prefix_len in thinking_step_prefix_len:
            for scale in scaling_rates:
                rel_file_stem = f"use_teacher_model-mur/gpqa_diamond/{i}/preifx_len_{prefix_len}"
                out_json = RES_ROOT / f"{rel_file_stem}.json"

                # Ensure the parent directory exists before running
                out_json.parent.mkdir(parents=True, exist_ok=True)

                # Skip if already completed (handy for resuming)
                if out_json.exists():
                    print(f"[skip] already exists: {out_json}")
                    continue

                cmd = [
                    "python", SCRIPT,
                    "--data_path", "data/gpqa_diamond_test.json",
                    "--thinking_step_prefix_length", str(prefix_len),
                    "--scaling_rate", str(scale),
                    "--aim_gpu", str(2),  # keep your current behavior
                    "--file_name", rel_file_stem,  # script will add /res/ prefix and .json
                ]

                print("Running:", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[fail] exit {e.returncode}: {' '.join(cmd)}")
                    failures.append((rel_file_stem, e.returncode))
                    # continue to next combo

    if failures:
        print("\nSummary of failures:")
        for stem, code in failures:
            print(f" - {stem} (exit {code})")
    else:
        print("\nAll runs completed successfully.")