import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    candidates = [4,8,12]
    verify_nums = [1]
    scaling_rates = [0.9]
    thinking_step_prefix_len = [64]  # 0 for vanilla CoT
    iterations = [3]

    # Where guided_search-mur.py writes results:
    # /home/pw58/efficient_reasoning/MUR/res/{file_name}.json
    RES_ROOT = Path("thinking_step_prefix")

    SCRIPT = "code/thinkingstep_prefix/MUR/guided_search-mur_thinkingstep_prefix.py"

    failures = []

    for i in iterations:
        for cand in candidates:
            for prefix_len in thinking_step_prefix_len:
                for scale in scaling_rates:
                    rel_file_stem = f"guided_search-mur/aime2025/1.7B/cand{cand}/{i}/preifx_len_{prefix_len}"
                    out_json = RES_ROOT / f"{rel_file_stem}.json"

                    # Ensure the parent directory exists before running
                    out_json.parent.mkdir(parents=True, exist_ok=True)

                    # Skip if already completed (handy for resuming)
                    if out_json.exists():
                        print(f"[skip] already exists: {out_json}")
                        continue

                    cmd = [
                        "python3", SCRIPT,
                        "--data_path", "data/aime2025_test.json",
                        "--candidate_num", str(cand),
                        "--verify_num", str(1),
                        "--thinking_step_prefix_length", str(prefix_len),
                        "--scaling_rate", str(scale),
                        "--aim_gpu", str("4,5"),  # keep your current behavior
                        "--policy_gpu", str(4),
                        "--critic_gpu", str(5),
                        "--file_name", rel_file_stem,  # script will add /res/ prefix and .json
                        "--policy", "Qwen/Qwen3-1.7B",
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