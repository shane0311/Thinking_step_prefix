#!/usr/bin/env python3
import argparse, json, re, csv, os
from typing import Tuple, Optional, Dict, List
from collections import defaultdict

MC_CHOICES = {"a", "b", "c", "d"}
YN_CHOICES = {"yes", "no", "true", "false"}

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
THE_ANSWER_IS_RE = re.compile(r"the answer is[:\s]*", re.IGNORECASE)

def normalize(s: str) -> str:
    return s.strip()

def only_letters_digits(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-\+\.]", "", s)

def extract_after_phrase(text: str) -> Optional[str]:
    """Grab text right after 'the answer is ...' to end-of-line or sentence."""
    m = THE_ANSWER_IS_RE.search(text)
    if not m: 
        return None
    after = text[m.end():].strip()
    # Stop at newline or obvious stop tokens
    after = re.split(r"(\n|</?output>|</?final>|</?answer>|Step\d+:)", after)[0].strip()
    return after

def extract_boxed(text: str) -> Optional[str]:
    m = BOXED_RE.search(text)
    return m.group(1).strip() if m else None

def first_choice_letter(text: str) -> Optional[str]:
    """
    Try to find a clear A/B/C/D in the final span:
    - standalone letter
    - '(C)', 'option C', 'answer: C', etc.
    """
    # Check boxed first (sometimes \boxed{C})
    bx = extract_boxed(text)
    if bx:
        bx_clean = bx.strip().lower()
        if bx_clean in MC_CHOICES:
            return bx_clean.upper()

    # Look right after 'the answer is'
    after = extract_after_phrase(text) or text

    # Common patterns
    m = re.search(r"\b([ABCD])\b", after, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\boption\s*([ABCD])\b", after, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # '(C)' etc.
    m = re.search(r"\(\s*([ABCD])\s*\)", after, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None

def yes_no_extract(text: str) -> Optional[str]:
    bx = extract_boxed(text)
    if bx and bx.strip().lower() in YN_CHOICES:
        v = bx.strip().lower()
        return "Yes" if v in {"yes", "true"} else "No"

    after = extract_after_phrase(text) or text
    # Look for yes/no keywords
    m = re.search(r"\b(yes|no|true|false)\b", after, flags=re.IGNORECASE)
    if m:
        v = m.group(1).lower()
        return "Yes" if v in {"yes", "true"} else "No"
    return None

def numeric_extract(text: str) -> Optional[str]:
    # Priority: \boxed{...}
    bx = extract_boxed(text)
    if bx:
        # Keep possible fractions like 3/7
        mfrac = re.fullmatch(r"\s*[-+]?\d+(?:/\d+)?\s*", bx)
        if mfrac:
            return mfrac.group(0).strip()
        # Else a float/int
        mnum = re.search(r"[-+]?\d+(?:\.\d+)?", bx)
        if mnum:
            return mnum.group(0)

    # Then try after 'the answer is'
    after = extract_after_phrase(text) or text
    # AIME-style integers or general numbers/fractions
    m = re.search(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", after)
    if m:
        return m.group(0)
    return None

def looks_mc(gt: str) -> bool:
    return gt.strip().upper() in {"A","B","C","D"}

def looks_yn(gt: str) -> bool:
    return gt.strip().lower() in {"yes","no","true","false"}

def is_fraction(s: str) -> bool:
    return re.fullmatch(r"[-+]?\d+/\d+", s.strip()) is not None

def frac_to_float(s: str) -> float:
    n,d = s.split("/")
    return float(int(n))/float(int(d))

def numeric_equal(pred: str, gt: str, atol: float, rtol: float) -> bool:
    """Compare integers, floats, or fractions with tolerances."""
    pred = pred.strip()
    gt = gt.strip()

    # Exact string match first
    if pred == gt:
        return True

    def to_float(x: str) -> Optional[float]:
        try:
            if is_fraction(x): return frac_to_float(x)
            return float(x)
        except:
            return None

    a, b = to_float(pred), to_float(gt)
    if a is None or b is None:
        return False
    return abs(a-b) <= (atol + rtol*abs(b))

def extract_predicted(final_text: str, task_hint: str, gt: str) -> str:
    """
    task_hint: one of {"auto","mc","yn","numeric","math"}
    """
    text = final_text.strip()

    # 1) Try format-specific by hint
    if task_hint == "mc":
        c = first_choice_letter(text)
        if c: return c
    elif task_hint == "yn":
        y = yes_no_extract(text)
        if y: return y
    elif task_hint in {"numeric","math"}:
        n = numeric_extract(text)
        if n: return n

    # 2) Auto fallback by ground truth
    if looks_mc(gt):
        c = first_choice_letter(text)
        if c: return c
    if looks_yn(gt):
        y = yes_no_extract(text)
        if y: return y

    # 3) Generic catch-alls
    bx = extract_boxed(text)
    if bx: 
        return bx.strip()

    after = extract_after_phrase(text)
    if after:
        return after.strip().rstrip(".")
    
    # Last resort: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def compare(pred: str, gt: str, task_hint: str, atol: float, rtol: float) -> bool:
    if looks_mc(gt):
        return pred.strip().upper() == gt.strip().upper()
    if looks_yn(gt):
        # Normalize truthy to Yes/No
        g = gt.strip().lower()
        g = "Yes" if g in {"yes","true"} else "No"
        p = pred.strip().lower()
        p = "Yes" if p in {"yes","true"} else "No"
        return p == g
    # numeric-ish
    # If either looks numeric/fraction, try numeric compare; else exact normalized
    if re.search(r"[-+]?\d", pred) or re.search(r"[-+]?\d", gt) or "/" in pred or "/" in gt:
        return numeric_equal(pred, gt, atol, rtol)
    return normalize(pred).lower() == normalize(gt).lower()

def extract_param_from_filename(filename: str, param_name: str) -> Optional[str]:
    """Extract parameter value from filename using various patterns"""
    patterns = [
        rf'{param_name}_(\w+)',
        rf'{param_name}(\d+)',
        rf'{param_name}_(\d+)',
        rf'{param_name}[-_](\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def evaluate_single_file(path: str, task: str, atol: float, rtol: float) -> Tuple[float, int, int]:
    """Evaluate a single file and return (accuracy, correct, total)"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0.0, 0, 0

    total = len(data)
    correct = 0

    for i, ex in enumerate(data):
        gt = str(ex.get("ground_truth", "")).strip()
        final_text = str(ex.get("final_answer", ex.get("current_traj","")))

        pred = extract_predicted(final_text, task, gt)
        ok = compare(pred, gt, task, atol, rtol)
        correct += int(ok)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def evaluate_batch(base_path: str, task: str, atol: float, rtol: float, 
                  param_name: str, folders: List[str], save_json: str, save_csv: str):
    """Evaluate multiple files and create structured output"""
    
    # Dictionary to store results: {param_value: {folder: {accuracy, correct, total}}}
    results = defaultdict(dict)
    all_files_processed = []
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping.")
            continue
            
        # Process all JSON files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            
            # Extract parameter from filename
            param_value = extract_param_from_filename(json_file, param_name)
            if param_value is None:
                print(f"Could not extract {param_name} from {json_file}, skipping.")
                continue
            
            # Evaluate the file
            accuracy, correct, total = evaluate_single_file(file_path, task, atol, rtol)
            
            # Store the result
            results[param_value][folder] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'filename': json_file
            }
            
            all_files_processed.append({
                'file_path': file_path,
                'param_value': param_value,
                'folder': folder,
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            })
            
            print(f"Processed {json_file}: {param_name}={param_value}, folder={folder}, accuracy={accuracy:.4f} ({correct}/{total})")
    
    # Save detailed JSON results
    if save_json:
        detailed_results = {
            'summary': dict(results),
            'all_files': all_files_processed,
            'parameters': {
                'base_path': base_path,
                'param_name': param_name,
                'folders': folders,
                'task': task,
                'atol': atol,
                'rtol': rtol
            }
        }
        
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Detailed results saved to {save_json}")
    
    # Save CSV summary
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        
        # Get all parameter values and sort them
        all_param_values = sorted(results.keys(), key=lambda x: (x.isdigit() and int(x), x))
        
        with open(save_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = [param_name] + folders
            writer.writerow(header)
            
            # Write data rows
            for param_value in all_param_values:
                row = [param_value]
                for folder in folders:
                    if folder in results[param_value]:
                        accuracy = results[param_value][folder]['accuracy']
                        row.append(f"{accuracy:.4f}")
                    else:
                        row.append('')
                writer.writerow(row)
        
        print(f"CSV summary saved to {save_csv}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    all_param_values = sorted(results.keys(), key=lambda x: (x.isdigit() and int(x), x))
    
    # Print header
    header = f"{param_name:<15}" + "".join(f"{folder:<10}" for folder in folders)
    print(header)
    print("-" * len(header))
    
    # Print data rows
    for param_value in all_param_values:
        row_data = [f"{param_value:<15}"]
        for folder in folders:
            if folder in results[param_value]:
                accuracy = results[param_value][folder]['accuracy']
                correct = results[param_value][folder]['correct']
                total = results[param_value][folder]['total']
                cell = f"{accuracy:.4f}"
                row_data.append(f"{cell:<10}")
            else:
                row_data.append(f"{'N/A':<10}")
        print("".join(row_data))
    
    print(f"\nProcessed {len(all_param_values)} different {param_name} values across {len(folders)} folders")
    print(f"Total files processed: {len(all_files_processed)}")
    
    return results

def main():
    path = "/Users/shanewang/Documents/Rice_project/final/efficient_reasoning/self_consistency/result/1.7B"
    p = argparse.ArgumentParser(description="Evaluate accuracy from multiple result JSONs with parameter analysis.")
    
    # Single file mode (original functionality)
    p.add_argument("--results", type=str, 
                   help="Path to single results JSON (for single file mode).")
    
    # Batch mode arguments
    p.add_argument("--base_path", type=str, default=path,
                   help="Base path containing folders with JSON files (for batch mode).")
    p.add_argument("--folders", type=str, nargs="+", default=["1", "2", "3"],
                   help="List of folder names to process (default: 1 2 3).")
    p.add_argument("--param_name", type=str, default="aime_2025",
                   help="Parameter name to extract from filenames (default: prefix_len).")
    
    # Common arguments
    p.add_argument("--task", type=str, default="auto",
                   choices=["auto","mc","yn","numeric","math"],
                   help="Hint for answer parsing; 'auto' usually works.")
    p.add_argument("--atol", type=float, default=1e-8, help="Abs tolerance for numeric compare.")
    p.add_argument("--rtol", type=float, default=0.0, help="Rel tolerance for numeric compare.")
    
    # Output arguments
    p.add_argument("--save_json", type=str, default=path+"/analyze.json",
                   help="Path to save detailed results JSON.")
    p.add_argument("--save_csv", type=str, default=path+"/analyze.csv",
                   help="Path to save summary results CSV.")
    p.add_argument("--show_examples", type=int, default=10,
                   help="Show up to N mistakes in stdout (single file mode only).")
    
    args = p.parse_args()
    
    # Determine mode based on arguments
    if args.results:
        # Single file mode (original functionality)
        print("Running in single file mode...")
        
        with open(args.results, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data)
        correct = 0
        rows = []
        mistakes = []

        for i, ex in enumerate(data):
            gt = str(ex.get("ground_truth", "")).strip()
            final_text = str(ex.get("final_answer", ex.get("current_traj","")))

            pred = extract_predicted(final_text, args.task, gt)
            ok = compare(pred, gt, args.task, args.atol, args.rtol)
            correct += int(ok)

            row = {
                "idx": i,
                "ground_truth": gt,
                "predicted": pred,
                "correct": ok,
            }
            rows.append(row)
            if not ok and len(mistakes) < args.show_examples:
                mistakes.append(row)

        acc = correct / total if total else 0.0

        print(f"File: {args.results}")
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {acc:.4f}")

        if mistakes:
            print("\nSample mistakes:")
            for m in mistakes:
                print(f"- idx {m['idx']} | GT={m['ground_truth']} | Pred={m['predicted']}")
        
        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
            with open(args.save_csv, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSaved per-item results to: {args.save_csv}")
    
    elif args.base_path:
        # Batch mode (new functionality)
        print("Running in batch mode...")
        
        if not args.save_json and not args.save_csv:
            # Set default output files
            args.save_json = os.path.join(args.base_path, "detailed_results.json")
            args.save_csv = os.path.join(args.base_path, "summary_results.csv")
        
        evaluate_batch(
            args.base_path, 
            args.task, 
            args.atol, 
            args.rtol,
            args.param_name,
            args.folders,
            args.save_json,
            args.save_csv
        )
    
    else:
        print("Error: Must specify either --results (single file mode) or --base_path (batch mode)")
        return 1

if __name__ == "__main__":
    main()