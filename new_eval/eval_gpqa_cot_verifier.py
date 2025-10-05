import os
import csv
import json
from collections import defaultdict
import re

def getAnswer(response):
    """Extract answer from response text"""
    if "\\boxed" in response[-50:]:
        pred = response.split("\\boxed")[-1]
    else:
        pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

def evaluate_file(file_path):
    """Evaluate a single JSON file and return accuracy"""
    try:
        with open(file_path) as file:
            prediction = json.load(file)
    except:
        return 0.0
    
    correct_num = 0
    total_evaluated = 0
    
    for i in range(len(prediction)):
        if "final_answer" in prediction[i]:
            response = prediction[i]['final_answer']
            pred = getAnswer(response)
            gt = prediction[i]['ground_truth']
            try:
                if gt == pred:
                    correct_num += 1
                total_evaluated += 1
            except:
                continue
                
        elif "all_answers" in prediction[i]:
            response = prediction[i]
            gt = prediction[i]['ground_truth']
            pred = getAnswer(response['all_answers'][0])
            if pred == gt:
                correct_num += 1
            total_evaluated += 1
            
        else:
            response = prediction[i]
            gt = prediction[i]['ground_truth']
            if "all_answers" in response:
                all_ans = [0, 0, 0, 0]
                for each in response["all_answers"]:
                    char = getAnswer(each)
                    if char == "A":
                        all_ans[0] += 1
                    elif char == "B":
                        all_ans[1] += 1
                    elif char == "C":
                        all_ans[2] += 1
                    elif char == "D":
                        all_ans[3] += 1
                
                if gt == "A":
                    num_gt = 0
                elif gt == "B":
                    num_gt = 1
                elif gt == "C":
                    num_gt = 2
                elif gt == "D":
                    num_gt = 3
                
                max_ans = max(all_ans)
                if all_ans.index(max_ans) == num_gt:
                    correct_num += 1
                total_evaluated += 1
    
    accuracy = correct_num / total_evaluated if total_evaluated > 0 else 0.0
    return accuracy

def extract_prefix_len(filename):
    """Extract prefix_len value from filename"""
    # Look for patterns like "prefix_len_200", "preifx_len_200", etc.
    match = re.search(r'pre?i?fx_len_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    # Base path
    base_path = "/home/pw58/efficient_reasoning/MUR/thinking_step_prefix/llm_as_a_critic-per_step_scale/gpqa_diamond/4B"
    
    # Dictionary to store results: {prefix_len: {folder: accuracy}}
    results = defaultdict(dict)
    
    # Process folders 1, 2, 3
    for folder in [1, 2, 3]:
        folder_path = os.path.join(base_path, str(folder))
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping.")
            continue
            
        # Process all JSON files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            
            # Extract prefix_len from filename
            prefix_len = extract_prefix_len(json_file)
            if prefix_len is None:
                print(f"Could not extract prefix_len from {json_file}, skipping.")
                continue
            
            # Evaluate the file
            accuracy = evaluate_file(file_path)
            
            # Store the result
            results[prefix_len][folder] = accuracy
            
            print(f"Processed {json_file}: prefix_len={prefix_len}, folder={folder}, accuracy={accuracy:.4f}")
    
    # Create output CSV
    output_file = f"{base_path}/analyze.csv"
    
    # Get all prefix_len values and sort them
    all_prefix_lens = sorted(results.keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['prefix_len', '1', '2', '3'])
        
        # Write data rows
        for prefix_len in all_prefix_lens:
            row = [prefix_len]
            for folder in [1, 2, 3]:
                accuracy = results[prefix_len].get(folder, '')
                if accuracy != '':
                    row.append(f"{accuracy:.4f}")
                else:
                    row.append('')
            writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(all_prefix_lens)} different prefix_len values")
    
    # Print summary
    print("\nSummary:")
    print("prefix_len\t1\t2\t3")
    for prefix_len in all_prefix_lens:
        row_data = [str(prefix_len)]
        for folder in [1, 2, 3]:
            accuracy = results[prefix_len].get(folder, '')
            if accuracy != '':
                row_data.append(f"{accuracy:.4f}")
            else:
                row_data.append('N/A')
        print('\t'.join(row_data))

if __name__ == "__main__":
    main()