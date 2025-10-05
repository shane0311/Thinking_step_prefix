import os
import json
import numpy as np
import argparse
import time
import sys
import torch

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from utils.generate_prompts import ciritique_last_generation, ciritique_last_generation_math
import random, time

random_seed = int(time.time() * 1000) % (2**31)

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def load_model_and_tokenizer(model_dir, gpu_memory_utilization=0.6):
    model = LLM(model=model_dir, tensor_parallel_size=1, max_model_len=8096*2,
                trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, tokenizer.eos_token


def get_system_prompt(data_path):
    if 'math' in data_path.lower() or 'aime' in data_path.lower():
        return 'You are a helpful math assistant. '
    elif 'reclor' in data_path.lower() or 'gpqa' in data_path.lower() or 'logiqa' in data_path.lower():
        return 'You are a helpful assistant. Here is a question and four candidate answers. You need to reason step by step and choose the most likely answer from the four candidate answers. Answer "A", "B", "C", or "D".'
    elif 'strategyqa' in data_path.lower():
        return 'You are a helpful assistant. After each step, you may receive a feedback from the user, indicating that the previous step is incorrect. You should then revise your solution accordingly. Please answer "Yes" or "No".'
    return ''


def build_policy_input(tokenizer, question, traj, step_idx, stop_token):
    system_prompt = get_system_prompt(args.data_path)
    chat = [{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': f'Q: {question}\nAlways end your solution with the phrase \'the answer is\' followed by your final answer. Start your solution with \'Step{step_idx}:\'\n'}]
    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    input_text = input_text.replace(stop_token, "").strip()
    input_text += '\n'.join(traj) + \
        f'\nStep{step_idx}:' if step_idx > 0 else f'\nStep0:'
    return input_text


def generate_prefix_candidates(policy_model, tokenizer, input_text, args, step_idx):
    """Generate multiple prefix candidates for the current reasoning step"""
    sampling_params = SamplingParams(
        max_tokens=args.thinking_step_prefix_length, 
        temperature=0.6, 
        logprobs=1, 
        n=args.candidate_num,
        stop=None  # Don't stop early for prefixes
    )
    outputs = policy_model.generate(input_text, sampling_params)
    
    prefixes = []
    logps = []
    for output in outputs[0].outputs:
        prefix_text = output.text.strip()
        logp = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
        prefixes.append(prefix_text)
        logps.append(logp)
    
    return prefixes, logps, sum(len(o.token_ids) for o in outputs[0].outputs)


def complete_step_from_prefix(policy_model, tokenizer, input_text, selected_prefix, stop_token, step_idx):
    """Complete the reasoning step from the selected prefix"""
    # Construct input with the selected prefix
    full_input = input_text + selected_prefix
    prefix_tokens = len(tokenizer.encode(selected_prefix))
    sampling_params = SamplingParams(
        max_tokens=2048 - prefix_tokens,  # Adjust max tokens
        temperature=0.6, 
        stop=["Step"], 
        logprobs=1
    )
    outputs = policy_model.generate(full_input, sampling_params)
    
    completion = outputs[0].outputs[0].text.strip()
    full_step = selected_prefix + completion
    logp = outputs[0].outputs[0].cumulative_logprob / (len(outputs[0].outputs[0].token_ids) + 1e-8)
    
    return f"Step{step_idx}: {full_step}", logp, len(outputs[0].outputs[0].token_ids)


def select_best_prefix_candidate(critic_model, tokenizer, question, traj, prefix_candidates, args, stop_token, step_idx):
    """Select the best prefix candidate using the critic model"""
    analyze_inputs = []
    
    # Create temporary full steps from prefixes for critic evaluation
    for prefix in prefix_candidates:
        temp_step = f"Step{step_idx}: {prefix}..."  # Add ellipsis to indicate incomplete
        temp_traj = traj + [temp_step]
        
        if 'math' in args.data_path.lower() or 'aime' in args.data_path.lower():
            critic_prompt_dict = ciritique_last_generation_math(question, temp_traj)
        else:
            critic_prompt_dict = ciritique_last_generation(question, temp_traj)

        critic_input = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': critic_prompt_dict['system_prompt']},
             {'role': 'user', 'content': critic_prompt_dict['user_prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
        analyze_start = f"<analyze>\nLet's analyze the beginning of step {step_idx} step by step: "
        analyze_inputs.append(critic_input.replace(stop_token, "").strip() + analyze_start)

    sampling_params = SamplingParams(
        max_tokens=1024, 
        temperature=0.6,
        stop=['</analyze>\n', '```python'],
        include_stop_str_in_output=True, 
        n=args.verify_num
    )
    analyze_outputs = critic_model.generate(analyze_inputs, sampling_params)

    output_inputs = []
    output_start = "<output>\n**Judgement**: $\\boxed"
    for idx, out in enumerate(analyze_outputs):
        for result in out.outputs:
            analyze_text = result.text.strip()
            output_inputs.append(analyze_inputs[idx] + analyze_text + output_start)

    sampling_params = SamplingParams(
        max_tokens=2048, 
        temperature=0.6,
        stop=['</output>\n', '</think>\n', '```python'],
        include_stop_str_in_output=True, 
        logprobs=1
    )
    output_outputs = critic_model.generate(output_inputs, sampling_params)

    yes_logps = [0. for _ in range(len(prefix_candidates))]
    for idx, critic_output in enumerate(output_outputs):
        for result in critic_output.outputs:
            for token_logprobs in result.logprobs:
                for token, info in token_logprobs.items():
                    if info.decoded_token == 'Yes':
                        yes_logps[idx // args.verify_num] += np.exp(info.logprob)
                        break

    return int(np.argmax(yes_logps)), yes_logps


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

    # Load models and tokenizers
    policy_model, policy_tokenizer, policy_stop_token = load_model_and_tokenizer(
        args.policy, gpu_memory_utilization=0.3)
    critic_model, critic_tokenizer, critic_stop_token = load_model_and_tokenizer(
        args.critic, gpu_memory_utilization=0.6)
    system_prompt = get_system_prompt(args.data_path)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    all_res = []
    all_policy_output_tokens, all_critic_output_tokens = 0, 0
    start_time = time.time()

    for idx, example in enumerate(test_data):
        print(f"Processing {idx} / {len(test_data)}")
        question = example['input']
        current_traj, candidate_traj = [], []
        momentum_uncertainty, get_answer = 0, False

        for step_idx in range(args.max_steps):
            try:
                input_text = build_policy_input(
                    policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
                if args.thinking_step_prefix_length == 0:
                    # Vanilla CoT: direct step generation
                    sampling_params = SamplingParams(max_tokens=2048, temperature=0.6, stop=["Step"])
                    outputs = policy_model.generate(input_text, sampling_params)
                    output = outputs[0].outputs[0]
                    current_traj.append(f"Step{step_idx}: {output.text.strip()}")
                    all_policy_output_tokens += len(output.token_ids)
                    logp = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
                    cur_signal = logp
                else:                
                    # Generate prefix candidates instead of full steps
                    prefix_candidates, prefix_logps, prefix_tokens = generate_prefix_candidates(
                        policy_model, policy_tokenizer, input_text, args, step_idx)
                    all_policy_output_tokens += prefix_tokens
                    
                    # Select best prefix using critic
                    best_prefix_idx, yes_scores = select_best_prefix_candidate(
                        critic_model, critic_tokenizer, question, current_traj, 
                        prefix_candidates, args, critic_stop_token, step_idx)
                    
                    selected_prefix = prefix_candidates[best_prefix_idx]
                    
                    # Complete the step from the selected prefix
                    completed_step, step_logp, completion_tokens = complete_step_from_prefix(
                        policy_model, policy_tokenizer, input_text, selected_prefix, 
                        policy_stop_token, step_idx)
                    all_policy_output_tokens += completion_tokens
                    
                    current_traj.append(completed_step)
                    cur_signal = step_logp

                    candidate_traj.append({
                        'step_idx': str(step_idx),
                        'step_uncertainty': str(np.exp(-cur_signal)),
                        'momentum_uncertainty/gamma': str(np.exp(-momentum_uncertainty) / args.momentum_rate),
                        'selected_prefix_idx': str(best_prefix_idx),
                        'prefix_candidates': prefix_candidates,
                        'selected_prefix': selected_prefix,
                        'completed_step': completed_step
                    })

                momentum_uncertainty = args.momentum_rate * momentum_uncertainty + \
                    (1 - args.momentum_rate) * cur_signal

                if "the answer is" in ''.join(current_traj).lower():
                    get_answer = True
                    break

            except Exception as e:
                print(f"Step error: {e}")
                continue

        # Try to finish with one last step if no answer found
        if not get_answer:
            try:
                input_text = build_policy_input(
                    policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
                sampling_params = SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1)
                outputs = policy_model.generate(input_text, sampling_params)
                current_traj.append(outputs[0].outputs[0].text.strip())
                all_policy_output_tokens += len(outputs[0].outputs[0].token_ids)
            except Exception as e:
                print(f"Final fallback error: {e}")

        all_res.append({
            'question': question,
            'ground_truth': example['target'],
            'current_traj': '\n'.join(current_traj),
            'final_answer': current_traj[-1] if current_traj else 'No answer',
            'candidate_traj': candidate_traj
        })

        os.makedirs(os.path.dirname(f'result/per_step_scale/{args.file_name}.json'), exist_ok=True)
        with open(f'result/per_step_scale/{args.file_name}.json', 'w') as f:
            json.dump(all_res, f, indent=4)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    os.makedirs(os.path.dirname(f'result/per_step_scale/time/{args.file_name}.txt'), exist_ok=True)
    with open(f'result/per_step_scale/time/{args.file_name}.txt', 'w') as f:
        f.write(f'\n\n{args.file_name}  time: {end_time - start_time}\n\n')
        f.write(f'all_policy_output_tokens: {all_policy_output_tokens}\n')
        f.write(f'all_critic_output_tokens: {all_critic_output_tokens}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/gpqa_diamond_test.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--momentum_rate', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--file_name', type=str,
                        default='vanilla_cot/guided_search-prefix-seed42')
    parser.add_argument('--candidate_num', type=int, default=4)
    parser.add_argument('--verify_num', type=int, default=1)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--aim_gpu', type=int, default=0)
    parser.add_argument('--policy', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--critic', type=str, default='GenPRM/GenPRM-1.5B')
    # New parameter for prefix length
    parser.add_argument('--thinking_step_prefix_length', type=int, default=0,
                        help='Number of tokens to generate for thinking step prefixes')
    args = parser.parse_args()

    main(args)