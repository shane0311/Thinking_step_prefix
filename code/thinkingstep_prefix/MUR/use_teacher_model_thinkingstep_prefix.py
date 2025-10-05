import os
import json
import numpy as np
import re
import argparse
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def setup_model_and_tokenizer(model_path, gpu_mem):
    model = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8096*2,
                trust_remote_code=True, gpu_memory_utilization=gpu_mem)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, tokenizer.eos_token


def get_system_prompt(data_path):
    path = data_path.lower()
    if any(x in path for x in ['math', 'aime', 'amc']):
        return 'You are a helpful math assistant.'
    elif any(x in path for x in ['reclor', 'gpqa', 'logiqa']):
        return 'You are a helpful assistant. Here is a question and four candidate answers. You need to reason step by step and choose the most likely answer from the four candidate answers. Answer "A", "B", "C", or "D".'
    elif 'strategyqa' in path:
        return 'You are a helpful assistant. After each step, you may receive a feedback from the user, indicating that the previous step is incorrect. You should then revise your solution accordingly. Please answer "Yes" or "No".'
    return ''


def apply_chat(tokenizer, messages, stop_token, add_gen=True):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=add_gen)
    return inputs.replace(stop_token, "").strip()


def run(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

    policy_model, policy_tokenizer, policy_stop = setup_model_and_tokenizer(
        args.policy, 0.2)
    teacher_model, teacher_tokenizer, teacher_stop = setup_model_and_tokenizer(
        args.teacher_model, 0.3)
    system_prompt = get_system_prompt(args.data_path)

    with open(args.data_path, encoding='utf-8') as f:
        test_data = json.load(f)

    all_res = []
    total_policy_tokens = total_teacher_tokens = 0
    start_time = time.time()

    for idx, item in enumerate(test_data):
        print(f"Processing {idx + 1} / {len(test_data)}")
        question = item['input']
        current_traj = []
        candidate_traj = []
        momentum_uncertainty = 0
        get_answer = False

        for step in range(args.max_steps):
            try:
                prompt = apply_chat(policy_tokenizer, [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop)
                if step > 0:
                    prompt += '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                else:
                    prompt += '\nStep0:'

                if args.thinking_step_prefix_length == 0:
                    # Vanilla CoT: generate full step directly with policy model
                    outputs = policy_model.generate(prompt, SamplingParams(
                        max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1))
                    output = outputs[0].outputs[0]
                    step_text = output.text.strip()
                    logp = output.cumulative_logprob
                    total_policy_tokens += len(output.token_ids)
                    avg_logp = logp / (len(output.token_ids) + 1e-8)
                    
                    # For vanilla CoT, we don't have prefixes, so use the full step
                    prefix_candidates = [step_text[:50] + "..." if len(step_text) > 50 else step_text]
                    selected_prefix = prefix_candidates[0]
                    completed_step = f"Step{step}: {step_text}"
                    
                else:
                    # Always generate policy model prefix first
                    policy_prefix_outputs = policy_model.generate(prompt, SamplingParams(
                        max_tokens=args.thinking_step_prefix_length, temperature=0.6, logprobs=1))
                    policy_prefix_output = policy_prefix_outputs[0].outputs[0]
                    policy_prefix = policy_prefix_output.text.strip()
                    policy_prefix_logp = policy_prefix_output.cumulative_logprob
                    policy_prefix_avg_logp = policy_prefix_logp / (len(policy_prefix_output.token_ids) + 1e-8)
                    total_policy_tokens += len(policy_prefix_output.token_ids)
                    
                    # Check if we should use teacher model
                    use_teacher = (
                        step >= 0 and 
                        np.exp(policy_prefix_avg_logp) < np.exp(momentum_uncertainty) * args.scaling_rate
                    )
                    
                    if use_teacher:
                        print(f"Using teacher model for question {idx}, step {step} (policy_prefix_prob={np.exp(policy_prefix_avg_logp):.4f} < threshold={np.exp(momentum_uncertainty) * args.scaling_rate:.4f})")
                        
                        # Teacher model generates the prefix
                        teacher_prompt = apply_chat(teacher_tokenizer, [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                        ], teacher_stop)
                        if step > 0:
                            teacher_prompt += '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                        else:
                            teacher_prompt += '\nStep0:'

                        teacher_outputs = teacher_model.generate(teacher_prompt, SamplingParams(
                            max_tokens=args.thinking_step_prefix_length, temperature=0.6, logprobs=1))
                        teacher_output = teacher_outputs[0].outputs[0]
                        teacher_prefix = teacher_output.text.strip()
                        total_teacher_tokens += len(teacher_output.token_ids)
                        
                        # Use teacher's prefix as selected
                        prefix_candidates = [policy_prefix, teacher_prefix]
                        selected_prefix = teacher_prefix
                    else:
                        # Use policy model's prefix
                        prefix_candidates = [policy_prefix]
                        selected_prefix = policy_prefix

                    # Complete step from selected prefix using policy model
                    completion_prompt = prompt + selected_prefix
                    selected_prefix_token_len = len(policy_tokenizer.tokenize(selected_prefix))
                    completion_outputs = policy_model.generate(completion_prompt, SamplingParams(
                        max_tokens=2048 - selected_prefix_token_len, temperature=0.6, stop=["Step"], logprobs=1))
                    completion_text = completion_outputs[0].outputs[0].text.strip()
                    total_policy_tokens += len(completion_outputs[0].outputs[0].token_ids)
                    
                    step_text = selected_prefix + completion_text
                    completed_step = f"Step{step}: {step_text}"
                    # Use completion's logprob for momentum calculation
                    avg_logp = completion_outputs[0].outputs[0].cumulative_logprob / \
                        (len(completion_outputs[0].outputs[0].token_ids) + 1e-8)

                current_traj.append(f"Step{step}: {step_text}")
                
                # Add candidate trajectory information
                candidate_traj.append({
                    'step_idx': str(step),
                    'step_uncertainty': str(np.exp(-avg_logp)),
                    'momentum_uncertainty/gamma': str(np.exp(-momentum_uncertainty) / args.scaling_rate),
                    'prefix_candidates': prefix_candidates,
                    'selected_prefix': selected_prefix,
                    'completed_step': completed_step
                })

                if "the answer is" in ''.join(current_traj).lower():
                    get_answer = True
                    break

                momentum_uncertainty = args.momentum_rate * \
                    momentum_uncertainty + (1 - args.momentum_rate) * avg_logp
            except Exception as e:
                print(f"Step {step} error: {e}")

        if not get_answer:
            try:
                final_prompt = apply_chat(policy_tokenizer, [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop) + '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                outputs = policy_model.generate(final_prompt, SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1))
                current_traj.append(outputs[0].outputs[0].text.strip())
                total_policy_tokens += len(outputs[0].outputs[0].token_ids)
            except Exception as e:
                print(f"Final answer fallback error: {e}")

        all_res.append({
            'question': question,
            'ground_truth': item['target'],
            'current_traj': '\n'.join(current_traj),
            'final_answer': current_traj[-1] if current_traj else 'No answer',
            'candidate_traj': candidate_traj
        })

        os.makedirs(os.path.dirname(f'../thinking_step_prefix/{args.file_name}.json'), exist_ok=True)
        with open(f'../thinking_step_prefix/{args.file_name}.json', 'w') as f:
            json.dump(all_res, f, indent=4)

    elapsed = time.time() - start_time
    print(f"Total time taken: {elapsed:.2f} seconds")

    os.makedirs(os.path.dirname(f'../thinking_step_prefix/time/{args.file_name}.txt'), exist_ok=True)
    with open(f'../thinking_step_prefix/time/{args.file_name}.txt', 'w') as f:
        f.write(f"{args.file_name} time: {elapsed:.2f} sec\n")
        f.write(f"All policy output tokens: {total_policy_tokens}\n")
        f.write(f"All teacher output tokens: {total_teacher_tokens}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='../data/gpqa_diamond_test.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--momentum_rate', type=float, default=0.9)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--file_name', type=str,
                        default='llm_as_a_teacher-prefix.json')
    parser.add_argument('--aim_gpu', type=int, default=3)
    parser.add_argument('--policy', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--teacher_model', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--thinking_step_prefix_length', type=int, default=50,
                        help='Number of tokens to generate for thinking step prefixes')
    args = parser.parse_args()
    run(args)