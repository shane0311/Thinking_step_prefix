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


def extract_critic_judgment(critic_output_text):
    try:
        analyze_text = re.search(
            r'<analyze>(.*?)</analyze>', critic_output_text, re.DOTALL).group(1)
    except:
        analyze_text = ''
    try:
        output_text = re.search(
            r'<output>(.*?)</(?:output|think)>', critic_output_text, re.DOTALL).group(1)
    except:
        output_text = ''

    analyze_text = analyze_text.replace('<analyze>', '').replace('</analyze>', '')\
        .replace('paragraph_', 'Step').replace('paragraph', 'Step')\
        .replace('**Judgement**:', 'So the correctness of the step is:')
    return analyze_text, output_text


def apply_chat(tokenizer, messages, stop_token, add_gen=True):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=add_gen)
    return inputs.replace(stop_token, "").strip()


def run(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

    policy_model, policy_tokenizer, policy_stop = setup_model_and_tokenizer(
        args.policy, 0.6)
    critic_model, critic_tokenizer, critic_stop = setup_model_and_tokenizer(
        args.critic, 0.3)
    system_prompt = get_system_prompt(args.data_path)

    with open(args.data_path, encoding='utf-8') as f:
        test_data = json.load(f)

    # Approximate tokens for prefix (rough estimate: 1 token ≈ 4 characters)
    prefix_max_tokens = max(1, args.thinking_step_prefix_length // 4)

    all_res = []
    total_policy_tokens = total_critic_tokens = 0
    start_time = time.time()

    for idx, item in enumerate(test_data):
        print(f"Processing {idx + 1} / {len(test_data)}")
        question = item['input']
        current_traj = []
        momentum_uncertainty = 0
        get_answer = False

        for step in range(args.max_steps):
            try:
                base_prompt = apply_chat(policy_tokenizer, [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop)
                
                if step > 0:
                    prompt = base_prompt + '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                else:
                    prompt = base_prompt + '\nStep0:'

                # Track all generated prefixes for this step
                generated_prefixes = []
                
                # Step 1: Generate prefix first
                prefix_outputs = policy_model.generate(prompt, SamplingParams(
                    max_tokens=prefix_max_tokens, temperature=0.6, logprobs=1))
                prefix_output = prefix_outputs[0].outputs[0]
                step_prefix = prefix_output.text.strip()
                prefix_logp = prefix_output.cumulative_logprob
                total_policy_tokens += len(prefix_output.token_ids)

                prefix_avg_logp = prefix_logp / (len(prefix_output.token_ids) + 1e-8)
                generated_prefixes.append(step_prefix)

                # Step 2: Evaluate prefix with critic if uncertainty threshold is met
                prefix_approved = True
                if np.exp(prefix_avg_logp) < np.exp(momentum_uncertainty) * args.scaling_rate and step > 0 and step_prefix:
                    print(f"Critic invoked for question {idx}, step {step} prefix")
                    
                    # Create trajectory with prefix for critic evaluation
                    critic_traj = []
                    for t in current_traj:
                        if isinstance(t, dict):
                            critic_traj.append(f"Step{t['step_idx']}: {t['completed_step']}")
                        else:
                            critic_traj.append(t)
                    critic_traj.append(f"Step{step}: {step_prefix}")
                    
                    if 'math' in args.data_path.lower() or 'aime' in args.data_path.lower():
                        prompt_dict = ciritique_last_generation_math(
                            question, critic_traj)
                    else:
                        prompt_dict = ciritique_last_generation(
                            question, critic_traj)

                    critic_prompt = apply_chat(critic_tokenizer, [
                        {'role': 'system',
                            'content': prompt_dict['system_prompt']},
                        {'role': 'user', 'content': prompt_dict['user_prompt']}
                    ], critic_stop)

                    analyze_input = critic_prompt + \
                        f"\n<analyze>\nLet's analyze the paragraph {step} step by step: "
                    analyze_outputs = critic_model.generate(analyze_input, SamplingParams(
                        max_tokens=4096, temperature=0.6, stop=['</analyze>\n', '```python'], include_stop_str_in_output=True))
                    analyze_text = analyze_outputs[0].outputs[0].text.strip()
                    total_critic_tokens += len(
                        analyze_outputs[0].outputs[0].token_ids)

                    output_input = analyze_input + analyze_text + \
                        "\n<output>\n**Judgement**: $\\boxed"
                    judge_outputs = critic_model.generate(output_input, SamplingParams(max_tokens=4096, temperature=0.6, stop=[
                                                          '</output>\n', '</think>\n', '```python'], include_stop_str_in_output=True))
                    output_text = judge_outputs[0].outputs[0].text.strip()
                    total_critic_tokens += len(
                        judge_outputs[0].outputs[0].token_ids)

                    analyze_content, judge_content = extract_critic_judgment(
                        f"<analyze>{analyze_text}<output>{output_text}")

                    # Step 3: If prefix is incorrect, regenerate prefix
                    if "yes" not in judge_content.lower():
                        prefix_approved = False
                        print(f"Prefix rejected, regenerating for step {step}")
                        
                        # Build previous trajectory as string for revision prompt
                        prev_traj_str = '\n'.join([
                            f"Step{t['step_idx']}: {t['completed_step']}" if isinstance(t, dict) 
                            else t for t in current_traj
                        ])
                        
                        revision_prompt = apply_chat(policy_tokenizer, [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"},
                            {'role': 'assistant',
                                'content': prev_traj_str + f'\nStep{step}: {step_prefix}'},
                            {'role': 'user', 'content': f"\nYour reasoning start is incorrect.\n{analyze_content}\nPlease revise the beginning of your solution."},
                            {'role': 'assistant', 'content': f"Refined Step{step}: "}
                        ], policy_stop, add_gen=False)

                        revised_prefix_outputs = policy_model.generate(revision_prompt, SamplingParams(
                            max_tokens=prefix_max_tokens, temperature=0.6, logprobs=1))
                        revised_prefix_output = revised_prefix_outputs[0].outputs[0]
                        step_prefix = revised_prefix_output.text.strip()
                        total_policy_tokens += len(revised_prefix_output.token_ids)
                        prefix_avg_logp = revised_prefix_output.cumulative_logprob / \
                            (len(revised_prefix_output.token_ids) + 1e-8)
                        generated_prefixes.append(step_prefix)

                # Step 4: If prefix is approved, generate the rest of the step
                continuation_prompt = prompt + step_prefix
                continuation_outputs = policy_model.generate(continuation_prompt, SamplingParams(
                    max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1))
                continuation_output = continuation_outputs[0].outputs[0]
                continuation_text = continuation_output.text.strip()
                total_policy_tokens += len(continuation_output.token_ids)

                # Combine prefix and continuation
                full_step_text = step_prefix + continuation_text
                selected_prefix = step_prefix
                completed_step = full_step_text
                
                current_traj.append({
                    'step_idx': str(step),
                    'step_uncertainty': str(np.exp(-prefix_avg_logp)),
                    'momentum_uncertainty/gamma': str(np.exp(-momentum_uncertainty) / args.momentum_rate),
                    'prefixes': generated_prefixes,
                    'selected_prefix': selected_prefix,
                    'completed_step': completed_step
                })

                if "the answer is" in completed_step.lower():
                    get_answer = True
                    break

                # Update momentum uncertainty based on prefix (since that's what gets evaluated)
                momentum_uncertainty = args.momentum_rate * \
                    momentum_uncertainty + (1 - args.momentum_rate) * prefix_avg_logp
                    
            except Exception as e:
                print(f"Step {step} error: {e}")

        if not get_answer:
            try:
                # Build trajectory string for final prompt
                traj_str = '\n'.join([
                    f"Step{t['step_idx']}: {t['completed_step']}" if isinstance(t, dict) 
                    else t for t in current_traj
                ])
                
                final_prompt = apply_chat(policy_tokenizer, [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop) + '\n' + traj_str + f'\nStep{step}:'
                outputs = policy_model.generate(final_prompt, SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1))
                final_text = outputs[0].outputs[0].text.strip()
                current_traj.append({
                    'step_idx': str(step),
                    'step_uncertainty': 'N/A',
                    'momentum_uncertainty/gamma': 'N/A',
                    'prefixes': [final_text],
                    'selected_prefix': final_text,
                    'completed_step': final_text
                })
                total_policy_tokens += len(outputs[0].outputs[0].token_ids)
            except Exception as e:
                print(f"Final answer fallback error: {e}")

        all_res.append({
            'question': question,
            'ground_truth': item['target'],
            'current_traj': current_traj,
            'final_answer': current_traj[-1]['completed_step'] if current_traj and isinstance(current_traj[-1], dict) else 'No answer'
        })

        os.makedirs(os.path.dirname(f'../thinking_step_prefix/{args.file_name}.json'), exist_ok=True)
        with open(f'../thinking_step_prefix/{args.file_name}.json', 'w') as f:
            json.dump(all_res, f, indent=4)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    os.makedirs(os.path.dirname(f'../thinking_step_prefix/time/{args.file_name}.txt'), exist_ok=True)
    with open(f'../thinking_step_prefix/time/{args.file_name}.txt', 'w') as f:
        f.write(f'\n\n{args.file_name}  time: {end_time - start_time}\n\n')
        f.write(f'all_policy_output_tokens: {total_policy_tokens}\n')
        f.write(f'all_critic_output_tokens: {total_critic_tokens}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='../data/gpqa_diamond_test.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--momentum_rate', type=float, default=0.9)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--thinking_step_prefix_length', type=int, default=100,
                        help='Length of step prefix in characters. Model generates and validates this prefix before completing the full step.')
    parser.add_argument('--file_name', type=str,
                        default='llm_as_a_critic-mur.json')
    parser.add_argument('--aim_gpu', type=int, default=1)
    parser.add_argument('--policy', type=str, default='Qwen3-1.7B')
    parser.add_argument('--critic', type=str, default='genprm1.5B')
    args = parser.parse_args()
    run(args)