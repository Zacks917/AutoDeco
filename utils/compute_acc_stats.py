import argparse
import glob
import itertools
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import List, Union

import numpy as np
from tqdm import tqdm

from boxed_extract import compute_score


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def parse_metrics_from_ascii_table(text: str) -> list[float]:
    """Parse an ASCII table string and extract all numeric values under the 'Acc' column.

    The function looks for a header row (a line starting with '|' that contains 'Acc')
    then finds subsequent data rows (lines starting with '|') and collects the Acc values.
    Returns a list of acc values.
    """
    acc_values: list[float] = []
    acc_header_idx = None
    found_header = False

    # Split into lines and iterate
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        
        # Extract cells by splitting on '|' and stripping whitespace
        cells = [c.strip() for c in line.split('|')[1:-1]]
        
        # Detect header row (contains 'Acc')
        if acc_header_idx is None:
            # Look for Acc column (case insensitive)
            for idx, c in enumerate(cells):
                if c.lower().startswith('acc'):
                    acc_header_idx = idx
                    found_header = True
                    break
        
        # If header was found and this is a data row (not the header row itself)
        if acc_header_idx is not None and found_header and line.startswith('|') and '|  Acc  ' not in line:
            # Parse Acc column
            if acc_header_idx < len(cells):
                acc_cell = cells[acc_header_idx]
                # Match float numbers
                m = re.search(r"\d+(?:\.\d+)?", acc_cell)
                if m:
                    try:
                        acc_values.append(float(m.group(0)))
                    except ValueError:
                        pass

    return acc_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generation_log'),
        help='Directory containing the JSON or TXT files.'
    )
    parser.add_argument(
        '--prefix',
        default='DeepSeek-R1-Distill-Qwen-7B-temp1.0-top_p0.9-pass@16-seed',
        help='Filename prefix to match.'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=16,
        help='Value of k for pass@k calculation.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output.'
    )
    args = parser.parse_args()

    # Find all matching TXT and JSON files
    pattern_txt = os.path.join(args.dir, f"{args.prefix}*.txt")
    pattern_json = os.path.join(args.dir, f"{args.prefix}*.json")
    files_txt = sorted(glob.glob(pattern_txt))
    files_json = sorted(glob.glob(pattern_json))
    
    if not files_txt and not files_json:
        print(f"No files matched: {args.prefix}*.txt or {args.prefix}*.json")
        return

    print(f"TXT files matched: {len(files_txt)}")
    print(f"JSON files matched: {len(files_json)}")
    
    # Result 1: Calculate Acc from TXT files
    print("\n" + "="*60)
    print("Result 1: Accuracy")
    print("="*60)
    
    if files_txt:
        all_acc: list[float] = []
        
        for fp in files_txt:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(fp, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            accs = parse_metrics_from_ascii_table(content)
            if accs:
                all_acc.extend(accs)

        if all_acc:
            acc_mean = mean(all_acc)
            acc_std = pstdev(all_acc) if len(all_acc) > 1 else 0.0
            print(f"Acc list ({len(all_acc)} values): {all_acc}")
            print(f"Mean Acc: {acc_mean:.2f}")
            print(f"Std Acc: {acc_std:.2f}")
        else:
            print("No Acc values found in TXT files.")
    else:
        print("No TXT files found for Acc calculation.")
    
    # Result 2: Calculate Pass@k from JSON files
    print("\n" + "="*60)
    print(f"Result 2: Pass@{args.k}")
    print("="*60)
    
    if not files_json:
        print("No JSON files found for Pass@k calculation.")
        return
    
    # For JSON files, use the new method
    # Step 1: Read all matching JSON files and merge by problem
    # Structure: problem -> list of all solutions across all files
    problem_to_solutions = defaultdict(list)
    
    for fp in files_json:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    problem = data.get('problem', '')
                    ground_truth = data.get('ground_truth', '')
                    solutions = data.get('solutions', [])
                    
                    # Store problem, ground truth, and solutions
                    problem_to_solutions[problem].append({
                        'ground_truth': ground_truth,
                        'solutions': solutions
                    })
        except Exception as e:
            print(f"WARN: Error reading {os.path.basename(fp)}: {e}")
            continue

    if not problem_to_solutions:
        print("No valid data found in JSON files.")
        return

    # Step 2: Compute pass@k for each problem
    problem_stats = {}
    
    for problem, problem_data_list in problem_to_solutions.items():
        # Get ground truth (should be consistent across all entries)
        ground_truth = problem_data_list[0]['ground_truth']
        
        # Collect all solutions for this problem
        all_solutions = []
        for entry in problem_data_list:
            all_solutions.extend(entry['solutions'])
        
        # Step 3: Extract and match answers, compute correct count
        correct_count = 0
        total_attempts = len(all_solutions)
        
        for solution in all_solutions:
            if compute_score(solution, ground_truth):
                correct_count += 1
        
        # Also calculate average accuracy for this problem
        avg_acc = (correct_count / total_attempts * 100) if total_attempts > 0 else 0.0
        
        problem_stats[problem] = {
            'total_attempts': total_attempts,
            'correct_count': correct_count,
            'avg_acc': avg_acc
        }
    
    # Step 4: Calculate pass@k using the statistical estimator
    all_total_attempts = [stats['total_attempts'] for stats in problem_stats.values()]
    all_correct_count = [stats['correct_count'] for stats in problem_stats.values()]
    all_pass_at_k = estimate_pass_at_k(all_total_attempts, all_correct_count, args.k).tolist()
    
    # Compute mean and std
    if all_pass_at_k:
        pass_at_k_mean = mean(all_pass_at_k) * 100
    else:
        pass_at_k_mean = 0.0

    
    # Print results
    print(f"Number of problems: {len(problem_stats)}")
    if args.verbose:
        print(f"Pass@{args.k} list: {[f'{x*100:.2f}%' for x in all_pass_at_k]}")
    print(f"Mean Pass@{args.k}: {pass_at_k_mean:.2f}%")    
    
    # Optional: Show per-problem statistics
    if len(problem_stats) <= 10:
        print(f"\nPer-problem statistics:")
        for i, (problem, stats) in enumerate(problem_stats.items(), 1):
            print(f"  Problem {i}: Pass@{args.k}={stats['pass_at_k']*100:.2f}%, "
                  f"Acc={stats['avg_acc']:.2f}%, "
                  f"Correct={stats['correct_count']}/{stats['total_attempts']}")


if __name__ == '__main__':
    main()


