import os
import json
import argparse
from utils import preprocess, run_test


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, required=True)
parser.add_argument('--gsm_path', type=str, default='grade-school-math/grade_school_math/data/')
parser.add_argument('--checkpoint', type=str, default='bigscience/bloom-7b1')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--greedy', action='store_true')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--num_thoughts', type=int, default=20)
parser.add_argument('--num_few_shots', type=int, default=8)
parser.add_argument('--only_equations', action='store_true')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
train_path = args.gsm_path + "train.jsonl"
test_path = args.gsm_path + "test.jsonl"


print("Loading dataset...")
with open(train_path, 'r') as f:
    train_lines = f.readlines()
with open(test_path, 'r') as f:
    test_lines = f.readlines()
train_dataset = [preprocess(json.loads(line)) for line in train_lines]
test_dataset  = [preprocess(json.loads(line)) for line in test_lines]


accuracy = run_test(
    train_dataset,
    test_dataset,
    args.seed,
    args.greedy,
    args.temperature,
    args.batch_size,
    args.num_thoughts,
    args.num_few_shots,
    args.only_equations,
    args.checkpoint,
)

print(f"Seed: {args.seed}, Accuracy: {accuracy}")