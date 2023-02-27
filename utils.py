import re
import random
import requests
import dataclasses
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclasses.dataclass
class Example:
    question: str
    answer: str
    thought: str
    equations: str


def preprocess(example):
    """
    Преобразует пример из json в объект Example
    """
    question = example['question'].strip()
    thought, answer = example['answer'].split("####")
    thought, answer = thought.strip().replace('\n', ' '), answer.strip()
    equations = re.findall(r'<<(.+?)>>', thought) # парсим уравнения
    for equation in equations:  # удаляем уравнения из thought
        thought = thought.replace(f'<<{equation}>>', '')
    equations = '. '.join(equations)
    for c in ['$', ',', '%', '€', '"']: # удаляем лишние символы
        answer = answer.replace(c, '')
    answer = int(answer)
    if thought[-1] != '.':  # добавляем точку если отсутсвтует
        thought += '.'
    return Example(
        question=question,
        thought=thought,
        answer=answer,
        equations=equations,
    )


def generate_few_shots(dataset, seed, num_few_shots, only_equations):
    """
    Генерирует num_few_shots примеров для few shot предсказания.
    Если only_equations=True, то в примерах будут только уравнения.
    Используется seed для воспроизводимости.
    """
    dataset_copy = dataset.copy()
    random.Random(seed).shuffle(dataset_copy)
    prompt = ''
    for example in dataset_copy[:num_few_shots]:
        prompt += f'Q: {example.question}\n'
        prompt += f'A: {example.equations if only_equations else example.thought} ' 
        prompt += f'The answer is {example.answer}.\n\n'
    return prompt


def run_test(
    train_dataset,
    test_dataset,
    seed,
    greedy,
    temperature,
    batch_size,
    num_thoughts,
    num_few_shots,
    only_equations,
    checkpoint,
):
    """
    Вычисляет accuracy модели checkpoint на датасете. Если greedy=False,
    то используется "ансамбль" из num_thoughts предсказаний, и ответом
    будет самый частый среди них. Если only_equations=True, то в примерах
    будут только уравнения. Количество примеров - num_few_shots.
    """
    if checkpoint == "bigscience/bloom-petals":
        API_URL = "https://api-inference.huggingface.co/models/" + checkpoint
        headers = {"Authorization": "Bearer <hf_token>"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
    else:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto")
        pipe = pipeline(
            "text-generation",
            model=model.cuda(),
            tokenizer=tokenizer,
            device=0,
        )
        
    print("Running test...")
    correct = 0
    for example in tqdm(test_dataset):

        few_shots = generate_few_shots(train_dataset, seed, num_few_shots, only_equations)
        prompt = few_shots + f"Q: {example.question}\nA:"
        n_iters = 1 if greedy else num_thoughts
        prompt_batches = [prompt] * n_iters
        if checkpoint == "bigscience/bloom-petals": # bloom-petals не работает с pipeline
            payload = {
                "inputs": prompt_batches,
                "max_new_tokens": 128,
                "temperature": temperature,
                "do_sample": False if greedy else True,
                "return_full_text": False,
                "options": {
                    "wait_for_model": True,
                }
            }
            generated_texts = query(payload)
            # print(generated_texts)
            generated_texts = [
                text['generated_text'].split('\n\n')[0] for text in generated_texts
            ]
        else:
            generated_texts = pipe(
                prompt_batches,
                max_new_tokens=128,
                temperature=temperature,
                do_sample=False if greedy else True,
                return_full_text=False,
                batch_size=batch_size,
            )
            # выкидываем все что после первого переноса строки
            generated_texts = [
                text[0]['generated_text'].split('\n\n')[0] for text in generated_texts
            ]
        answers = []
        for text in generated_texts:
            try:
                answer = text.split('The answer is ')[1].split('.')[0]
                # убираем пунктуацию и спец символы
                for c in ['$', ',', '%', '€', '"', ' ']:
                    answer = answer.replace(c, '')
                answer = ''.join([i for i in answer if not i.isalpha()]) # убираем буквы
                answer = int(answer)
            except:
                answer = None
            answers.append(answer)
        pred_answer = max(answers, key=answers.count)
        if pred_answer == example.answer:
            correct += 1

    return correct / len(test_dataset)