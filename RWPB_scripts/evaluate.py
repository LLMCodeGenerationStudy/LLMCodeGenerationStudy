import json
import os
import tempfile
import re
import sys
import io
import traceback
import subprocess

def find_function_names(code):
    """
        obtain the function signature
    """
    pattern = r"\bdef\s+(\w+)\s*\("
    matches = re.findall(pattern, code)
    
    return matches


def process_answer(text):
    """
        extract the function body
    """
    text = text.strip('[PYTHON]').strip('[/PYTHON]')
    
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    else:
        match = re.search(r'Here(.*?)\n', text)
        if match:
            text = re.sub('Here(.*?)\n', '', text, count=1)
        match = re.search(r'One approach(.*?)\n', text)
        if match:
            text = re.sub('One approach(.*?)\n', '', text, count=1)

    if text.startswith("markdown"):
        text = text[8:]
    if text.endswith('</s>'):
        return text[:-4]
    else:
        return text



cnt = 0
t_pass = 0
t_partial_wrong = 0

def run_code(code):
    """
        execute the extracted code
    """
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    if True:
        try:
            result = subprocess.run(["python", temp_file_path], capture_output=True, text=True, timeout=60)
            errors = result.stderr
            if "AssertionError" in errors:
                errors = "function_error"
        except subprocess.TimeoutExpired:
            errors = "timeout error"

        os.remove(temp_file_path)
    
    return errors





file_name = './LLMGeneratedCode/rwpb-llama3.json'

with open(file_name, 'r') as f:
    datas = json.load(f)

for item in datas:
    cnt += 1

    # model_generated_code
    solution = item['solution']

    # the programming problem in RWPB + its canonical solution = the complete function body
    canonical_solution = item['prompt'] + item['canonical_solution']
    
    # the function signature of canonical solution
    solution_signature = find_function_names(canonical_solution)[0]


    canonical_solution = canonical_solution.replace(solution_signature, 'SOLUTION_SIGNATURE')

    lines = item['unprocess_testcases'].split('\n')
    code = solution + '\n' + canonical_solution + '\n'
    tmp_assert_num = 0
    tmp_wrong_num = 0
    for line in lines:
        code = code + line + '\n'
        if line.startswith('assert'):
            tmp_assert_num += 1
            content = run_code(code)
            content = content.replace("\n", "")
            content = content.replace("\b", "")
            content = content.strip()
            if content != "":
                tmp_wrong_num += 1
    
    if tmp_wrong_num != tmp_assert_num and tmp_wrong_num != 0:
        print(f"{item['task_id']}")
        t_partial_wrong += 1
    
    if tmp_wrong_num == 0:
        t_pass += 1

print(cnt)
print(f"pass rate: {(t_pass)/cnt}")
print(f"partial wrong rate: {(t_partial_wrong)/cnt}")


# with open(os.path.join(file_name), 'w') as f:
#     json.dump(datas, f, indent=1, ensure_ascii=False)
