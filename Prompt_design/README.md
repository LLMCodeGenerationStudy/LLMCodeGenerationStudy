# Code Generation Prompt Design

To ensure experimental consistency across various LLMs, we employed a standardized prompt format, as outlined in an existing study [1].
Notably, our prompts excluded library import information, necessitating that the models autonomously identify and import the required libraries for the APIs utilized.

## Code Generation Prompt
```
Please provide a self-contained Python script that solves the following problem in a markdown code block:

{Programming problem}

Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:

{Generated code}

```


# Docstring Generation Prompt
```
Your task is to write a comprehensive docstring for the provided Python function. The docstring should adhere to the following criteria: naturalness, accuracy, and clarity.

**Naturalness:**  
Ensure the docstring follows a human writing style, making it naturally readable from a developer’s perspective. Aim for a conversational yet professional tone that developers will find easy to understand.

**Accuracy:**  
The description must accurately describe the function's functionality. Ensure that the arguments and return values are described accurately, with types and names identical to those in the code.

**Clarity:**  
The docstring should be clear, precise, and concise, avoiding any redundant or irrelevant information.

The docstring should includes:

1. A brief description of what the function does.

2. Args of the functions: A detailed description of each argument, including its type and purpose.

3. Returns of the functions: A detailed description of the return value, including its type.


Here is an example:

## Python function:
def add_numbers(*args: int) -> int:
    if not all(isinstance(arg, int) for arg in args):
        raise TypeError("All arguments must be integers.")
    return sum(args)

## Docstring:
"""
Adds multiple integers and returns their sum.

Args:
    *args (int): A variable number of integer arguments to be added.

Returns:
    int: The sum of all provided integer arguments.
"""

Now you are asked to complete the task as required.

## Python function:
{Python function}

## Docstring:
```

# Unit Tests Generation Prompt
```
Your task is to write unit tests using `assertion` for the provided Python function and its corresponding docstring. 
The unit tests should comprehensively cover all functional aspects, including normal cases, edge cases, and error handling. Ensure that the test cases include:

Typical scenarios that the function is expected to handle.
Edge cases that test the boundaries and limits of the function's input.
Error handling to validate the function's robustness against invalid inputs and unexpected behavior.


Here is an example:

## Python function:
def add_numbers(*args: int) -> int:
"""
Adds multiple integers and returns their sum.

Parameters:
    *args (int): A variable number of integer arguments to be added.

Returns:
    int: The sum of all provided integer arguments.
"""
    if not all(isinstance(arg, int) for arg in args):
        raise TypeError("All arguments must be integers.")
    return sum(args)

## Unit tests:
assert add_numbers(1, 2, 3) == 6
assert add_numbers(-1, -2, -3) == -6
assert add_numbers(-1, 2, -3) == -2
assert add_numbers(0, 0, 0) == 0
assert add_numbers(10) == 10
assert add_numbers(1000000, 2000000, 3000000) == 6000000


Now you are asked to complete the task as required.

## Python function:
{Python function with a docstring}

## Unit tests:
```

# Enhance Unit Test Coverage Prompt
```
Your task is to write comprehensive unit tests using assertions for the provided Python function. 
Ensure that the unit tests thoroughly cover all given target code branches.


Here is an example:

## Python function:
def calculate_discount(price, is_member, is_holiday):
    if price <= 0:
        raise ValueError("Price must be positive")
    
    if is_holiday:
        if is_member:
            return price * 0.7  # 30% discount
        return price * 0.8  # 20% discount
    else:
        if is_member:
            return price * 0.9  # 10% discount
    return price

## Target code branches:
return price * 0.7  # 30% discount

## Unit tests:
```
assert calculate_discount(10, True, True) == 7
```

Now you are asked to complete the task as required. **Unit tests must cover the target code branches.**

## Python function:
{function}

## Target code branches:
{target branches}

## Unit tests:






# Enhance Unit Test Compilation Error Prompt


Your task is to revise the unit tests using assertions for the provided Python function, based on the incorrect unit tests and corresponding compilation error messages. 

The revised unit tests should fix the compilation errors. Ensure the corrected unit tests do not produce any compilation errors.


Here is an example:

## Python function:
def calculate_discount(price, discount_rate, is_member):
    if price < 0 or discount_rate < 0:
        raise ValueError("Price and discount rate must be non-negative")
    if is_member:
        discount_rate += 0.05
    final_price = price * (1 - discount_rate)
    if final_price < 0:
        final_price = 0
    return round(final_price, 2)

## Incorrect Unit tests:
assert calculate_discount(10, True) == 7


## Compilation error message
TypeError: test_discount() missing 1 required positional argument: is_member

## Revised unit test
assert calculate_discount(10, True, True) == 7

Now you are asked to complete the task as required. **Incorrect unit tests should be fixed.**

## Python function:
{function}

## Incorrect Unit tests:
{unit tests}

## Compilation error message
{log}

## Revised unit test
```

# Mitigate Bugs via self-critic
```
You are an expert in code bug fixing. There is a programming problem description along with the corresponding erroneous code, and the compilation/execution error logs. Additionally, I will provide you with a list of possible types and causes of errors in the code. Please first analyze the possible locations and causes of errors in this code, and then correct it based on the analysis results.

## Problem description is:
{prompt}

## Error generation code is:
{solution}

## Compilation and execution error message
{error log}

## Possible error types and causes:
1.1 Bugs due to incorrect syntax structure.
1.2 LLMs only output explanation without code.
2.1 Bugs due to misjudge variable type or misunderstand API usage.
2.2 Bugs due to missing library or incorrectly import library.
2.3 LLMs refer functions or variables without definition.
2.4 Bugs due to out-of-bound access or missing corner case check. 
2.5 LLMs output code with incorrect arguments.
3.1 LLMs misunderstand the problem requirement, lack knowledge to implement, or produce the faulty logic. 
3.2 LLMs output code without any logic or apply incorrect knowledge.
3.3 Bugs due to incorrectly follow input-output format.  

Please first analyze the causes of errors in the code, and identify the positions of the errors. Then correct the bugs and generate the correct code. 
**Do not generate test samples or use samples, just generate the correct code at the end.**
```

## Reference

[1] Liu J, Xia C S, Wang Y, et al. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation[J]. Advances in Neural Information Processing Systems, 2024, 36.