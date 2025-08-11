"""
Prompt utilities for the FinAdvisor LLM inference pipeline.
"""

def generate_prompt(instruction, input_text=None):
    """
    Generate a formatted prompt for the FinAdvisor LLM.
    
    Args:
        instruction (str): The main instruction/question for the model
        input_text (str, optional): Additional context or input
        
    Returns:
        str: Formatted prompt ready for tokenization
    """
    if input_text and input_text.strip() and input_text.lower() != "none":
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def extract_response(generated_text):
    """
    Extract the response from the generated text.
    
    Args:
        generated_text (str): The full generated text from the model
        
    Returns:
        str: Cleaned response text
    """
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[1].strip()
    else:
        response = generated_text.strip()
    
    # Remove any trailing special tokens or unwanted content
    if '"""' in response:
        response = response.split('"""')[0].strip()
    
    return response
