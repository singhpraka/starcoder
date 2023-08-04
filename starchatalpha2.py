# -*- coding: utf-8 -*-
!pip -q install langchain transformers
!pip -q install accelerate bitsandbytes sentencepiece Xformers

!nvidia-smi

from huggingface_hub import notebook_login
notebook_login()

import torch
import transformers
from transformers import GenerationConfig, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/starchat-alpha")



model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/starchat-alpha",
                                              load_in_8bit=True,
                                              device_map='auto',
                                              torch_dtype=torch.float16
                                              )

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/starchat-beta",
                                              load_in_8bit=True,
                                              device_map='auto',
                                             quantization_config=quantization_config,
                                              #torch_dtype=torch.float16
                                              )

"""### Prompt"""

def generate_response(input_prompt):
    system_prompt = "<|system|>\nBelow is a conversation between a human user and a helpful AI coding assistant.<|end|>\n"

    user_prompt = f"<|user|>\n{input_prompt}<|end|>\n"

    assistant_prompt = "<|assistant|>"

    full_prompt = system_prompt + user_prompt + assistant_prompt

    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs,
                            eos_token_id = 0,
                            pad_token_id = 0,
                            max_length=256,
                            early_stopping=True)
    output =  tokenizer.decode(outputs[0])
    output = output[len(full_prompt):]
    if "<|end|>" in output:
        cutoff = output.find("<|end|>")
        output = output[:cutoff]
    print(input_prompt+'\n')

    print(output)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# generate_response("Implement a recursive function to calculate the factorial of a given number.")

generate_response("Implement a program that validates whether the given password satisfies certain criteria, such as having at least one uppercase letter and one special character.")
generate_response("Create a program that converts temperature from Celsius to Fahrenheit.")