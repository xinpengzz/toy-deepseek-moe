import torch
from transformers import GenerationConfig

from deepseek.configuration_deepseek import DeepseekV2Config
from deepseek.modeling_deepseek import DeepseekV2ForCausalLM
from deepseek.tokenization_deepseek_fast import DeepseekTokenizerFast

model_path = "./deepseek"

model_config_path = "deepseek/config.json"
model_config = DeepseekV2Config.from_pretrained(model_config_path)

tokenizer = DeepseekTokenizerFast.from_pretrained(model_path)

model = DeepseekV2ForCausalLM(model_config)

model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)