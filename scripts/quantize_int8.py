"""Gera o artefato INT8 (LLM.int8 / bitsandbytes) em disco. Uso: python quantize_int8.py <dir_fp16> <dir_saida>"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
src, dst = sys.argv[1:3]
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16, device_map={"": 0},
                                         quantization_config=BitsAndBytesConfig(load_in_8bit=True))
m.save_pretrained(dst); AutoTokenizer.from_pretrained(src).save_pretrained(dst)
print("salvo em", dst)
