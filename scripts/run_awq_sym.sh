#!/bin/bash
# sensibilidade: AWQ simétrico (sym=True) no Qwen3-8B-Base, mesma calibração C4 x128; PPL e custo (série única)
cd ~/tcc-ptq || exit 1; export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$HOME/tcc-ptq/.venv-quant/bin:$PATH HF_HUB_DISABLE_XET=1; PY=.venv-quant/bin/python
tag=qwen3-8b-base; out=models/Qwen3-8B-Base-awq4-sym; FAIL=0
echo "=== $tag awq sym $(date +%T) quantize"; $PY quantize_awq_sym.py models/Qwen3-8B-Base $out --nsamples 128 || { echo "FALHA quantize"; echo ALLDONE-SYM $(date +%T) falhas=1; exit 1; }
echo "=== ppl";  $PY ppl_wikitext2.py --model $out --quant pre --out results/${tag}_awq4_sym.json || FAIL=$((FAIL+1))
echo "=== cost"; $PY bench_cost.py --model $out --quant pre --out results/${tag}_awq4_sym_cost.json || FAIL=$((FAIL+1))
echo ALLDONE-SYM $(date +%T) falhas=$FAIL
