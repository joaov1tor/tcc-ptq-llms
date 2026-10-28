#!/bin/bash
# INT8: quantiza -> PPL completo -> custo, um modelo por vez (medicao de custo exige GPU sem outra carga nossa)
cd ~/tcc-ptq; PY=.venv-quant/bin/python
for pair in "qwen3-8b-base:Qwen3-8B-Base" "llama3-8b:Meta-Llama-3-8B"; do
  tag=${pair%%:*}; dir=models/${pair##*:}; out=models/${pair##*:}-int8
  echo "=== $tag $(date +%T) quantize"; $PY quantize_int8.py $dir $out || continue
  echo "=== $tag ppl";  $PY ppl_wikitext2.py --model $out --quant pre --out results/${tag}_int8.json
  echo "=== $tag cost"; $PY bench_cost.py   --model $out --quant pre --out results/${tag}_int8_cost.json
done
echo ALLDONE $(date +%T)
