#!/bin/bash
# Fila da madrugada. Espera (1) o run_awq.sh terminar e (2) o arquivo-sinal NOITE_OK (criado apos o /code-review).
# Parte B usa os MESMOS 128 documentos do baseline (so muda o empacotamento em blocos de 2048). Parte A: replicas de custo (GPU livre, antes das quantizacoes). Parte B: sensibilidade da calibracao (blocos de 2048).
cd ~/tcc-ptq || exit 1
while pgrep -f "[r]un_awq.sh" >/dev/null || [ ! -f NOITE_OK ]; do sleep 30; done
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$HOME/tcc-ptq/.venv-quant/bin:$PATH HF_HUB_DISABLE_XET=1; PY=.venv-quant/bin/python; FAIL=0
echo "=== A replicas de custo $(date +%T)"
for r in r2 r3; do
 for t in "qwen3-8b-base:Qwen3-8B-Base" "llama3-8b:Meta-Llama-3-8B"; do
  tag=${t%%:*}; d=models/${t##*:}
  echo "--- $tag fp16 $r";  $PY bench_cost.py --model $d --quant none --out results/${tag}_fp16_cost_$r.json || FAIL=$((FAIL+1))
  for q in int8 gptq4 awq4; do
   echo "--- $tag $q $r"; $PY bench_cost.py --model ${d}-$q --quant pre --out results/${tag}_${q}_cost_$r.json || FAIL=$((FAIL+1))
  done
 done
done
echo "=== B sensibilidade calibracao $(date +%T)"
for t in "qwen3-8b-base:Qwen3-8B-Base" "llama3-8b:Meta-Llama-3-8B"; do
 tag=${t%%:*}; d=models/${t##*:}
 for m in gptq awq; do
  out=${d}-${m}4-c2048
  echo "--- $tag $m c2048 quantize $(date +%T)"; $PY quantize_${m}_c2048.py $d $out --nsamples 128 || { FAIL=$((FAIL+1)); continue; }
  echo "--- $tag $m c2048 ppl";  $PY ppl_wikitext2.py --model $out --quant pre --out results/${tag}_${m}4_c2048.json || FAIL=$((FAIL+1))
 done
done
echo ALLDONE-NOITE $(date +%T) falhas=$FAIL
