#!/bin/sh
# Repite con el METIS actual los dos estudios de octubre (ordinales, 2400 evaluaciones; fugas, 288).
O=~/strat/oct; PY=~/doc-prueba/hefesto/.venv/bin/python
cd $O
export PYTHONPATH=/home/ubuntu/doc-split/metis OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 METIS_N_JOBS=1 LOKY_MAX_CPU_COUNT=1 PYTHONWARNINGS=ignore
echo "| METIS (METIS_AGENT) | estudios de operadores de octubre repetidos con el METIS actual (~/strat/oct, tmux oct); apaga al acabar si no hay procesos ajenos | tmux | 14 (1 hilo) | $(date -u '+%Y-%m-%d %H:%M') | ~1 h | EN CURSO |" >> ~/COMPUTE_SHARING.md
{
  for k in 0 1 2 3 4 5 6 7; do
    for d in adult cardio rrhh; do echo "$PY estudio_ordinales.py $d ord/${d}_$k.jsonl $((k+1)) $k > log/ord_${d}_$k.log 2>&1"; done
  done
  for k in 0 1 2 3 4 5 6 7; do
    for d in adult cardio telco rrhh; do echo "$PY estudio_privacidad.py $d priv/${d}_$k.jsonl $((k+1)) $k > log/priv_${d}_$k.log 2>&1"; done
  done
} > log/cola.txt
taskset -c 2-15 xargs -P 14 -I{} sh -c "{}" < log/cola.txt
sed -i 's/risk_aversion": 5.0/risk_aversion": 3.0/; s/("mean", "median", "ssd", "fsd")/("mean", "median", "ssd")/' pisos_familia.py
$PY pisos_familia.py ord fidelity casi_copia > res_fid_l12.txt 2>&1
$PY pisos_familia.py ord fidelity casi_copia mean/mean > res_fid_l3.txt 2>&1
$PY pisos_familia.py priv privacy "" > res_priv.txt 2>&1
$PY analiza_privacidad.py priv > res_priv_delta.txt 2>&1
wc -l ord/*.jsonl priv/*.jsonl | tail -1 > recuento.txt
grep -l "Traceback" log/*.log res_*.txt > errores.txt 2>/dev/null
sed -i "s/^\(| METIS (METIS_AGENT) | estudios de operadores de octubre.*\)| EN CURSO |$/\1| HECHO $(date -u '+%m-%d %H:%M') UTC |/" ~/COMPUTE_SHARING.md
echo FIN > fin
for n in 1 2; do
  ajenos=$(ps -eo pcpu,args --no-headers | awk '$1>20' | grep -v -e strat -e "ps -eo" | wc -l)
  [ "$ajenos" -gt 0 ] && { echo "no apago: $ajenos procesos ajenos $(date -u)" > apagado.txt; exit 0; }
  sleep 120
done
echo "apago $(date -u)" > apagado.txt
sudo -n shutdown -h +60
