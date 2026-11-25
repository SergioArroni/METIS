#!/bin/sh
# Reparto único estratificado (METIS e9b75b3 en ~/doc-split): banco de defectos rehecho y todo reevaluado.
#  0. copias de los scripts en ~/strat con sus rutas y los parches (strat_parches.py)
#  1. comprobación de la línea base: 3 casos del benchmark con el código anterior (cacd39a) y las cotas viejas
#  2. banco 2, estudio, privacidad y utilidad y matices generados de nuevo con el reparto estratificado
#  3. calibraciones (banco, sin tipos, banco pu y benchmark), un proceso por tabla
#  4. cola de 14 procesos de 1 hilo: METIS, METIS sin tipos, estudio, métricas nuevas, SDMetrics, SynthEval,
#     synthcity, banco pu, matices, benchmark y efecto de las etiquetas
#  5. une las partes, fuerza del trío y FIN.  La AWS no se apaga (la Fase 3 es su dueña).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/ubuntu/doc-split/metis
S=~/strat; V=$S/validadores; W=$S/v3; B=$S/bench2
PY=~/doc-prueba/hefesto/.venv/bin/python
mkdir -p $V $W $B/res $B/log $S/log
cp ~/validadores/*.py ~/validadores/SE_analysis_config.json $V/
cp ~/v3/*.py $W/
ln -sfn ~/validadores/.venv $V/.venv
ln -sfn ~/validadores/.venv-sc $V/.venv-sc
$PY $S/strat_parches.py > $S/log/parches.log 2>&1 || { echo "parches FALLAN" > $S/log/fin; exit 1; }
# banco de privacidad y utilidad: como en lanza_bench2.sh, sobre el script ya parcheado
sed -e 's|dataset_name=f"banco3_{TAG}"|dataset_name=f"banco4_{TAG}"|g' \
    -e 's|conf_dir = Path.home() / "strat" / "v3" / "conf"|conf_dir = Path.home() / "strat" / "v3" / "conf4"|' \
    -e 's|^config\["calibration"\] = {\*\*cal, "bounds_file": str(bounds)}$|config["calibration"] = {**cal, "bounds_file": str(bounds)}\nif os.environ.get("EXTRA_UTILITY") == "1":\n    config["metrics"] = list(config["metrics"]) + ["utility.tts", "utility.trts", "utility.ttrs"]|' \
    $W/metis_banco_v3.py > $W/metis_banco_pu.py
grep -c "banco4_\|EXTRA_UTILITY\|conf4" $W/metis_banco_pu.py > $S/log/pu_sed.log

# 1. línea base (código cacd39a, cotas y configuración del 3 nov)
mkdir -p $S/base_bench
(cd ~/doc-base/metis && PYTHONPATH=/home/ubuntu/doc-base/metis taskset -c 0-1 $PY ~/v3/evalua_bench_metis.py eval telco $S/base_bench/m_telco_check.jsonl 0/20 > $S/base_bench/check.log 2>&1) &

# 2. generación
for d in rrhh cardio telco adult; do
  (cd ~/doc-split/metis && taskset -c 2-15 $PY $V/gen_casos.py $d $V/casos2 && taskset -c 2-15 $PY $V/gen_casos_het.py $d $V/casos2 \
   && taskset -c 2-15 $PY $V/gen_casos_e12.py $d $V/casos2 && $PY $V/a_csv.py $V/casos2/$d \
   && taskset -c 2-15 $PY $V/gen_estudio.py $d $V/casos2 $V/casos_estudio \
   && $PY $W/gen_privutil.py $V/casos2/$d $W/pu/casos && $PY $W/gen_matices.py $V/casos2/$d $W/matices/casos) > $S/log/gen_$d.log 2>&1 &
done
wait
grep -l "Traceback\|Error" $S/log/gen_*.log > $S/log/gen_errores.txt
tail -n 3 $S/log/gen_*.log > $S/log/gen_resumen.txt

# 3. calibraciones
mkdir -p $W/res/partes $W/res_estudio $W/pu/res $W/matices/res $W/limites $V/res2/partes $V/res_estudio $V/res_sc/partes
cd ~/doc-split/metis
pids=""
for d in rrhh cardio telco adult; do
  taskset -c 2-15 $PY $W/metis_banco_v3.py $V/casos2/$d $W/res/partes/cal_$d.jsonl 0 0 > $S/log/cal_$d.log 2>&1 &
  pids="$pids $!"
  PLANO=1 taskset -c 2-15 $PY $W/metis_banco_v3.py $V/casos2/$d $W/res/partes/cal_plano_$d.jsonl 0 0 > $S/log/cal_plano_$d.log 2>&1 &
  pids="$pids $!"
  EXTRA_UTILITY=1 taskset -c 2-15 $PY $W/metis_banco_pu.py $W/pu/casos/$d $W/pu/res/cal_$d.jsonl 0 0 > $S/log/cal_pu_$d.log 2>&1 &
  pids="$pids $!"
done
for d in cardio airbnb telco rrhh; do
  taskset -c 2-15 $PY $W/evalua_bench_metis.py cal $d > $B/log/cal_$d.log 2>&1 &
  pids="$pids $!"
done
wait $pids
tail -n 2 $S/log/cal_*.log $B/log/cal_*.log > $S/log/cal_resumen.txt

# 4. cola
{
  for k in 0 1 2 3 4; do for d in rrhh cardio telco adult; do
    echo "cd ~/doc-split/metis && $PY $W/metis_banco_v3.py $V/casos2/$d $W/res/partes/metis_${d}_$k.jsonl $k $((k+1))"
    echo "cd ~/doc-split/metis && PLANO=1 $PY $W/metis_banco_v3.py $V/casos2/$d $W/res/partes/plano_${d}_$k.jsonl $k $((k+1))"
    echo "cd ~/doc-split/metis && $PY $W/metis_banco_v3.py $V/casos_estudio/$d $W/res_estudio/metis_est_${d}_$k.jsonl $k $((k+1))"
    echo "cd $V && $PY metricas_nuevas.py casos2/$d res_estudio/nuevas_${d}_$k.jsonl $k $((k+1))"
    echo "cd $V && $PY metricas_nuevas.py casos_estudio/$d res_estudio/nuevas_est_${d}_$k.jsonl $k $((k+1))"
    echo "cd $V && .venv/bin/python evalua.py sdmetrics casos2/$d res2/partes/sdm_${d}_$k.jsonl $k $((k+1))"
    echo "cd $V && .venv/bin/python evalua.py syntheval casos2/$d res2/partes/se_${d}_$k.jsonl $k $((k+1))"
    echo "cd $V && CUDA_VISIBLE_DEVICES= .venv-sc/bin/python evalua.py synthcity casos2/$d res_sc/partes/sc_${d}_$k.jsonl $k $((k+1))"
    echo "cd ~/doc-split/metis && EXTRA_UTILITY=1 $PY $W/metis_banco_pu.py $W/pu/casos/$d $W/pu/res/m_${d}_$k.jsonl $k $((k+1))"
    echo "cd ~/doc-split/metis && $PY $W/metis_banco_v3.py $W/matices/casos/$d $W/matices/res/m_${d}_$k.jsonl $k $((k+1))"
  done; done
  for d in cardio airbnb telco rrhh; do
    n=4; [ $d = rrhh ] && n=2
    i=0; while [ $i -lt $n ]; do
      echo "cd ~/doc-split/metis && $PY $W/evalua_bench_metis.py eval $d $B/res/m_${d}_$i.jsonl $i/$n"
      i=$((i+1))
    done
  done
  for d in rrhh cardio telco adult; do
    echo "cd ~/doc-split/metis && $PY $W/efecto_etiquetas.py $V/casos2/$d $W/limites/etiquetas_$d.jsonl"
  done
} > $S/log/cola.txt
i=0
while IFS= read -r job; do
  i=$((i+1)); echo "$job > $S/log/t_$i.log 2>&1"
done < $S/log/cola.txt | taskset -c 2-15 xargs -P 14 -I{} sh -c "{}"

# 5. partes unidas
cat $W/res/partes/metis_*.jsonl > $W/res/metis.jsonl
cat $W/res/partes/plano_*.jsonl > $W/res/metis_plano_todos.jsonl
cat $V/res2/partes/sdm_*.jsonl > $V/res2/sdm2_todos.jsonl
cat $V/res2/partes/se_*.jsonl > $V/res2/syntheval.jsonl
cat $V/res_sc/partes/sc_*.jsonl > $V/res_sc/synthcity.jsonl
ln -sf $V/res2/sdm2_todos.jsonl $W/res/sdm2_todos.jsonl
ln -sf $V/res2/syntheval.jsonl $W/res/syntheval.jsonl
for f in $V/res_estudio/nuevas_*.jsonl; do ln -sf $f $W/res_estudio/; done
(cd ~/doc-split/metis && $PY $W/fuerza_trio.py $V/casos2 $V/casos_estudio > $W/limites/fuerza_trio.txt 2>&1)
grep -l "Traceback" $S/log/t_*.log > $S/log/errores.txt
wc -l $W/res/metis.jsonl $W/res/metis_plano_todos.jsonl $V/res2/*.jsonl $V/res_sc/synthcity.jsonl $B/res/m_*.jsonl $W/pu/res/m_*.jsonl $W/matices/res/m_*.jsonl > $S/log/recuento.txt
sed -i "s/^\(| METIS (METIS_AGENT) | reparto único estratificado .*\)| EN CURSO |$/\1| HECHO $(date -u '+%m-%d %H:%M') UTC |/" ~/COMPUTE_SHARING.md
echo FIN > $S/log/fin
