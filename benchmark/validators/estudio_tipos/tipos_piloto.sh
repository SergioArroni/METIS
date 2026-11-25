#!/bin/sh
# Estudio de los tipos (17 nov), piloto: defectos propios de cada tipo en adult, rrhh y rrhh recodificada, con
# cinco esquemas (formato, plano, sin ordinales, completo y el que METIS deduce solo).  Uso: tipos_piloto.sh "<datasets>"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/ubuntu/doc-split/metis
S=~/strat; T=$S/tipos; L=$T/log
PY=~/doc-prueba/hefesto/.venv/bin/python
DSS="${1:-adult rrhh rrhh_cod}"
mkdir -p $L $T/res $T/casos
sudo -n shutdown -c 2>/dev/null
rm -f $T/fin
echo "| METIS (METIS_AGENT) | estudio de los tipos: defectos propios de cada tipo con cinco esquemas, $DSS (tmux tip) | tmux | 14 (1 hilo) | $(date -u '+%Y-%m-%d %H:%M') | ~1 h | EN CURSO |" >> ~/COMPUTE_SHARING.md
# metis_banco_v3 con el esquema deducido por METIS (NIVEL=inf) y la configuración base tomada de la meta
$PY - <<'EOF'
import pathlib
s = pathlib.Path("/home/ubuntu/strat/v3/metis_banco_v3.py").read_text()
a = "DS if DS != 'adult' else 'rrhh'"
assert s.count(a) == 1
s = s.replace(a, "meta.get('config_base') or (DS if DS != 'adult' else 'rrhh')")
b = 'config["data"] = {"schema": schema,'
assert s.count(b) == 1
s = s.replace(b, 'if NIVEL == "inf":\n    from metis.infrastructure.typing import infer_schema\n'
              '    schema = dict(infer_schema(pd.read_pickle(CASES / "A.pkl")).schema)\n    TAG = f"{DS}_ninf"\n'
              '    print(DS, "esquema deducido", schema, flush=True)\n' + b)
pathlib.Path("/home/ubuntu/strat/tipos/metis_banco_t.py").write_text(s)
EOF
cd $S/validadores
for d in $DSS; do
  case $d in *_cod) $PY $T/gen_tipos.py ${d%_cod} $S/validadores/casos2 $T/casos cod ;; *) $PY $T/gen_tipos.py $d $S/validadores/casos2 $T/casos ;; esac
done > $L/gen.log 2>&1
niveles() { case $1 in *_cod) echo "0 3 inf" ;; *) echo "0 1 2 3 inf" ;; esac; }
orden() {  # dataset nivel k_desde k_hasta etiqueta
  case $2 in 1) E="PLANO=1" ;; 3) E="X=1" ;; *) E="NIVEL=$2" ;; esac
  echo "cd ~/doc-split/metis && $E $PY $T/metis_banco_t.py $T/casos/$1 $T/res/$5.jsonl $3 $4 > $L/$5.log 2>&1"
}
cd ~/doc-split/metis
for d in $DSS; do for n in $(niveles $d); do orden $d $n 0 0 cal_${d}_n$n; done; done | xargs -P 13 -I{} sh -c "{}"
for k in 0 1 2 3 4; do for d in $DSS; do for n in $(niveles $d); do orden $d $n $k $((k+1)) ${d}_n${n}_$k; done; done; done | xargs -P 14 -I{} sh -c "{}"
grep -l "Traceback" $L/*.log > $L/errores.txt
cat $T/res/*.jsonl | wc -l > $L/recuento.txt
sed -i "s/^\(| METIS (METIS_AGENT) | estudio de los tipos: .*\)| EN CURSO |$/\1| HECHO $(date -u '+%m-%d %H:%M') UTC |/" ~/COMPUTE_SHARING.md
echo FIN > $T/fin
ajenos=$(ps -eo pcpu,args --no-headers | awk '$1>20' | grep -v -e strat -e validadores -e doc-split -e "ps -eo" | wc -l)
[ "$ajenos" -gt 0 ] || sudo -n shutdown -h +180
