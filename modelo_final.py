"""
Reto: Propensión de conversión de clientes — MODELO FINAL OPTIMIZADO
=====================================================================
Ejecutar en PyCharm (o en terminal):  python modelo_final.py

Requisitos (una sola vez, en la terminal de PyCharm):
    pip install pandas numpy scikit-learn lightgbm scipy

Coloca train.csv y test.csv en la MISMA carpeta que este archivo
(o cambia DATA_DIR abajo). Al terminar se crea submission_optimizado.csv.

Mejoras frente a la versión 1 (Gini 0,2515 → ~0,26 en validación):
  1. Variables de REGLAS de negocio descubiertas en los datos (interacciones fuertes).
  2. Se elimina n_mes: la propensión de cada cliente no cambia con el tiempo y n_mes solo añadía ruido.
  3. LightGBM con árboles más pequeños y más regularizados (menos sobreajuste).
  4. Promedio de varias semillas aleatorias (predicciones más estables).
  5. Ensamble por rankings con una regresión logística con splines + reglas.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta de este script
VALIDAR = True            # True: mide el Gini en meses pasados antes de entregar (tarda unos minutos más)
MESES_VALID = [202609, 202610, 202611]  # meses usados como "examen" en la validación temporal
N_SEMILLAS = 5            # cuántas semillas promediar en LightGBM
SALIDA = "submission_optimizado.csv"

# ----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ----------------------------------------------------------------------------
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"train {train.shape} | test {test.shape}")

df = pd.concat([train, test], ignore_index=True)  # unimos para crear variables igual en ambos

# ----------------------------------------------------------------------------
# 2. VARIABLES
# ----------------------------------------------------------------------------
cats = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"]
bools = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"]
for c in bools:
    df[c] = df[c].astype(int)                 # True/False -> 1/0
for c in cats:
    df[c] = df[c].astype("category")          # texto -> categoría (LightGBM lo maneja directo)

# NOTA: dias_ultima_interaccion NO se usa (tiene drift: en diciembre es ruido puro).
nums = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
        "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses",
        "visitas_web_ultimos_90_dias", "distancia_sucursal_km", "dia_preferido_pago"]

# Variables económicas
df["deuda"] = df.ratio_deuda_ingresos * df.ingresos                                  # deuda absoluta
df["saldo_ingresos"] = df.saldo_promedio / df.ingresos                               # liquidez relativa
df["prod_por_anio"] = df.numero_productos / (df.antiguedad_cuenta_meses / 12 + 1)    # ritmo de vinculación

# Reglas descubiertas en el análisis (tasas de conversión observadas en train):
#   riesgo bajo + 3 o más productos                         -> ~24 %  (promedio general 15 %)
#   riesgo bajo + móvil activo + tarjeta + 2 o más productos -> ~26-39 %
#   riesgo alto + más de 180 días sin transaccionar          -> ~5 %   (frente a ~13 % con <=180 días)
riesgo_bajo = df.banda_riesgo == "low"
riesgo_alto = df.banda_riesgo == "high"
movil_y_tarjeta = (df.activo_movil == 1) & (df.tiene_tarjeta_credito == 1)
df["r_bajo_3prod"] = (riesgo_bajo & (df.numero_productos >= 3)).astype(int)
df["r_bajo_movil_tarjeta"] = (riesgo_bajo & movil_y_tarjeta & (df.numero_productos >= 2)).astype(int)
df["r_alto_inactivo"] = (riesgo_alto & (df.dias_ultima_transaccion > 180)).astype(int)
df["inactivo_180"] = (df.dias_ultima_transaccion > 180).astype(int)
REGLAS = ["r_bajo_3prod", "r_bajo_movil_tarjeta", "r_alto_inactivo", "inactivo_180"]

F_LGB = nums + bools + cats + ["deuda", "saldo_ingresos", "prod_por_anio"] + REGLAS

# Para la logística: numéricas continuas con splines (curvas flexibles)
NUM_SPLINE = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "saldo_promedio",
              "dias_ultima_transaccion", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
              "distancia_sucursal_km"]
CAT_GLM = cats + ["numero_productos"]
F_GLM = NUM_SPLINE + CAT_GLM + bools + REGLAS

# ----------------------------------------------------------------------------
# 3. MODELOS
# ----------------------------------------------------------------------------
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.01,  # muchos árboles, cada uno aporta poco
    num_leaves=4,                           # árboles pequeños: captan interacciones simples sin memorizar ruido
    min_child_samples=400,                  # cada hoja necesita al menos 400 clientes
    subsample=0.8, subsample_freq=1,        # cada árbol ve el 80 % de las filas
    colsample_bytree=0.7,                   # y el 70 % de las variables
    reg_lambda=10,                          # regularización fuerte
    verbose=-1, n_jobs=-1,
)


def modelo_lgb(a, b):
    """LightGBM promediando las probabilidades de N_SEMILLAS semillas distintas."""
    preds = []
    for s in range(N_SEMILLAS):
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=s)
        m.fit(a[F_LGB], a.objetivo)
        preds.append(m.predict_proba(b[F_LGB])[:, 1])
    return np.mean(preds, axis=0)


def modelo_glm(a, b):
    """Regresión logística con splines + variables de reglas."""
    pre = ColumnTransformer([
        ("spl", make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3)), NUM_SPLINE),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_GLM),
        ("bin", "passthrough", bools + REGLAS),
    ])
    m = make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000))
    m.fit(a[F_GLM], a.objetivo)
    return m.predict_proba(b[F_GLM])[:, 1]


def ensamble(p_lgb, p_glm):
    """Promedio de rankings (LightGBM pesa el doble). El Gini solo mira el orden.
    Para que la entrega siga siendo una PROBABILIDAD, al cliente en la posición k del
    ranking del ensamble se le asigna la k-ésima probabilidad (ordenada) de LightGBM:
    el orden es el del ensamble y los valores son probabilidades realistas."""
    r = 2 * rankdata(p_lgb) + rankdata(p_glm)
    posicion = rankdata(r, method="ordinal").astype(int) - 1
    return np.sort(p_lgb)[posicion]


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


# ----------------------------------------------------------------------------
# 4. VALIDACIÓN TEMPORAL (opcional)
# ----------------------------------------------------------------------------
trn = df[df.mes <= 202611]
if VALIDAR:
    print("\nValidación temporal: entrenar con meses anteriores y predecir el mes indicado")
    tabla = []
    for mes in MESES_VALID:
        t0 = time.time()
        a, b = trn[trn.mes < mes], trn[trn.mes == mes]
        pl, pg = modelo_lgb(a, b), modelo_glm(a, b)
        fila = dict(mes=mes, lgb=gini(b.objetivo, pl), glm=gini(b.objetivo, pg),
                    ensamble=gini(b.objetivo, ensamble(pl, pg)))
        tabla.append(fila)
        print(f"  {mes}: LGB {fila['lgb']:.4f} | GLM {fila['glm']:.4f} | "
              f"ENSAMBLE {fila['ensamble']:.4f}  ({time.time() - t0:.0f}s)")
    tabla = pd.DataFrame(tabla).set_index("mes")
    print("\nGini medio:\n", tabla.mean().round(4).to_string())
    print("(Referencia versión 1, LightGBM simple: 0.2515)")

# ----------------------------------------------------------------------------
# 5. ENTRENAMIENTO FINAL Y ENTREGA
# ----------------------------------------------------------------------------
print("\nEntrenando con enero a noviembre y prediciendo diciembre...")
dic = df[df.mes == 202612]
pred = ensamble(modelo_lgb(trn, dic), modelo_glm(trn, dic))

entrega = pd.DataFrame({"id_cliente": dic.id_cliente.values, "prediccion": pred})
assert list(entrega.columns) == ["id_cliente", "prediccion"]
assert len(entrega) == len(test) and (entrega.id_cliente.values == test.id_cliente.values).all()
assert entrega.prediccion.between(0, 1).all() and entrega.prediccion.notna().all()
entrega.to_csv(os.path.join(DATA_DIR, SALIDA), index=False)
print(f"Listo: {SALIDA} con {len(entrega)} filas")
print(entrega.prediccion.describe().round(4).to_string())
