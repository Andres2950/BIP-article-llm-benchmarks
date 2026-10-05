import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    bertscore_activo = True
except ImportError:
    bertscore_activo = False

STOPWORDS = {"un", "una", "el", "la", "los", "las", "de", "del", "y", "a", "en"}


def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(ch for ch in texto if ch not in string.punctuation)
    texto = re.sub(r"\s+", " ", texto.strip())
    tokens = [t for t in texto.split() if t not in STOPWORDS]
    return " ".join(tokens)


def f1_score(esperada: str, obtenida: str) -> float:
    esperada_tokens = normalizar_texto(esperada).split()
    obtenida_tokens = normalizar_texto(obtenida).split()

    comunes = Counter(esperada_tokens) & Counter(obtenida_tokens)
    num_comunes = sum(comunes.values())

    if num_comunes == 0 or not obtenida_tokens or not esperada_tokens:
        return 0.0

    precision = num_comunes / len(obtenida_tokens)
    recall = num_comunes / len(esperada_tokens)
    return 2 * (precision * recall) / (precision + recall)


def extraer_si_no(texto: str):
    t = texto.lower().strip()
    
    match = re.search(r'\b(sí|si|no)\b', t)
    if match:
        palabra = match.group(1)
        if palabra == 'si':
            return True
        return palabra == 'sí'
    return None

def acierto_binario(esperada: str, obtenida: str):
    e = extraer_si_no(esperada)
    o = extraer_si_no(obtenida)
    if e is None or o is None:
        return None
    return 1.0 if e == o else 0.0


def _es_acierto(valor) -> float:
    if valor is None or pd.isna(valor):
        return 0.0
    return float(valor)


def calcular_pesos_si_no(df: pd.DataFrame) -> dict:
    """Pesos por bag para que sí y no aporten lo mismo.

    Se calculan una vez con las respuestas esperadas del primer modelo:
    las bags son las mismas para todos. Cada clase presente reparte 1/n_clases
    entre sus filas (0.5 / n_clase cuando hay sí y no).
    """
    yes_no = df[df["question_type"] == "yes_no"]
    if yes_no.empty:
        return {}

    modelo_ref = yes_no["model"].iloc[0]
    referencia = yes_no[yes_no["model"] == modelo_ref]
    pesos = {}

    for bag_id, grupo in referencia.groupby("bag_id"):
        n_si, n_no, pesos_bag = _pesos_de_grupo(grupo)
        pesos[int(bag_id)] = {"n_si": n_si, "n_no": n_no, "pesos": pesos_bag}
        print(
            f"Pesos bag {int(bag_id)} (desde {modelo_ref}): "
            f"sí={n_si} peso={pesos_bag.get(True)}, no={n_no} peso={pesos_bag.get(False)}"
        )

    for modelo, grupo_modelo in yes_no.groupby("model"):
        if modelo == modelo_ref:
            continue
        for bag_id, grupo in grupo_modelo.groupby("bag_id"):
            n_si, n_no, _ = _pesos_de_grupo(grupo)
            ref = pesos.get(int(bag_id))
            if ref is None or n_si != ref["n_si"] or n_no != ref["n_no"]:
                raise ValueError(
                    f"La bag {int(bag_id)} de {modelo} no tiene la misma mezcla "
                    f"sí/no que {modelo_ref} ({n_si} sí, {n_no} no)"
                )

    return pesos


def _pesos_de_grupo(grupo: pd.DataFrame):
    clases = [extraer_si_no(str(respuesta)) for respuesta in grupo["expected_answer"]]
    n_si = sum(clase is True for clase in clases)
    n_no = sum(clase is False for clase in clases)
    n_clases = (n_si > 0) + (n_no > 0)
    pesos_bag = {}
    if n_clases:
        parte = 1.0 / n_clases
        if n_si:
            pesos_bag[True] = parte / n_si
        if n_no:
            pesos_bag[False] = parte / n_no
    return n_si, n_no, pesos_bag


def calcular_balanced_accuracy(df: pd.DataFrame, resultado_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy balanceada por modelo y bag, reutilizando los pesos de la bag."""
    vacio = pd.DataFrame(columns=["model", "question_type", "bag_id", "balanced_accuracy"])
    pesos = calcular_pesos_si_no(df)
    if not pesos:
        return vacio

    if len(df) != len(resultado_df):
        raise ValueError("El detalle de métricas no coincide con el CSV de entrada")

    filas = pd.DataFrame({
        "bag_id": df["bag_id"].to_numpy(),
        "model": df["model"].to_numpy(),
        "question_type": df["question_type"].to_numpy(),
        "clase": [extraer_si_no(str(respuesta)) for respuesta in df["expected_answer"]],
        "acierto_binario": resultado_df["acierto_binario"].to_numpy(),
    })
    filas = filas[filas["question_type"] == "yes_no"]

    registros = []
    for (modelo, bag_id), grupo in filas.groupby(["model", "bag_id"], sort=False):
        pesos_bag = pesos.get(int(bag_id), {}).get("pesos", {})
        if not pesos_bag:
            score = None
        else:
            score = 0.0
            for clase, acierto in zip(grupo["clase"], grupo["acierto_binario"]):
                peso = pesos_bag.get(clase)
                if peso is None:
                    continue
                score += peso * _es_acierto(acierto)
        registros.append({
            "model": modelo,
            "question_type": "yes_no",
            "bag_id": bag_id,
            "balanced_accuracy": score,
        })

    return pd.DataFrame(registros)

#########################


_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2"],
    use_stemmer=False
)


def calcular_rouge(esperada: str, obtenida: str) -> dict:
    scores = _rouge_scorer.score(esperada, obtenida)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure
    }


def calcular_meteor(esperada: str, obtenida: str) -> float:
    ref_tokens = word_tokenize(esperada.lower(), language="spanish")
    hyp_tokens = word_tokenize(obtenida.lower(), language="spanish")
    return meteor_score([ref_tokens], hyp_tokens)


################

def calcular_bert_batch(esperadas: list, obtenidas: list):
    if not bertscore_activo:
        return [None] * len(esperadas)

    try:
        P, R, F1 = bert_score_fn(
            obtenidas,
            esperadas,
            lang="es",
            verbose=False
        )
        return F1.tolist()
    except Exception as e:
        print(f"NO SE PUDO CALCULAR BERTScore: {e}")
        print("Se deja vacío en el reporte")
        return [None] * len(esperadas)


def evaluar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    filas = df.to_dict("records")
    resultados = []

    for fila in filas:
        esperada = str(fila["expected_answer"])
        obtenida = str(fila["response"])
        tipo = fila["question_type"]

        r = {
            "bag_id": fila["bag_id"],
            "model": fila["model"],
            "question_id": fila["question_id"],
            "question_type": tipo,
            "f1_token": None,
            "acierto_binario": None,
            "rouge1": None,
            "rouge2": None,
            "meteor": None,
            "bertscore": None,
        }

        if tipo in ("yes_no", "short_answer"):
            r["f1_token"] = f1_score(esperada, obtenida)
            if tipo == "yes_no":
                r["acierto_binario"] = acierto_binario(esperada, obtenida)

        elif tipo == "open_ended":
            r.update(calcular_rouge(esperada, obtenida))
            r["meteor"] = calcular_meteor(esperada, obtenida)

        resultados.append(r)

    resultado_df = pd.DataFrame(resultados)

    # BersScore se calcula en batch, solo para respuesta abierta
    preguntas_abiertas = df["question_type"] == "open_ended"
    if preguntas_abiertas.any():
        esperadas = df.loc[preguntas_abiertas, "expected_answer"].astype(str).tolist()
        obtenidas = df.loc[preguntas_abiertas, "response"].astype(str).tolist()
        berts = calcular_bert_batch(esperadas, obtenidas)
        resultado_df.loc[preguntas_abiertas.values, "bertscore"] = berts

    return resultado_df


def resumen_modelos(resultado_df: pd.DataFrame) -> pd.DataFrame:
    metricas = ["f1_token", "acierto_binario",
                "rouge1", "rouge2", "meteor", "bertscore"]

    return (
        resultado_df
        .groupby(["model", "question_type"])[metricas]
        .agg(["mean", "std", "count"])
        .round(4)
    )


def resumen_bagging(df_completo: pd.DataFrame, balanced_por_bag: pd.DataFrame | None = None) -> dict:

    # Metricas rendimdiento
    metricas_rendimiento = [
        "total_duration_sec", "output_tokens",
        "cpu_avg", "memory_avg", "gpu_util_avg", "gpu_mem_avg"
    ]

    promedio_por_bag = (
            df_completo
            .groupby(["model", "bag_id"])[metricas_rendimiento]
            .mean()
            .reset_index()
    )
    stats_entre_bags = (
        promedio_por_bag
        .groupby("model")[metricas_rendimiento]
        .agg(["mean", "std"])
    )
    minmax_crudo = (
        df_completo
        .groupby("model")[metricas_rendimiento]
        .agg(["min", "max"])
    )
    rendimiento = pd.concat([stats_entre_bags, minmax_crudo], axis=1)
    rendimiento = rendimiento.reindex(
        columns=pd.MultiIndex.from_product(
            [metricas_rendimiento, ["mean", "std", "min", "max"]]
        )
    ).round(4)

    # Metricas calidad. balanced_accuracy ya viene por bag, no por fila.
    metricas_fila = [
        "f1_token", "rouge1", "rouge2",
        "meteor", "bertscore",
    ]
    metricas_calidad = metricas_fila + ["balanced_accuracy"]
    promedio_calidad_por_bag = (
        df_completo
        .groupby(["model", "question_type", "bag_id"])[metricas_fila]
        .mean()
        .reset_index()
    )
    if balanced_por_bag is not None and not balanced_por_bag.empty:
        promedio_calidad_por_bag = promedio_calidad_por_bag.merge(
            balanced_por_bag,
            on=["model", "question_type", "bag_id"],
            how="left",
        )
    else:
        promedio_calidad_por_bag["balanced_accuracy"] = pd.NA

    stats_calidad_entre_bags = (
        promedio_calidad_por_bag
        .groupby(["model", "question_type"])[metricas_calidad]
        .agg(["mean", "std"])
    )
    minmax_calidad_crudo = (
        df_completo
        .groupby(["model", "question_type"])[metricas_fila]
        .agg(["min", "max"])
    )
    if balanced_por_bag is not None and not balanced_por_bag.empty:
        minmax_bal = (
            balanced_por_bag
            .groupby(["model", "question_type"])["balanced_accuracy"]
            .agg(["min", "max"])
        )
        minmax_bal.columns = pd.MultiIndex.from_product(
            [["balanced_accuracy"], ["min", "max"]]
        )
        minmax_calidad_crudo = pd.concat([minmax_calidad_crudo, minmax_bal], axis=1)
    calidad = pd.concat([stats_calidad_entre_bags, minmax_calidad_crudo], axis=1)
    calidad = calidad.reindex(
        columns=pd.MultiIndex.from_product(
            [metricas_calidad, ["mean", "std", "min", "max"]]
        )
    ).round(4)
    calidad = calidad.dropna(axis=0, how="all")

    return {"rendimiento": rendimiento, "calidad": calidad}


def asegurar_recursos_nltk():
    recursos = {
        "tokenizers/punkt_tab": "punkt_tab",
        "copora/wordnet.zip": "wordnet",
        "copora/omw-1.4.zip": "omw-1.4",
    }

    for ruta, nombre in recursos.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            print(f"Descargando recurso NLTK faltante: {nombre}")
            nltk.download(nombre, quiet=True)


def ejecutar_evaluacion(csv_path: str, out_dir: str | None = None) -> dict:
    asegurar_recursos_nltk()
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out = out_dir / csv_path.stem

    df = pd.read_csv(csv_path)
    resultado_df = evaluar_dataframe(df)
    balanced_por_bag = calcular_balanced_accuracy(df, resultado_df)
    ruta_detalle = f"{base_out}_metricas.csv"
    resultado_df.to_csv(ruta_detalle, index=False)
    print(f"Resultados por fila guardados en {ruta_detalle}")

    columnas_rendimiento = [
        "bag_id", "model", "question_id", "total_duration_sec",
        "output_tokens", "cpu_avg", "memory_avg",
        "gpu_util_avg", "gpu_mem_avg"
    ]
    df_completo = resultado_df.merge(
        df[columnas_rendimiento], on=["bag_id", "model", "question_id"]
    )

    resumenes = resumen_bagging(df_completo, balanced_por_bag)

    ruta_rendimiento = f"{base_out}_rendimiento.csv"
    ruta_calidad = f"{base_out}_calidad.csv"
    resumenes["rendimiento"].to_csv(ruta_rendimiento)
    resumenes["calidad"].to_csv(ruta_calidad)


    print("\n=== Rendimiento por modelo (mean/std entre bags, min/max crudo) ===")
    print(resumenes["rendimiento"])
    print(f"\nGuardado en: {ruta_rendimiento}")

    print("\n=== Calidad por modelo y tipo de pregunta ===")
    print(resumenes["calidad"])
    print(f"\nGuardado en: {ruta_calidad}")

    return {
        "detalle": resultado_df,
        "rendimiento": resumenes["rendimiento"],
        "calidad": resumenes["calidad"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default=None, help="Carpeta de salida, por defecto la misma que csv_path")
    args = parser.parse_args()
    ejecutar_evaluacion(args.csv_path, args.out_dir)


if __name__ == "__main__":
    main()
