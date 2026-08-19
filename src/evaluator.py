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
    t = texto.lower()
    if t.startswith("sí") or t.startswith("si,") or t.startswith("si.") or t.startswith("si "):
        return True
    if t.startswith("no"):
        return False

    return None


def acierto_binario(esperada: str, obtenida: str):
    e = extraer_si_no(esperada)
    o = extraer_si_no(obtenida)
    if e is None or o is None:
        return None
    return 1.0 if e == o else 0.0

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


def resumen_bagging(df_completo: pd.DataFrame) -> dict:

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

    # Metricas calidad
    metricas_calidad = [
        "f1_token", "rouge1", "rouge2",
        "meteor", "bertscore",
    ]
    promedio_calidad_por_bag = (
        df_completo
        .groupby(["model", "question_type", "bag_id"])[metricas_calidad]
        .mean()
        .reset_index()
    )
    stats_calidad_entre_bags = (
        promedio_calidad_por_bag
        .groupby(["model", "question_type"])[metricas_calidad]
        .agg(["mean", "std"])
    )
    minmax_calidad_crudo = (
        df_completo
        .groupby(["model", "question_type"])[metricas_calidad]
        .agg(["min", "max"])
    )
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

    resumenes = resumen_bagging(df_completo)

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
