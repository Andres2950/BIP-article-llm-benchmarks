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
from rouge_score.tokenizers import Tokenizer

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


def aislar_respuesta(texto: str) -> str:
    """Texto después del último </think>, cortado en el siguiente <think>.

    Es el recorte con el que se rescata la respuesta del razonamiento.
    Si el modelo no cierra el bloque, el resultado queda vacío.
    """
    t = str(texto)
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    t = re.split(r"<think>", t, maxsplit=1, flags=re.IGNORECASE)[0]
    return t.strip()


def tiene_razonamiento(texto: str) -> float:
    """1 si el texto guardado trae un marcador de razonamiento, 0 si no.

    Los marcadores son <think>, </think> y 'Thinking Process'. En las
    preguntas sí/no el CSV ya guarda solo la etiqueta extraída, así que
    esta marca casi no aparece ahí aunque el modelo haya razonado.
    """
    t = str(texto).lower()
    if "</think>" in t or "<think>" in t or "thinking process" in t:
        return 1.0
    return 0.0


def calcular_balanced_accuracy(df: pd.DataFrame, resultado_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas de sí/no por modelo y bag.

    balanced_accuracy reparte 0.5 entre cada clase; una respuesta sin
    etiqueta cuenta como error. recall_si y recall_no usan el mismo
    criterio. tasa_parseo es la fracción con etiqueta extraíble.
    proporcion_si es la fracción cuya etiqueta es sí; las que no parsean
    no cuentan como sí.
    """
    columnas = [
        "model", "question_type", "bag_id", "balanced_accuracy",
        "recall_si", "recall_no", "tasa_parseo", "proporcion_si",
    ]
    vacio = pd.DataFrame(columns=columnas)
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
        "pred": [extraer_si_no(str(respuesta)) for respuesta in df["response"]],
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

        clases = list(grupo["clase"])
        preds = list(grupo["pred"])
        n_si = sum(clase is True for clase in clases)
        n_no = sum(clase is False for clase in clases)
        aciertos_si = sum(pred is True and clase is True for pred, clase in zip(preds, clases))
        aciertos_no = sum(pred is False and clase is False for pred, clase in zip(preds, clases))
        n = len(grupo)
        registros.append({
            "model": modelo,
            "question_type": "yes_no",
            "bag_id": bag_id,
            "balanced_accuracy": score,
            "recall_si": (aciertos_si / n_si) if n_si else None,
            "recall_no": (aciertos_no / n_no) if n_no else None,
            "tasa_parseo": sum(pred is not None for pred in preds) / n if n else None,
            "proporcion_si": sum(pred is True for pred in preds) / n if n else None,
        })

    return pd.DataFrame(registros)

#########################


class TokenizerEspanol(Tokenizer):
    """El tokenizer de rouge_score borra todo lo que no es [a-z0-9].

    Eso parte cáncer en c + ncer y detección en detecci + n. Aquí se
    conservan las letras Unicode, incluidos acentos y ñ.
    """

    _separador = re.compile(r"[^\w]+", flags=re.UNICODE)

    def tokenize(self, text):
        text = text.lower().replace("_", " ")
        text = self._separador.sub(" ", text)
        return [token for token in text.split() if token]


_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2"],
    use_stemmer=False,
    tokenizer=TokenizerEspanol(),
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

        aislada = aislar_respuesta(obtenida)
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
            "razonamiento": tiene_razonamiento(obtenida),
            "f1_aislado": None,
            "rouge1_aislado": None,
            "rouge2_aislado": None,
        }

        if tipo in ("yes_no", "short_answer"):
            r["f1_token"] = f1_score(esperada, obtenida)
            r["f1_aislado"] = f1_score(esperada, aislada)
            if tipo == "yes_no":
                r["acierto_binario"] = acierto_binario(esperada, obtenida)

        elif tipo == "open_ended":
            r.update(calcular_rouge(esperada, obtenida))
            r["meteor"] = calcular_meteor(esperada, obtenida)
            aislado = calcular_rouge(esperada, aislada)
            r["rouge1_aislado"] = aislado["rouge1"]
            r["rouge2_aislado"] = aislado["rouge2"]

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


def _throughput_por_bag(df_completo: pd.DataFrame) -> pd.DataFrame:
    """Tokens por segundo de cada bag: tokens emitidos / segundos de generación.

    No es el promedio de los cocientes por pregunta. Una pregunta corta
    pesaría igual que una larga, y el número del paper es el cociente de
    los totales.
    """
    suma = (
        df_completo
        .groupby(["model", "bag_id"])[["output_tokens", "total_duration_sec"]]
        .sum()
    )
    duracion = suma["total_duration_sec"].where(suma["total_duration_sec"] > 0)
    tasa = (suma["output_tokens"] / duracion).rename("tokens_per_sec")
    return tasa.reset_index()


def resumen_bagging(df_completo: pd.DataFrame, balanced_por_bag: pd.DataFrame | None = None) -> dict:

    # Metricas rendimdiento. tokens_per_sec se agrega aparte, por bag.
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
    tasas = _throughput_por_bag(df_completo)
    stats_tasa = tasas.groupby("model")["tokens_per_sec"].agg(["mean", "std", "min", "max"])
    stats_tasa.columns = pd.MultiIndex.from_product([["tokens_per_sec"], ["mean", "std", "min", "max"]])
    rendimiento = pd.concat([rendimiento, stats_tasa], axis=1)
    rendimiento = rendimiento.reindex(
        columns=pd.MultiIndex.from_product(
            [metricas_rendimiento[:2] + ["tokens_per_sec"] + metricas_rendimiento[2:],
             ["mean", "std", "min", "max"]]
        )
    ).round(4)

    # Por fila: calidad léxica, marca de razonamiento y rescoring del
    # texto aislado. Por bag, y solo en sí/no: balanced_accuracy, recall
    # por clase, tasa de parseo y proporción de sí.
    metricas_fila = [
        "f1_token", "rouge1", "rouge2",
        "meteor", "bertscore",
        "razonamiento", "f1_aislado", "rouge1_aislado", "rouge2_aislado",
    ]
    metricas_por_bag = [
        "balanced_accuracy", "recall_si", "recall_no",
        "tasa_parseo", "proporcion_si",
    ]
    metricas_calidad = metricas_fila + metricas_por_bag
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
        for metrica in metricas_por_bag:
            promedio_calidad_por_bag[metrica] = pd.NA

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
        minmax_bag = (
            balanced_por_bag
            .groupby(["model", "question_type"])[metricas_por_bag]
            .agg(["min", "max"])
        )
        minmax_calidad_crudo = pd.concat([minmax_calidad_crudo, minmax_bag], axis=1)
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


def unir_por_fila(df: pd.DataFrame, resultado_df: pd.DataFrame) -> pd.DataFrame:
    """Pega los recursos al lado de las métricas, en el mismo orden de filas.

    No se une por question_id: dentro de una bag ese id se repite, y un
    merge por esa clave duplica filas. evaluar_dataframe recorre el CSV
    en orden, así que la fila i de un lado es la fila i del otro.
    """
    if len(df) != len(resultado_df):
        raise ValueError("El detalle de métricas no coincide con el CSV de entrada")

    claves = ["bag_id", "model", "question_id", "question_type"]
    izquierda = df[claves].reset_index(drop=True).astype(str)
    derecha = resultado_df[claves].reset_index(drop=True).astype(str)
    if not izquierda.equals(derecha):
        raise ValueError(
            "El detalle de métricas no está en el mismo orden que el CSV de entrada"
        )

    recursos = [
        "total_duration_sec", "output_tokens", "cpu_avg", "memory_avg",
        "gpu_util_avg", "gpu_mem_avg",
    ]
    return pd.concat(
        [resultado_df.reset_index(drop=True), df[recursos].reset_index(drop=True)],
        axis=1,
    )


def ejecutar_evaluacion(csv_path: str, out_dir: str | None = None) -> dict:
    asegurar_recursos_nltk()
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out = out_dir / csv_path.stem

    df = pd.read_csv(csv_path)
    resultado_df = evaluar_dataframe(df)
    balanced_por_bag = calcular_balanced_accuracy(df, resultado_df)

    df_completo = unir_por_fila(df, resultado_df)
    duracion = df_completo["total_duration_sec"].astype(float)
    df_completo["tokens_per_sec"] = (
        df_completo["output_tokens"].astype(float) / duracion.where(duracion > 0)
    )
    resultado_df = df_completo.drop(columns=[
        "total_duration_sec", "output_tokens", "cpu_avg", "memory_avg",
        "gpu_util_avg", "gpu_mem_avg",
    ])
    ruta_detalle = f"{base_out}_metricas.csv"
    resultado_df.to_csv(ruta_detalle, index=False)
    print(f"Resultados por fila guardados en {ruta_detalle}")

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
