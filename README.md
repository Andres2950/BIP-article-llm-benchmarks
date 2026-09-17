# BIP-article-llm-benchmarks
Benchmarking de LLMs para tareas relacionadas con el procesamiento y consulta de documentación normativa del Centro Nacional de Detección Temprana de Cáncer Gastrointestinal (CNDTCG).

Este repositorio tiene datasetes, documentos de contexto, resultados y código utilizado para los experimientos hechos para un artículo subido al BIP.

## Descripción

El proyecto evalúa distintos modelos de lenguaje en preguntas sobre documentación del CNDTCG.

Los experimientos buscan considerar la calidad de respuesta y el rendimiento computacional de los modelos. Esto para analizar el comportamiento de distintos modelos con distintos niveles de cuantización.

Los benchmarks fueron diseñadps para ejecutarse localmente (probado solo con modelos pequeños) o en infraestructura de computación de alto rendimiento como el [supercomputador Kabré](https://kabre.cenat.ac.cr/).

## Objetivos

Los principales objetivos de este proyecto son:

- Evaluar distintos modelos sobre un dataset de preguntas basado en documentos del CNDTCG
- Comparar calidad de respuestas
- Evaluar consumo de recursos computacionales
- Analizar efecto de diferentes configuraciones

## Modelos evaluados

El repositorio contiene los  resultados correspondientes a los modelos que fueron evaluadios:

- DeepSeek R1 Distill Qwen F16
- DeepSeek R1 Distill Qwen Q6
- Llama Q4 + DeepSeek R1 Distill Qwen Q2, Llama Q8
- Qwen 3.5 4B

## Dataset

Las preguntas utilizadas para el benchmark se encuentran en:

```
datasets/
├── dataset.json
└── dataset_normativas.csv
```

El formato soportado por el programa es el csv. 
El arquivo JSON es el mismo dataset, solamente usando un formato distinto.

Los documentos utilizados como contexto para el benchmark se encuentran en:

```
context_docs/
```

Estos incluyen documentación normatica y administratica relacionada con el CNDTCG.
El dataset cuenta con 300 preguntas, 100 de cada tipo de pregunta (sí o no, respuesta corta, respuesta abierta).

## Metodología

De forma general, el proceso experimental consiste en:

- Hacer preguntas para el dataset a partir de los documentos de contexto
- Configurar el modelo
- Utilizar RAG para cargar los documentos, se carga ncomo un vector store
- Ejecutar benchmark para cada uno de los modelos
  - Obtener pregunta
  - Obtener los TOP K=5 chunks de los documentos más similares a la pregunta, utilizando FAISS
  - Generar respuesta
  - Medir rendimiento
- Evaluar resultados





