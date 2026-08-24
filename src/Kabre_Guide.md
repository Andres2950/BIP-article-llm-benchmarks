# Guía para ejecutar benchmarks de LLM en Kabré

Este documento describe el procedimiento completo para ejecutar un benchmark de modelos de lenguaje (LLM) en el supercomputador Kabré, utilizando el repositorio `BIP-article-llm-benchmarks` y el sistema de colas SLURM.

---

## 1. Conexión a Kabré

Desde la computadora local, abra una terminal y ejecute:

```bash
ssh -p 22022 usuario@kabre.cenat.ac.cr
```

- Ingrese la contraseña cuando se solicite.
- Si es la primera vez, presiona Enter tres veces cuando se pregunte por la creación de llaves SSH.
- Ingrese el código correspondiente al segundo factor (OATH).

---

## 2. Obtener el repositorio del proyecto

### Opción A: Clonar desde GitHub (repositorio público)

```bash
cd /work/$USER
git clone https://github.com/Andres2950/BIP-article-llm-benchmarks BIP-article-llm-benchmarks
```

### Opción B: Copiar desde local con SCP

**Ejecute esto en su computadora local (no en Kabré):**

```bash
scp -P 22022 -r /ruta/local/BIP-article-llm-benchmarks usuario@kabre.cenat.ac.cr:/work/usuario/
```

Reemplace `/ruta/local/` con la ubicación real del proyecto en la PC.

---

## 3. Navegar al proyecto

```bash
cd /work/$USER/BIP-article-llm-benchmarks
```

---

## 4. Cargar el módulo de Miniconda

Kabré utiliza módulos de ambiente para gestionar el software. Para disponer de `conda`:

```bash
module load miniconda/3
```

Verifique que conda esté disponible:

```bash
which conda
conda --version
```

---

## 5. Crear y activar el entorno de Python

Cree un entorno virtual con Python 3.10 (puede elegir otro nombre si lo deseas):

```bash
conda create -n llm-env python=3.10 -y
```

Active el entorno:

```bash
source activate llm-env
```

**Nota:** En los nodos de cómputo (dentro de scripts SLURM) se debe usar `source activate llm-env`, ya que `conda activate` puede fallar sin la inicialización completa del shell.

---

## 6. Instalar dependencias del proyecto

Dentro del entorno activado, instala los paquetes requeridos:

```bash
pip install -r requirements.txt
pip install llama-cpp-python   # si no está en el requirements
```

---

## 7. Configurar los modelos a evaluar

Edita el archivo que contiene la definición de los modelos (generalmente `src/main.py` o `config.py`):

```bash
nano src/main.py
```

Busque la lista de `ModelConfig` y agregue o descomente las entradas de los modelos que desea probar. Ejemplo:

```python
# Modelo de 11.3 GB
ModelConfig(
    name="DeepSeek-R1-Distill-Qwen-14B-Q6_K",
    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    format="gguf",
    filename="DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf",
    context_size=16384,
    gpu_layers=-1,
),
# Modelo de 18.4 GB
ModelConfig(
    name="Qwen3.5-35B-A3B-Q4_K_M",
    repo_id="lmstudio-community/Qwen3.5-35B-A3B-GGUF",
    format="gguf",
    filename="Qwen3.5-35B-A3B-Q4_K_M.gguf",
    context_size=16384,
    gpu_layers=-1,
),
```

Guarde los cambios (Ctrl+O, Enter, Ctrl+X).

---

## 8. Crear los scripts SLURM

Cree la carpeta `slurm` si no existe:

```bash
mkdir -p slurm
```

### Script de prueba (smoke_test.slurm)

Este script ejecuta una sola pregunta (ID 54) en 30 minutos para verificar que todo funciona.

```bash
nano slurm/smoke_test.slurm
```

Contenido:

```bash
#!/bin/bash
#SBATCH --job-name=llm_benchmark_smoketest
#SBATCH --output=./logs/smoke%j.out
#SBATCH --error=./logs/smoke_%j.err
#SBATCH --partition=nukwa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

set -euo pipefail

module load miniconda/3
source activate llm-env

export HF_HOME=/data/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/$USER/BIP-article-llm-benchmarks

mkdir -p logs out

nvidia-smi

python src/main.py --question_id 54
```

### Script del benchmark completo (benchmark_full.slurm)

Ejecuta el benchmark completo con todos los modelos configurados, con un límite de 24 horas.

```bash
nano slurm/benchmark_full.slurm
```

Contenido:

```bash
#!/bin/bash
#SBATCH --job-name=llm_benchmark_tanda1
#SBATCH --output=./logs/benchmark%j.out
#SBATCH --error=./logs/benchmark_%j.err
#SBATCH --partition=nukwa
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca

set -euo pipefail

module load miniconda/3
source activate llm-env

export HF_HOME=/data/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/$USER/BIP-article-llm-benchmarks

mkdir -p logs out

nvidia-smi

python src/main.py
```

---

## 9. Crear las carpetas auxiliares

```bash
mkdir -p /data/$USER/huggingface_cache logs out
```

---

## 10. Ejecutar el smoke test (prueba rápida)

Envía el trabajo de prueba al sistema de colas:

```bash
sbatch slurm/smoke_test.slurm
```

Monitorea el estado del trabajo:

```bash
squeue -u $USER
```

Cuando finalice (generalmente en 2 a 5 minutos), revise los logs generados:

```bash
ls -la logs/
cat logs/smoke_*.out
cat logs/smoke_*.err
```

- El archivo `*.out` debe mostrar la salida de `nvidia-smi` y la respuesta del modelo para la pregunta 54.
- El archivo `*.err` debe estar vacío o contener solo advertencias.

---

## 11. Ejecutar el benchmark completo

Si el smoke test funciona correctamente, lanza el trabajo definitivo:

```bash
sbatch slurm/benchmark_full.slurm
```

Puedes supervisar el progreso con:

```bash
watch -n 5 squeue -u $USER
```

Para ver la salida en tiempo real mientras el trabajo se ejecuta:

```bash
tail -f logs/benchmark*.out
```

Los resultados se guardarán en la carpeta `out/` con nombres como `benchmark_YYYYMMDD_HHMMSS.csv`.

---

## 12. Recuperar los resultados

Desde local, copie los archivos generados:

```bash
scp -P 22022 usuario@kabre.cenat.ac.cr:/work/usuario/BIP-article-llm-benchmarks/out/*.csv .
```

O si desea descargar todos los logs:

```bash
scp -P 22022 -r usuario@kabre.cenat.ac.cr:/work/usuario/BIP-article-llm-benchmarks/logs .
```

O si desea cancelar un job:
```bash 
scancel <Job ID>
```
---

## Notas importantes

- **Módulo de Miniconda:** Siempre cargue `module load miniconda/3` antes de usar `conda` o `source activate`. Puede agregar esta línea a su `~/.bashrc` para que se cargue automáticamente al iniciar sesión.
- **Uso de `source activate`:** Dentro de los scripts SLURM, utilice `source activate llm-env` en lugar de `conda activate`, ya que el entorno de ejecución de los nodos no siempre tiene inicializado el shell para `conda activate`.
- **Nodos excluidos:** El script `benchmark_full.slurm` excluye `nukwa-00` a `nukwa-03` debido a que esos nodos son muy pequeños. 
- **Cache de Hugging Face:** Se almacena en `/data/$USER/huggingface_cache`. Verifique el espacio disponible con `df -h /data`.
- **Al cambiar de nodo de login:** Recuerde volver a cargar el módulo y activar el entorno: `module load miniconda/3` y `source activate llm-env`.

---

## Solución de problemas comunes

| Problema | Causa probable | Solución |
|----------|---------------|----------|
| `conda: command not found` | El módulo de Miniconda no está cargado | Ejecutar `module load miniconda/3` |
| `source activate: No such file or directory` | El módulo no está cargado o el entorno no existe | Cargar el módulo y verificar la existencia del entorno |
| `conda: error: invalid choice 'activate'` | Se usó `conda activate` en un script SLURM sin inicialización previa | Reemplazar por `source activate llm-env` |
| `ModuleNotFoundError` | Falta alguna dependencia de Python | Instalar con `pip install -r requirements.txt` y verificar `llama-cpp-python` |
| `CUDA out of memory` | El modelo es demasiado grande para la GPU disponible | Reducir `gpu_layers` o usar una cuantización más agresiva |
| Trabajo en estado `PD` (pending) | No hay recursos disponibles en la cola | Esperar; si persiste, probar con menos CPUs o memoria |
| Error de `.bashrc` en logs | El script intenta hacer `source ~/.bashrc` y el archivo no existe | Eliminar esa línea del script SLURM |


---

Con esta guía podrá ejecutar los benchmarks de LLM en Kabré de manera reproducible. Si encuentra problemas adicionales, consulte la documentación oficial en `kabre.cenat.ac.cr`.