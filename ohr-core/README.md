# ohr-core

`ohr-core` es el paquete reusable del nucleo **Orthogonal Honeycomb Routing (OHR)** para clasificacion tabular multiclase.

Este paquete contiene solo el **nucleo del modelo**. No incluye datasets reales, campanas de benchmarking ni logica externa.

## Que Es OHR

En este repositorio, OHR se entiende como un **flujo estructurado de procesamiento tabular**, no como un clasificador aislado. Su papel es reorganizar explicitamente la representacion antes de la decision final.

La formulacion base del flujo es:

`T -> E -> f -> P -> R -> C`

donde:

- `T`: tabulacion o estructuracion de la entrada
- `E`: embedding
- `f`: adaptacion ligera
- `P`: proyeccion ortogonal
- `R`: routing jerarquico
- `C`: decision final o agregacion de expertos

En terminos mas precisos:

> OHR se define como un flujo estructurado de procesamiento tabular que organiza explicitamente las etapas de tabulacion, representacion, adaptacion, proyeccion ortogonal, enrutamiento jerarquico y decision final, con el proposito de analizar si una reorganizacion previa del espacio de caracteristicas puede aportar trazabilidad analitica y desempeno competitivo frente a enfoques tradicionales aplicados directamente sobre los datos.

Su implementacion actual sigue el pipeline:

`tabularizer -> preprocessing -> embedding -> adapter -> projection -> routing -> classifier`

## Alcance Del Paquete

`ohr-core` esta pensado para:

- definir y ejecutar la arquitectura OHR;
- ofrecer configuracion reproducible;
- entrenar y evaluar OHR sobre datos tabulares en memoria;
- exponer metricas internas de routing y cooperacion entre expertos.

No esta pensado para:

- cargar datasets reales como parte del wheel;
- contener comparaciones contra otros modelos;
- incluir tablas o reportes;
- actuar como repositorio de resultados experimentales.

## Instalacion

Desde la raiz del repositorio que contiene `ohr-core/`:

```powershell
python -m pip install -e .\ohr-core
```

Con dependencias de desarrollo:

```powershell
python -m pip install -e .\ohr-core[dev]
```

Verificacion rapida:

```powershell
python -c "from ohr import OHRClassifier; print('ohr ok')"
```

## Uso Minimo

```python
import numpy as np
import pandas as pd

from ohr import OHRClassifier

rng = np.random.default_rng(42)
X = pd.DataFrame(rng.normal(size=(60, 6)), columns=[f"f{i}" for i in range(6)])
y = np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

model = OHRClassifier()
model.config.embedding_dim = 16
model.config.training.epochs = 2
model.config.training.batch_size = 16

model.compile(device="cpu")
model.fit(X, y, validation_split=0.2)
metrics = model.evaluate(X, y)
predictions = model.predict_labels(X.iloc[:5])
```

## Configuracion

`ohr-core` puede inicializarse con:

- la configuracion por defecto empaquetada;
- un YAML o JSON externo;
- un objeto `OHRConfig`;
- un diccionario compatible.

Si no se pasa configuracion, `OHRClassifier()` usa automaticamente `default_ohr.yaml`.

### Bloques Principales

- `tabularizer`
- `preprocessing`
- `embedding`
- `adapter`
- `projection`
- `routing`
- `expert`
- `aggregator`
- `training`

### Configuraciones Base Del Repositorio

Dentro de `configs/` solo se mantienen ejemplos base:

- `default_ohr.yaml`
- `soft_linear_d2.yaml`

Las variantes experimentales o de  deben vivir fuera del paquete.

## Componentes Tecnicos Del Flujo

### 1. Tabularizer

Convierte `D_raw` en una tabla numerica estable.

Responsabilidades:

- aceptar `pandas.DataFrame` o `numpy.ndarray`
- validar estructura no vacia
- preservar orden estable de features
- forzar conversion numerica
- reemplazar infinitos por `NaN` cuando corresponde

### 2. Preprocessing

Realiza higiene numerica ligera antes del embedding:

- imputacion configurable
- escalado `none`, `standard` o `robust`

### 3. Embedding

Produce `h = E(x)` antes del adaptador.

Modos soportados:

- `fixed`
- `proportional`
- `pca_based`

PCA forma parte de `E`, no de `P`, para mantener separacion experimental entre:

- estrategia de embedding basada en datos
- proyeccion ortogonal propia de OHR

### 4. Adapter

Aprende una representacion intermedia comun antes del bloque central de OHR.

### 5. Projection

Aplica la proyeccion ortogonal explicita. Esta etapa organiza la representacion antes del routing.

### 6. Routing

Distribuye masa de decision entre nodos, hojas o expertos de manera jerarquica.

### 7. Classifier

Agrega expertos y produce logits multiclase.

## API Publica Principal

- `OHRClassifier(...)`
- `load_ohr_config(path=None)`
- `load_default_ohr_config()`
- `compile(...)`
- `fit(X, y, validation_data=None, validation_split=...)`
- `evaluate(X, y, include_internal_metrics=True)`
- `compute_hive_metrics(X)`
- `get_routing_diagnostics(X, top_k=...)`
- `inspect_samples(X, top_k=...)`
- `predict(X)`
- `predict_proba(X)`
- `predict_logits(X)`
- `predict_labels(X)`
- `summary()`
- `save(path)`
- `load(path)`

## Metadata Y Diagnostico

El clasificador expone metadata util para analisis reproducible, incluyendo:

- configuracion resuelta
- seed
- tipo de embedding y dimension efectiva
- tipo de experto y proyeccion
- pesos de regularizacion
- epochs configuradas y entrenadas
- metricas finales

Tambien puede devolver metricas internas como:

- `routing_entropy`
- `mean_leaf_probability`
- `load_balance_score`
- `effective_experts`
- `mean_effective_depth`
- `mean_top_expert_probability`
- `mean_projection_penalty`

## Testing

Desde `ohr-core/`:

```powershell
python -m unittest discover -s tests
```

## Relacion Con El Repositorio OHR

Si llegaste aqui desde el repositorio superior `OHR/`, ten en cuenta:

- `ohr-core/` es solo el nucleo;
- la evaluacion sobre CICIDS y las comparaciones con modelos de referencia viven fuera del paquete;
- esa separacion es intencional, para mantener el core limpio y reusable.
