# OHR: Orthogonal Honeycomb Routing

Repositorio separado para publicar el **nucleo OHR** y una **capa minima de evaluacion** alineada con el trabajo:

**"Comparacion experimental de un flujo estructurado de clasificacion tabular basado en la propuesta OHR frente a modelos tradicionales"**

## Que Es OHR

OHR no se plantea aqui como "otro clasificador" sin mas. En este trabajo se presenta como un **flujo estructurado de procesamiento tabular basado en OHR**, orientado a reorganizar la informacion antes de la decision final.

`T -> E -> f -> P -> R -> C`

donde:

- `T` es tabulacion o estructuracion de la entrada
- `E` es embedding
- `f` es adaptacion ligera
- `P` es proyeccion ortogonal
- `R` es routing jerarquico
- `C` es decision final o agregacion de expertos

La idea central no es definir OHR como un modelo aislado, sino como un pipeline que transforma, proyecta y enruta la representacion antes de clasificar.

Una formulacion precisa, alineada con el trabajo, es la siguiente:

> OHR se define como un flujo estructurado de procesamiento tabular que organiza explicitamente las etapas de tabulacion, representacion, adaptacion, proyeccion ortogonal, enrutamiento jerarquico y decision final, con el proposito de analizar si una reorganizacion previa del espacio de caracteristicas puede aportar trazabilidad analitica y desempeno competitivo frente a enfoques tradicionales aplicados directamente sobre los datos.

La motivacion experimental del repositorio es comparar ese flujo estructurado contra enfoques monoliticos aplicados directamente sobre datos tabulares.

## Estructura Del Repositorio

```text
OHR/
  README.md
  ohr-core/
  experiments/¿
```

### `ohr-core/`

Paquete reusable del nucleo OHR.

Incluye:

- codigo fuente del modelo
- configuracion base
- documentacion del paquete
- tests

No incluye:

- benchmarking general
- campanas de tesis complejas
- resultados generados
- tablas de comparacion externas

### `experiments/`

Capa minima de experimentacion orientada a evaluacion academica.

Actualmente contiene un notebook principal:

- `OHR_CICIDS_analysis.ipynb`

Ese notebook esta pensado para:

- cargar CICIDS localmente
- entrenar OHR y modelos de referencia
- comparar resultados
- revisar matrices de confusion
- analizar predicciones y metricas internas de OHR

## Alcance De Este Repositorio

Este repositorio sirve para compartir:

1. el **nucleo OHR** como paquete reproducible;
2. una **evaluacion experimental compacta** sobre CICIDS;
3. el contexto suficiente para reproducir una comparacion academica razonable.

No pretende contener todo el workspace original de desarrollo ni toda la infraestructura auxiliar usada durante el trabajo.

## Instalacion

Desde la raiz de esta carpeta:

```powershell
python -m pip install -e .\ohr-core
```

Si tambien quieres dependencias de desarrollo:

```powershell
python -m pip install -e .\ohr-core[dev]
```

Verificacion rapida:

```powershell
python -c "from ohr import OHRClassifier; print('ohr ok')"
```

## Como Ejecutar La Evaluacion

1. Coloca los CSV de CICIDS en:

```text
external_data/CICIDS2017/
```

2. Abre:

```text
experiments/OHR_CICIDS_analysis.ipynb
```

3. Ajusta la ruta del dataset si hace falta.
4. Ejecuta las celdas en orden.

El notebook esta pensado para comparar:

- `OHR base`
- `OHR sharp routing`
- `RandomForest`
- `LogisticRegression`
- `LinearSVC`

## Que Debe Revisar Un Evaluador Externo

Si alguien solo quiere evaluar el repositorio, lo importante es:

- instalar `ohr-core`
- apuntar el notebook a la carpeta local del dataset
- ejecutar la comparacion
- revisar:
  - metricas globales
  - matrices de confusion
  - comparacion de predicciones
  - metricas internas de OHR

## Relacion Con el trabajo

Este repositorio esta organizado para apoyar una discusion academica sobre tres ejes:

1. **rendimiento predictivo**
2. **eficiencia computacional**
3. **comportamiento interno del flujo estructurado**

Por eso la capa experimental no se limita a entrenar modelos, sino que tambien conserva trazabilidad y diagnosticos del routing cuando se usa OHR.

## Licencia

Este repositorio se distribuye bajo la **GNU General Public License v3.0**.

Si se publica en GitHub, conviene revisar tambien:

- si el dataset real debe quedar excluido del repositorio;
- si los resultados generados deben publicarse o regenerarse.
