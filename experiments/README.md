# Experiments

Esta carpeta contiene la **capa externa de evaluacion** usada para revisar OHR sobre datos tabulares reales, especialmente en el contexto de la tesis.

## Objetivo

La idea de esta carpeta es mantener fuera de `ohr-core` toda la logica de comparacion experimental.

Aqui vive lo que sirve para:

- cargar el dataset real;
- preparar un experimento reproducible;
- entrenar OHR y modelos de referencia;
- comparar resultados;
- revisar predicciones y metricas internas.

En el contexto de la tesis, esta capa experimental evalua a OHR como **flujo estructurado de procesamiento tabular** y no simplemente como un clasificador adicional dentro de una lista de modelos.

## Notebook Principal

- `OHR_CICIDS_analysis.ipynb`

Ese notebook esta disenado para que un tercero pueda:

1. indicar la carpeta local donde estan los CSV de CICIDS;
2. ejecutar la preparacion del dataset;
3. entrenar los modelos comparados;
4. revisar tablas, graficas, matrices de confusion y comparaciones de prediccion.

## Modelos Comparados

El experimento considera:

- `OHR base`
- `OHR sharp routing`
- `RandomForest`
- `LogisticRegression`
- `LinearSVC`

La variante `sharp routing` se conserva como la configuracion OHR recomendada para la comparacion principal, mientras que `OHR base` sirve como referencia interna.

## Relacion Con `ohr-core`

`ohr-core` sigue siendo unicamente el nucleo del modelo.

Esta carpeta:

- consume `ohr-core` como dependencia;
- no reimplementa OHR;
- no forma parte del wheel del paquete.

## Dataset

El notebook espera un dataset local, normalmente en:

```text
../external_data/CICIDS2017/
```

Si el dataset esta en otra ruta, se ajusta directamente en la celda de parametros del notebook.

## Uso Esperado

Esta carpeta esta pensada para:

- evaluacion academica;
- revision externa del experimento;
- generacion de tablas y evidencia para la tesis.

No esta pensada como libreria general ni como API estable.
