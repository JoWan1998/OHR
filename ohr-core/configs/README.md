# Base OHR Configs

Esta carpeta contiene unicamente **configuraciones base del paquete**.

Su funcion es ofrecer ejemplos minimos y reproducibles para usar `ohr-core` sin mezclar campanas experimentales ni configuraciones especificas de tesis dentro del nucleo.

## Archivos Incluidos

- `default_ohr.yaml`
  Configuracion base del paquete, alineada con el default empaquetado en `src/ohr/resources/configs/`.
- `soft_linear_d2.yaml`
  Ejemplo minimo con `soft routing`, experto lineal y profundidad 2.

## Que No Se Mantiene Aqui

No se incluyen:

- ablaciones de tesis;
- campanas comparativas;
- configs de benchmarking;
- variantes ligadas a un dataset concreto como CICIDS.

Ese material debe vivir fuera del paquete, en la capa externa de evaluacion del repositorio.

## Uso Esperado

```python
from ohr import OHRClassifier

model = OHRClassifier("ohr-core/configs/soft_linear_d2.yaml")
```

Si no se pasa una ruta, `OHRClassifier()` usa automaticamente la configuracion por defecto empaquetada.
