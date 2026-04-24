# face-match

**English (summary):** Small, dependency-light CLI to find the most **visually similar faces** in a folder of images (OpenCV YuNet + SFace). It does **not** prove legal identity, is not a certified biometric system, and must be used with a clear legal basis and governance. See **Disclaimer** below.

---

## Qué es

Herramienta de línea de comandos que, dada una **foto de consulta**, recorre una **carpeta de imágenes** (incluye subcarpetas), detecta rostros y ordena las fotos por **similitud** al rostro de la consulta. Pensada como base técnica para integraciones (no sustituye un producto completo de control de acceso o RR.HH.).

## Requisitos

- Python 3.9 o superior
- Conexión a Internet la **primera vez** (descarga ~40 MB de modelos ONNX de [OpenCV Zoo](https://github.com/opencv/opencv_zoo); luego quedan en caché local)

## Instalación

```bash
git clone https://github.com/Moca9801/face-match.git
cd face-match
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Instalación solo de ejecución (sin tests):

```bash
pip install .
```

## Uso

1. Coloca las imágenes de la galería en la carpeta `database` del directorio actual **o** indica otra con `--db`.
2. Ejecuta:

```bash
# Búsqueda estándar (usa umbral por defecto 0.363 para coseno)
face-match consulta.jpg

# Búsqueda en una carpeta específica con más resultados
face-match consulta.jpg --db C:\ruta\galeria -n 20

# Búsqueda estricta (solo coincidencias muy altas)
face-match consulta.jpg --threshold 0.5

# Reconstruir la base de datos (ignorar caché)
face-match consulta.jpg --rebuild
```

## Uso como Librería (SDK)

Puedes integrar el motor de búsqueda en tus propios scripts para recibir resultados estructurados (diccionarios y listas) sin ensuciar la consola:

```python
from pathlib import Path
from face_match import find_matches

# Ejecutar búsqueda programática
data = find_matches(
    query=Path("rostro.jpg"),
    db=Path("./galeria"),
    top=5,
    threshold=0.4
)

print(f"Escaneadas: {data['total_scanned']} fotos")
for res in data["results"]:
    print(f"Coincidencia: {res['path']} (Distancia: {res['distance']:.4f})")
```

Opciones:

| Opción (CLI) | Parámetro (SDK) | Atajo | Descripción |
|--------------|-----------------|-------|-------------|
| `query`      | `query`         | -     | Imagen con el rostro a buscar (obligatorio) |
| `--db`       | `db`            | -     | Carpeta de la galería (por defecto `./database`) |
| `--top`      | `top`           | `-n`  | Máximo de resultados (por defecto 10) |
| `--threshold`| `threshold`     | `-t`  | Umbral de similitud (def: 0.363 / 1.128) |
| `--metric`   | `distance`      | -     | `cosine` (0) o `l2` (1) |
| `--device`   | `device`        | -     | `cpu` o `gpu` |
| `--rebuild`  | `rebuild_cache` | -     | Ignora la caché y vuelve a extraer embeddings |

### ¿Cuándo usar CPU vs GPU?

El sistema es muy eficiente por defecto, pero puedes optimizarlo según tu hardware:

*   **Usar CPU (Por defecto):**
    *   Galerías pequeñas o medianas (< 10,000 fotos).
    *   Si no tienes una tarjeta NVIDIA configurada con CUDA.
    *   Para uso general en laptops, ya que consume menos energía.
*   **Usar GPU (`--device gpu`):**
    *   Galerías masivas (> 100,000 fotos).
    *   Cuando realizas un escaneo inicial pesado (`--rebuild`) y tienes una tarjeta NVIDIA.
    *   **Nota:** Requiere drivers de NVIDIA instalados y que las librerías (`opencv-python` con CUDA y `faiss-gpu`) sean compatibles con tu sistema. Si pides `gpu` y no está disponible, el sistema hará un *fallback* automático a `cpu`.

### Cómo interpretar los resultados

Este sistema soporta dos métricas de comparación:

- **Métrica Coseno (Recomendada):** Los valores suelen ir de 0 a 1. **Mayor es mejor**. Un valor > 0.363 suele indicar que es la misma persona.
- **Métrica L2 (Norma):** Distancia euclídea entre vectores. **Menor es mejor**. Un valor < 1.128 suele indicar coincidencia.

### Caché y Estabilidad

- **Caché Inteligente:** Se genera `.face_embeddings_cache.json` en la carpeta `--db`. Solo se recalculan las fotos modificadas, permitiendo búsquedas en milisegundos.
- **Descarga Segura:** Los modelos ONNX se descargan de [OpenCV Zoo](https://github.com/opencv/opencv_zoo) con validación de integridad y timeouts para mayor fiabilidad.


## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

### Seguridad y Privacidad de Datos

> [!CAUTION]
> **Datos Biométricos en Disco**: El archivo `.face_embeddings_cache.json` generado en la carpeta de la base de datos contiene representaciones matemáticas de rostros (embeddings) y **rutas relativas** de archivos de la galería. 
> - **No compartas** este archivo.
> - Trátalo como información confidencial según las leyes de protección de datos de tu región.
> - La caché es en formato JSON (seguro) para evitar riesgos de ejecución de código, pero el acceso al archivo debe estar restringido mediante permisos de sistema.

## Aviso legal, privacidad y uso responsable

- El **reconocimiento facial** y los **datos biométricos** están sometidos a leyes estrictas en muchos países (p. ej. RGPD en la UE, leyes locales en Latinoamérica). **No uses** este software sin base legal clara, política de privacidad, minimización de datos y, donde proceda, **consentimiento informado** de las personas afectadas.
- Esta herramienta devuelve **similitud visual** entre fotos, **no** certifica identidad ni debería usarse como única prueba en empleo, vigilancia o procesos judiciales sin controles humanos y procedimientos definidos por tu organización y asesoría jurídica.
- Los **falsos positivos** (personas distintas con rostro parecido) y **falsos negativos** son posibles; la iluminación, calidad, edad, accesorios y sesgos del modelo afectan el resultado.
- Los autores no se hacen responsables del uso que terceros hagan del software; se ofrece **«tal cual»** según la licencia MIT.

## Licencia

MIT — ver [LICENSE](LICENSE).

## Créditos

- Detección y reconocimiento basados en modelos publicados en [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet, SFace).
