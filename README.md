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
git clone https://github.com/Moca9801/tracking-face.git
cd tracking-face
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

Opciones:

| Opción | Atajo | Descripción |
|--------|-------|-------------|
| `query` | - | Imagen con el rostro a buscar (obligatorio) |
| `--db` | - | Carpeta de la galería (por defecto `./database`) |
| `--top` | `-n` | Máximo de resultados a mostrar (por defecto 10) |
| `--threshold`| `-t` | Umbral de similitud. (Coseno defecto: `0.363`, L2 defecto: `1.128`) |
| `--metric` | - | `cosine` (por defecto) o `l2` |
| `--rebuild` | - | Ignora la caché y vuelve a extraer embeddings |

### Cómo interpretar los resultados

Este sistema soporta dos métricas de comparación:

- **Métrica Coseno (Recomendada):** Los valores suelen ir de 0 a 1. **Mayor es mejor**. Un valor > 0.363 suele indicar que es la misma persona.
- **Métrica L2 (Norma):** Distancia euclídea entre vectores. **Menor es mejor**. Un valor < 1.128 suele indicar coincidencia.

### Caché y Estabilidad

- **Caché Inteligente:** Se genera `.face_embeddings_cache.pkl` en la carpeta `--db`. Solo se recalculan las fotos modificadas, permitiendo búsquedas en milisegundos.
- **Descarga Segura:** Los modelos ONNX se descargan de [OpenCV Zoo](https://github.com/opencv/opencv_zoo) con validación de integridad y timeouts para mayor fiabilidad.


## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

## Aviso legal, privacidad y uso responsable

- El **reconocimiento facial** y los **datos biométricos** están sometidos a leyes estrictas en muchos países (p. ej. RGPD en la UE, leyes locales en Latinoamérica). **No uses** este software sin base legal clara, política de privacidad, minimización de datos y, donde proceda, **consentimiento informado** de las personas afectadas.
- Esta herramienta devuelve **similitud visual** entre fotos, **no** certifica identidad ni debería usarse como única prueba en empleo, vigilancia o procesos judiciales sin controles humanos y procedimientos definidos por tu organización y asesoría jurídica.
- Los **falsos positivos** (personas distintas con rostro parecido) y **falsos negativos** son posibles; la iluminación, calidad, edad, accesorios y sesgos del modelo afectan el resultado.
- Los autores no se hacen responsables del uso que terceros hagan del software; se ofrece **«tal cual»** según la licencia MIT.

## Licencia

MIT — ver [LICENSE](LICENSE).

## Créditos

- Detección y reconocimiento basados en modelos publicados en [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (YuNet, SFace).
