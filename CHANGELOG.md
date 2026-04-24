# Historial de Cambios (Changelog)

Todas las novedades notables de este proyecto serán documentadas en este archivo.

## [0.2.0] - 2026-04-23
### Añadido
- **Motor FAISS**: Integración de búsqueda vectorial para alto rendimiento en grandes bases de datos.
- **Arquitectura SDK**: Nueva función `find_matches` que devuelve resultados como diccionarios/listas, desacoplando la lógica de la impresión en consola. Soporta caché en formato JSON para persistencia de datos.
- **Seguridad (Hardening)**: Migración de `pickle` a **JSON** para evitar riesgos de RCE y uso de **rutas relativas** en la caché para proteger la privacidad del usuario.
- **Infraestructura de Calidad**: Pipeline de CI con Ruff (linting), Mypy (strict typing) y tests multiplataforma (Linux/Windows).
- **Soporte GPU**: Parámetro `--device gpu` para aceleración por hardware (OpenCV + FAISS).
- **Validación de Integridad**: Comprobación de Hash SHA256 para la descarga automática de modelos ONNX.
- **Tipado**: Inclusión de archivo `py.typed` y definición de `SearchResponse` (TypedDict) para el SDK.

### Corregido
- Bug en el ordenamiento de similitud de Coseno (ahora muestra los más parecidos arriba).
- Mejora en los contadores de la consola y mensajes de error cuando no hay coincidencias.
- Robustez en la descarga de modelos con timeouts y fallbacks.
- Validación de archivos: Se utiliza `query.name` en los mensajes de `ValueError` para una mejor trazabilidad.

## [0.1.0] - 2026-03-20
### Añadido
- Primera versión funcional (CLI).
- Soporte básico para YuNet y SFace.
- Almacenamiento de embeddings en `.pkl` (sustituido por JSON en v0.2.0 por seguridad).
- Interfaz de línea de comandos (CLI).
