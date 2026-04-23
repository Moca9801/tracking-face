# Historial de Cambios (Changelog)

Todas las novedades notables de este proyecto serán documentadas en este archivo.

## [0.2.0] - 2026-04-23
### Añadido
- **Motor FAISS**: Integración de búsqueda vectorial para alto rendimiento en grandes bases de datos.
- **Arquitectura SDK**: Nueva función `find_matches` que devuelve resultados como diccionarios/listas, desacoplando la lógica de la impresión en consola.
- **Soporte GPU**: Parámetro `--device gpu` para aceleración por hardware (OpenCV + FAISS).
- **Validación de Integridad**: Comprobación de Hash SHA256 para la descarga automática de modelos ONNX.
- **Tipado**: Inclusión de archivo `py.typed` para soporte de static typing (mypy).

### Corregido
- Bug en el ordenamiento de similitud de Coseno (ahora muestra los más parecidos arriba).
- Mejora en los contadores de la consola y mensajes de error cuando no hay coincidencias.
- Robustez en la descarga de modelos con timeouts y fallbacks.

## [0.1.0] - 2026-04-10
### Añadido
- Versión inicial con YuNet y SFace.
- Soporte básico de caché en archivo `.pkl`.
- Interfaz de línea de comandos (CLI).
