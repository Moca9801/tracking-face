# Guía de Contribución

¡Gracias por tu interés en mejorar `face-match`! 

## Cómo contribuir

1. **Reportar Bugs**: Abre un *Issue* describiendo el problema, tu sistema operativo y cómo reproducirlo.
2. **Sugerir Mejoras**: Si tienes una idea para una nueva funcionalidad, abre un *Issue* para discutirla.
3. **Enviar Código**:
    - Haz un *Fork* del repositorio.
    - Crea una rama para tu mejora (`git checkout -b feature/mejora`).
    - Asegúrate de que los tests pasen (`pytest`).
    - Envía un *Pull Request*.

## Desarrollo Local

Si deseas contribuir al código, por favor asegúrate de seguir estos pasos antes de abrir un Pull Request:

1. **Instalar dependencias de desarrollo**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Verificar el estilo (Linting)**:
   ```bash
   ruff check src tests
   ```

3. **Verificar el tipado**:
   ```bash
   mypy src/face_match
   ```

4. **Ejecutar los tests**:
   ```bash
   pytest
   ```

Mantenemos una barra de calidad alta; el CI de GitHub no permitirá fusiones si estos comandos fallan.

## Proceso de Pull Request

1. **Crear una rama**: Usa un nombre descriptivo (ej: `fix-bug-identificacion` o `feat-nueva-metrica`).
2. **Realizar cambios**: Sigue los estándares de código y tipado.
3. **Validar localmente**: Asegúrate de que `ruff`, `mypy` y `pytest` pasen en verde.
4. **Abrir PR**: Describe claramente qué problema resuelve o qué funcionalidad añade tu cambio.

## Estándares de Código

- Usamos **anotaciones de tipos** (type hints) en todas las funciones nuevas.
- El código debe seguir el estilo **PEP 8**.
- Si añades una funcionalidad, añade un test en la carpeta `tests/`.

## Reporte de Seguridad

Si encuentras una vulnerabilidad de seguridad, por favor **no abras un issue público**. Utiliza la función de **[GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories/repository-security-advisories/about-repository-security-advisories)** para reportarla de forma privada. Esto nos permite coordinar una solución antes de que la vulnerabilidad se haga pública.
