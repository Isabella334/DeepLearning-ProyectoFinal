# app
Pantalla y visualizacion de cerebros (P4).

## Pantalla

```bash
streamlit run app/pantalla.py
```

- Corre sobre la mesa simulada de MuJoCo (P1): los objetos se quedan y el carrito los mueve.
- Ciclo: **Mirar la mesa** → **Decidir qué hacer** → (Sí/No si el robot pregunta) → **Ejecutar** →
  **Aprobado** / **Desaprobado**.
- La memoria queda en `robot.db` y lo aprendido en `modelos/<persona>/` (ninguno se sube al repo).
  Con `:memory:` como base de datos no se guarda nada.
- El tema (colores y tipografía) está en `.streamlit/config.toml`; correr desde la raíz del repo.
