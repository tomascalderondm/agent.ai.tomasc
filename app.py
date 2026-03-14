import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from google import genai
from google.cloud import bigquery
from google.oauth2 import service_account


# ============================================================
# CONFIGURACION BASE
# ============================================================

PROJECT_ID = "data-marketing-360"
BRAND_ID = "campo_noble"

CORE_DATASET = f"{BRAND_ID}_core"
AI_DATASET = f"{BRAND_ID}_ai"

MAX_ROWS_RESULT = 200
MAX_HISTORY_MESSAGES = 6
MAX_SQL_RETRIES = 2
MAX_BYTES_BILLED = 5 * 1024 * 1024 * 1024  # 5 GB


# ============================================================
# MAPA DE VERDAD OFICIAL
# ============================================================

MAPA_VERDAD: Dict[str, str] = {
    # CORE
    "fact_pedidos": f"{PROJECT_ID}.{CORE_DATASET}.fact_pedidos",
    "fact_pedido_productos": f"{PROJECT_ID}.{CORE_DATASET}.fact_pedido_productos",
    "dim_clientes": f"{PROJECT_ID}.{CORE_DATASET}.dim_clientes",
    "bridge_cliente_identidades": f"{PROJECT_ID}.{CORE_DATASET}.bridge_cliente_identidades",

    # AI
    "perfil_clientes_360": f"{PROJECT_ID}.{AI_DATASET}.perfil_clientes_360",
    "resumen_clientes": f"{PROJECT_ID}.{AI_DATASET}.resumen_clientes",
    "resumen_cliente_producto": f"{PROJECT_ID}.{AI_DATASET}.resumen_cliente_producto",
    "resumen_productos_retencion": f"{PROJECT_ID}.{AI_DATASET}.resumen_productos_retencion",
    "resumen_ventas_periodo": f"{PROJECT_ID}.{AI_DATASET}.resumen_ventas_periodo",
    "resumen_productos_ventas": f"{PROJECT_ID}.{AI_DATASET}.resumen_productos_ventas",
    "resumen_geografia": f"{PROJECT_ID}.{AI_DATASET}.resumen_geografia",
    "afinidad_productos": f"{PROJECT_ID}.{AI_DATASET}.afinidad_productos",
    "auditoria_datos": f"{PROJECT_ID}.{AI_DATASET}.auditoria_datos",
    "action_center": f"{PROJECT_ID}.{AI_DATASET}.action_center",
}

ALLOWED_TABLES = set(MAPA_VERDAD.values())


# ============================================================
# CREDENCIALES Y CLIENTES
# ============================================================

creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)

bq_client = bigquery.Client(credentials=creds, project=PROJECT_ID)
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_sql(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.strip()
    texto = texto.replace("```sql", "").replace("```", "").strip()
    return texto


def obtener_texto_modelo(response) -> str:
    if response is None:
        return ""
    return getattr(response, "text", "") or ""


def construir_contexto_historial(messages: List[Dict[str, str]], max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    recientes = messages[-max_messages:]
    return "\n".join([f"{m['role']}: {m['content']}" for m in recientes])


def obtener_tablas_prioritarias(pregunta_usuario: str) -> List[str]:
    q = pregunta_usuario.lower()

    prioridades = []

    if any(x in q for x in ["venta", "ventas", "mes", "año", "anio", "comparar", "facturación", "facturacion", "ticket"]):
        prioridades += [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if any(x in q for x in ["cliente", "clientes", "ltv", "segmento", "recurrente", "nuevo", "riesgo", "inactividad"]):
        prioridades += [
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_clientes"],
            MAPA_VERDAD["dim_clientes"],
        ]

    if any(x in q for x in ["producto", "productos", "retención", "retencion", "recompra", "gancho", "afinidad", "bundle"]):
        prioridades += [
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["resumen_cliente_producto"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["afinidad_productos"],
            MAPA_VERDAD["fact_pedido_productos"],
        ]

    if any(x in q for x in ["comuna", "geo", "geografía", "geografia", "ciudad", "zona", "territorio"]):
        prioridades += [
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if any(x in q for x in ["calidad", "auditoría", "auditoria", "error", "cobertura"]):
        prioridades += [
            MAPA_VERDAD["auditoria_datos"],
        ]

    if any(x in q for x in ["acción", "accion", "oportunidad", "winback", "fidelizar", "meta ads"]):
        prioridades += [
            MAPA_VERDAD["action_center"],
        ]

    if not prioridades:
        prioridades = [
            MAPA_VERDAD["resumen_clientes"],
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["perfil_clientes_360"],
        ]

    return list(dict.fromkeys(prioridades))


def obtener_esquemas_tablas(tablas_objetivo: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    esquemas: Dict[str, List[Tuple[str, str]]] = {}

    for table_fq in tablas_objetivo:
        project_id, dataset_id, table_name = table_fq.split(".")

        query = f"""
        SELECT
          column_name,
          data_type
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = @table_name
        ORDER BY ordinal_position
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("table_name", "STRING", table_name)
            ],
            use_query_cache=True,
            maximum_bytes_billed=MAX_BYTES_BILLED,
        )

        try:
            df = bq_client.query(query, job_config=job_config).to_dataframe()
            if not df.empty:
                esquemas[table_fq] = list(zip(df["column_name"].tolist(), df["data_type"].tolist()))
        except Exception:
            continue

    return esquemas


def formatear_esquemas_para_prompt(esquemas: Dict[str, List[Tuple[str, str]]]) -> str:
    bloques = []
    for tabla, cols in esquemas.items():
        columnas = ", ".join([f"{col} ({dtype})" for col, dtype in cols])
        bloques.append(f"- {tabla}: {columnas}")
    return "\n".join(bloques)


def validar_sql(query: str) -> Tuple[bool, str]:
    query_limpio = limpiar_sql(query)
    if not query_limpio:
        return False, "La consulta vino vacía."

    upper = query_limpio.upper()

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "La consulta debe ser solo SELECT o WITH."

    bloqueadas = ["INSERT ", "UPDATE ", "DELETE ", "MERGE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE "]
    for palabra in bloqueadas:
        if palabra in upper:
            return False, f"Se detectó una operación no permitida: {palabra.strip()}."

    tablas_en_query = set(re.findall(r"`([^`]+)`", query_limpio))
    tablas_fuera_mapa = [t for t in tablas_en_query if t not in ALLOWED_TABLES]

    if tablas_fuera_mapa:
        return False, f"La consulta intenta usar tablas fuera del Mapa de Verdad: {tablas_fuera_mapa}"

    if not tablas_en_query:
        return False, "La consulta debe referenciar tablas oficiales usando backticks."

    return True, ""


def ejecutar_query_segura(query: str) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    return bq_client.query(query, job_config=job_config).result().to_dataframe()


# ============================================================
# PROMPTS NOBLEBOTAI
# ============================================================

def construir_prompt_sql(
    pregunta_usuario: str,
    historial_contexto: str,
    tablas_prioritarias: List[str],
    esquemas_texto: str,
    error_previo: str = "",
) -> str:
    tablas_texto = "\n".join([f"- {t}" for t in tablas_prioritarias])

    return f"""
Eres NobleBotAI 🐷🐽, Director de Inteligencia de Campo Noble.

Tu trabajo es escribir SQL de BigQuery perfecto, rápido y sin alucinaciones.

# ROL Y PERSONALIDAD
Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽. Mi tono es ejecutivo, rápido y 100% preciso.

# MAPA DE VERDAD OFICIAL
Puedes usar ÚNICAMENTE estas tablas:

{chr(10).join([f"- {v}" for v in MAPA_VERDAD.values()])}

# TABLAS PRIORITARIAS PARA ESTA PREGUNTA
{tablas_texto}

# ESQUEMAS REALES DISPONIBLES
{esquemas_texto}

# PROTOCOLO DE CERO ALUCINACION
1. Prohibido inventar tablas o columnas.
2. Usa primero tablas resumen AI cuando sean suficientes.
3. Usa tablas CORE solo cuando la pregunta requiera mayor detalle histórico.
4. Si el usuario pide ventas por mes o año, prioriza `resumen_ventas_periodo`.
5. Si el usuario pide productos que más aportan, prioriza `resumen_productos_ventas`.
6. Si el usuario pide retención o recompra de productos, prioriza `resumen_productos_retencion`.
7. Si el usuario pide clientes, segmentos, LTV o riesgo, prioriza `perfil_clientes_360` y `resumen_clientes`.
8. Si el usuario pide detalle histórico de compras de un cliente, puedes usar `fact_pedidos` y `fact_pedido_productos`.
9. Si el usuario pide afinidad entre productos, usa `afinidad_productos`.
10. Si el usuario pide geografía/comunas, usa `resumen_geografia`.
11. Si la pregunta se resuelve con una tabla agregada, no uses una tabla más pesada.
12. Siempre limita a un resultado razonable si la pregunta pide detalle abierto.
13. Si el usuario nombra una persona y no entrega email exacto, usa búsqueda flexible con LIKE sobre nombre.
14. Si del historial ya existe un email, nombre o producto relevante, úsalo.

# MEMORIA OPERATIVA
{historial_contexto}

# PREGUNTA DEL USUARIO
{pregunta_usuario}

# ERROR PREVIO A CORREGIR
{error_previo}

# REGLAS TECNICAS
- Responde ÚNICAMENTE con SQL.
- La consulta debe empezar con SELECT o WITH.
- Usa backticks en TODAS las tablas.
- No escribas explicación.
- No uses CREATE, INSERT, UPDATE, DELETE, DROP ni ALTER.
- Si la pregunta pide detalle, devuelve columnas útiles y claras.
- Si la pregunta es ejecutiva, devuelve agregados útiles y ordenados.
- Usa alias legibles.
- Si no hace falta traer todo, usa LIMIT.
"""


def construir_prompt_respuesta(
    pregunta_usuario: str,
    historial_contexto: str,
    sql_usado: str,
    datos_texto: str,
    num_filas: int,
) -> str:
    return f"""
Eres NobleBotAI, el asistente IA chancho 🐷🐽 y Director de Inteligencia de Campo Noble.

# PERSONALIDAD OBLIGATORIA
- Siempre responde en español.
- Tono ejecutivo, rápido, preciso y estratégico.
- Nunca digas "I'm sorry, but I'm unable to assist...".
- Si hay poca data, no lo presentes como fallo: conviértelo en oportunidad estratégica.
- Nunca hables como bot genérico.
- No uses "jefe".

# PROTOCOLO HUMANO
- Si solo hay uno o pocos resultados, dilo como:
  - "Aquí detectamos una punta de lanza clara"
  - "La base todavía es pequeña, pero hay una oportunidad quirúrgica"
  - "No es volumen, es señal"
- Si ves cliente valioso inactivo, llámalo "Fuga de Valor".
- Si ves patrón claro, entrega recomendación accionable inmediata.

# ESTRUCTURA OBLIGATORIA DE RESPUESTA
1. Empieza con: "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽."
2. Luego: "Consulté en las tablas oficiales y aquí tienes el resultado..."
3. Explica el hallazgo principal en lenguaje ejecutivo.
4. Si hay números monetarios, habla en CLP.
5. Cierra con un insight o recomendación estratégica inmediata.

# CONTEXTO DE CONVERSACION
{historial_contexto}

# PREGUNTA
{pregunta_usuario}

# SQL USADO
{sql_usado}

# FILAS DEVUELTAS
{num_filas}

# DATOS
{datos_texto}
"""


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def generar_sql_con_reintentos(pregunta_usuario: str, historial_contexto: str) -> Tuple[str, pd.DataFrame]:
    tablas_prioritarias = obtener_tablas_prioritarias(pregunta_usuario)
    esquemas = obtener_esquemas_tablas(tablas_prioritarias)

    if not esquemas:
        raise ValueError("No pude obtener esquemas reales de las tablas oficiales.")

    esquemas_texto = formatear_esquemas_para_prompt(esquemas)

    error_previo = ""
    ultimo_error = ""

    for _ in range(MAX_SQL_RETRIES):
        prompt_sql = construir_prompt_sql(
            pregunta_usuario=pregunta_usuario,
            historial_contexto=historial_contexto,
            tablas_prioritarias=tablas_prioritarias,
            esquemas_texto=esquemas_texto,
            error_previo=error_previo,
        )

        respuesta_sql = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_sql,
        )

        query_limpio = limpiar_sql(obtener_texto_modelo(respuesta_sql))
        es_valida, motivo = validar_sql(query_limpio)

        if not es_valida:
            ultimo_error = motivo
            error_previo = f"La consulta previa fue inválida. Motivo: {motivo}"
            continue

        try:
            df = ejecutar_query_segura(query_limpio)
            return query_limpio, df
        except Exception as e:
            ultimo_error = str(e)
            error_previo = f"La consulta previa falló en BigQuery. Error: {str(e)}"

    raise ValueError(f"No pude construir una consulta válida tras reintentos. Último error: {ultimo_error}")


def responder_como_noblebot(
    pregunta_usuario: str,
    historial_contexto: str,
    sql_usado: str,
    df_datos: pd.DataFrame,
) -> str:
    if df_datos.empty:
        return (
            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
            "Consulté en las tablas oficiales y aquí tienes el resultado: por ahora no encontré filas que calcen exactamente con esa búsqueda.\n\n"
            "Eso no significa que no haya valor. Más bien nos dice que esta ruta está estrecha o que conviene reformular el ángulo de análisis para encontrar la señal correcta.\n\n"
            "Insight inmediato: probaría una búsqueda más quirúrgica por período, producto, comuna o una variación del nombre del cliente."
        )

    datos_texto = df_datos.head(60).to_string(index=False)

    prompt_final = construir_prompt_respuesta(
        pregunta_usuario=pregunta_usuario,
        historial_contexto=historial_contexto,
        sql_usado=sql_usado,
        datos_texto=datos_texto,
        num_filas=len(df_datos),
    )

    respuesta_final = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_final,
    )
    texto = obtener_texto_modelo(respuesta_final).strip()

    if not texto:
        texto = (
            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
            "Consulté en las tablas oficiales y aquí tienes el resultado.\n\n"
            "Ya tengo la data correcta, pero la redacción final salió vacía. "
            "Aun así, el SQL y los datos sí se recuperaron correctamente."
        )

    return texto


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

st.set_page_config(page_title="NobleBotAI 🐷🐽", layout="wide")
st.title("NobleBotAI 🐷🐽")
st.caption("Director de Inteligencia de Campo Noble")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

if "last_df" not in st.session_state:
    st.session_state.last_df = None

with st.sidebar:
    st.header("Configuración")
    st.markdown(f"**Proyecto:** `{PROJECT_ID}`")
    st.markdown(f"**Marca base:** `{BRAND_ID}`")

    st.subheader("Mapa de Verdad")
    for alias, fq_table in MAPA_VERDAD.items():
        st.markdown(f"- **{alias}** → `{fq_table}`")

    if st.button("Limpiar memoria"):
        st.session_state.messages = []
        st.session_state.last_sql = ""
        st.session_state.last_df = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregúntame por clientes, ventas, productos, comunas o retención..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("NobleBotAI está consultando las tablas oficiales..."):
            try:
                memoria_reciente = construir_contexto_historial(st.session_state.messages)
                sql_usado, df_datos = generar_sql_con_reintentos(prompt, memoria_reciente)
                respuesta = responder_como_noblebot(prompt, memoria_reciente, sql_usado, df_datos)

                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                st.session_state.last_sql = sql_usado
                st.session_state.last_df = df_datos

                with st.expander("Ver SQL usado"):
                    st.code(sql_usado, language="sql")

                with st.expander("Ver datos tabulares"):
                    st.dataframe(df_datos, use_container_width=True)

            except Exception as e:
                respuesta_error = (
                    "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                    "Consulté las rutas oficiales, pero hubo un problema técnico al construir o ejecutar la consulta.\n\n"
                    "No me rindo: esto normalmente se corrige afinando el filtro, el período o el ángulo de búsqueda. "
                    "Prueba con una pregunta más concreta, por ejemplo por cliente, producto, mes, comuna o segmento."
                )
                st.markdown(respuesta_error)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_error})

                with st.expander("Detalle técnico"):
                    st.code(str(e))
