import io
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from google import genai
from google.cloud import bigquery
from google.oauth2 import service_account
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# CONFIGURACION BASE
# ============================================================

PROJECT_ID = st.secrets.get("PROJECT_ID", "data-marketing-360")
LOCATION = st.secrets.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BRAND_ID = st.secrets.get("BRAND_ID", "campo_noble")

CORE_DATASET = f"{BRAND_ID}_core"
AI_DATASET = f"{BRAND_ID}_ai"

MAX_ROWS_RESULT = 300
MAX_HISTORY_MESSAGES = 6
MAX_SQL_RETRIES = 2
MAX_BYTES_BILLED = 5 * 1024 * 1024 * 1024  # 5 GB

# Configuración definida por ti
MODEL_SQL = st.secrets.get("MODEL_SQL", "gemini-3-pro")
MODEL_RESPONSE = st.secrets.get("MODEL_RESPONSE", "gemini-3.1-flash-lite")
MODEL_MEDIA = st.secrets.get("MODEL_MEDIA", "gemini-3.1-flash")

# Fallbacks opcionales por si algún alias falla
MODEL_FALLBACKS_SQL = [
    "gemini-3-pro",
    "gemini-3-flash-lite",
    "gemini-2.5-flash",
]

MODEL_FALLBACKS_RESPONSE = [
    "gemini-3-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
]

MODEL_FALLBACKS_MEDIA = [
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

ENABLE_EXTERNAL_CORROBORATION = st.secrets.get("ENABLE_EXTERNAL_CORROBORATION", True)


# ============================================================
# MAPA DE VERDAD OFICIAL
# ============================================================

MAPA_VERDAD: Dict[str, str] = {
    # CORE
    "fact_pedidos": f"{PROJECT_ID}.{CORE_DATASET}.fact_pedidos",
    "fact_pedido_productos": f"{PROJECT_ID}.{CORE_DATASET}.fact_pedido_productos",
    "fact_pedidos_limpios": f"{PROJECT_ID}.{CORE_DATASET}.fact_pedidos_limpios",
    "dim_clientes": f"{PROJECT_ID}.{CORE_DATASET}.dim_clientes",
    "bridge_cliente_identidades": f"{PROJECT_ID}.{CORE_DATASET}.bridge_cliente_identidades",

    # AI general
    "perfil_clientes_360": f"{PROJECT_ID}.{AI_DATASET}.perfil_clientes_360",
    "resumen_ventas_periodo": f"{PROJECT_ID}.{AI_DATASET}.resumen_ventas_periodo",
    "resumen_productos_ventas": f"{PROJECT_ID}.{AI_DATASET}.resumen_productos_ventas",
    "resumen_productos_retencion": f"{PROJECT_ID}.{AI_DATASET}.resumen_productos_retencion",
    "afinidad_productos": f"{PROJECT_ID}.{AI_DATASET}.afinidad_productos",
    "auditoria_datos": f"{PROJECT_ID}.{AI_DATASET}.auditoria_datos",

    # AI geografia canónica
    "geo_catalogo_chile_normalizado": f"{PROJECT_ID}.{AI_DATASET}.geo_catalogo_chile_normalizado",
    "geo_aliases": f"{PROJECT_ID}.{AI_DATASET}.geo_aliases",
    "geo_fuentes_raw": f"{PROJECT_ID}.{AI_DATASET}.geo_fuentes_raw",
    "geo_fuentes_limpias": f"{PROJECT_ID}.{AI_DATASET}.geo_fuentes_limpias",
    "geo_resolucion_comunas": f"{PROJECT_ID}.{AI_DATASET}.geo_resolucion_comunas",
    "geo_no_resueltos": f"{PROJECT_ID}.{AI_DATASET}.geo_no_resueltos",
    "geo_inteligencia_base": f"{PROJECT_ID}.{AI_DATASET}.geo_inteligencia_base",
    "geo_filtros_expandibles": f"{PROJECT_ID}.{AI_DATASET}.geo_filtros_expandibles",
    "resumen_geografia": f"{PROJECT_ID}.{AI_DATASET}.resumen_geografia",
}

ALLOWED_TABLES = set(MAPA_VERDAD.values())


# ============================================================
# CREDENCIALES Y CLIENTES
# ============================================================

creds_dict = dict(st.secrets["gcp_service_account"])

creds = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bq_client = bigquery.Client(
    credentials=creds,
    project=PROJECT_ID,
)

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=creds,
)


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def limpiar_sql(texto: str) -> str:
    if not texto:
        return ""
    return texto.strip().replace("```sql", "").replace("```", "").strip()


def obtener_texto_modelo(response) -> str:
    if response is None:
        return ""
    return getattr(response, "text", "") or ""


def resumir_dataframe_para_prompt(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "Sin filas."
    return df.head(max_rows).to_string(index=False)


def generar_contenido_con_fallback(prompt: str, preferred_model: str):
    candidatos = [preferred_model] + [m for m in MODEL_FALLBACKS if m != preferred_model]
    ultimo_error = None

    for model_name in candidatos:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response is not None:
                return response
        except Exception as e:
            ultimo_error = e
            continue

    raise ValueError(f"No pude generar contenido con ningún modelo Gemini configurado. Último error: {ultimo_error}")


def es_pregunta_medios_o_inversion(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower()
    keywords = [
        "inversión", "inversion", "medios", "paid media", "google ads", "meta ads",
        "presupuesto", "budget", "roas", "cpa", "cac", "campaña", "campana",
        "performance", "escalar", "escalamiento", "retargeting", "prospecting",
        "audiencia", "audiencias", "funnel", "embudo", "bidding", "puja"
    ]
    return any(x in q for x in keywords)


def es_pregunta_informe(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower()
    patrones = [
        "hazme un informe",
        "quiero un informe",
        "reporte",
        "resumen ejecutivo",
        "informe ejecutivo",
        "analisis mensual",
        "análisis mensual",
        "informe de enero",
        "informe de febrero",
    ]
    return any(p in q for p in patrones)


def es_solicitud_grafico(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower().strip()
    patrones = [
        "grafica", "gráfica", "grafícame", "graficame", "grafico", "gráfico",
        "hazme un grafico", "hazme un gráfico", "chart", "plot", "barra",
        "barras", "linea", "línea", "lineas", "líneas", "visualiza", "visualizame"
    ]
    return any(p in q for p in patrones)


def es_solicitud_pdf(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower().strip()
    patrones = [
        "pdf", "descargable", "informe pdf", "reporte pdf", "genera pdf",
        "exporta pdf", "descargar informe", "descargar pdf"
    ]
    return any(p in q for p in patrones)


def es_followup_de_respuesta_anterior(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower().strip()

    patrones_followup = [
        "de ese", "de esa", "de esos", "de esas", "ese ranking", "esa tabla", "eso",
        "esos resultados", "esas comunas", "esas ciudades", "esas ventas", "esos clientes",
        "profundiza", "ahonda", "explica más", "explica mas", "desarrolla", "compara con",
        "y en ese caso", "y de ahí", "y de ahi", "de los que me diste", "de lo anterior",
        "del ranking anterior", "de la respuesta anterior", "de lo que me mostraste",
        "de lo que dijiste", "de lo que me dijiste", "siguiendo con eso", "sobre eso",
        "en ese caso", "en ese escenario", "cual de esos", "cuál de esos",
        "cual de estas", "cuál de estas", "el segundo", "la segunda",
        "el primero", "la primera", "el top 3", "el top 5", "el ranking",
        "ese top", "ese resultado", "esa respuesta", "vitacura?", "y vitacura?"
    ]

    if any(p in q for p in patrones_followup):
        return True

    palabras_cortas = q.split()
    if len(palabras_cortas) <= 12:
        indicadores = [
            "cual", "cuál", "ese", "esa", "eso", "esas", "esos",
            "mismo", "misma", "tambien", "también", "ahora", "entonces", "y", "pero"
        ]
        if any(x in palabras_cortas for x in indicadores):
            return True

    return False


# ============================================================
# NUEVAS FUNCIONES GEOGRÁFICAS CRÍTICAS
# ============================================================

def es_pregunta_geografica(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower()
    keywords = [
        "comuna", "comunas", "región", "region", "provincia", "provincias",
        "ciudad", "ciudades", "geografía", "geografia", "territorio", "mapa"
    ]
    return any(k in q for k in keywords)


def es_pregunta_geografica_de_clientes(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower()

    hay_geo = es_pregunta_geografica(pregunta_usuario)
    hay_cliente = any(k in q for k in [
        "cliente", "clientes", "distribución", "distribucion", "porcentaje",
        "participación", "participacion", "share", "concentración", "concentracion",
        "penetración", "penetracion", "cuántos clientes", "cuantos clientes",
        "clientes por comuna", "clientes por region", "clientes por provincia"
    ])

    hay_metricas_clientes = any(k in q for k in [
        "clientes", "cliente único", "cliente unico", "clientes únicos", "clientes unicos"
    ])

    return hay_geo and (hay_cliente or hay_metricas_clientes)


def construir_bloque_reglas_geograficas_criticas() -> str:
    return f"""
## REGLAS CRÍTICAS DE ANÁLISIS GEOGRÁFICO

La asignación geográfica de un cliente debe obtenerse exclusivamente desde la tabla:
- `{MAPA_VERDAD["dim_clientes"]}`

utilizando la columna:
- `cliente_id`

### REGLA 1 — ORIGEN DE LA COMUNA
La comuna de un cliente debe provenir únicamente de:
- `dim_clientes.comuna_principal`

Nunca utilizar la comuna proveniente de:
- `fact_pedidos`
- `fact_pedidos_limpios`
- `geo_fuentes_raw`
- `geo_fuentes_limpias`
- `geo_inteligencia_base`

Las tablas de pedidos pueden contener direcciones múltiples por cliente y NO deben usarse para asignar geografía del cliente.

### REGLA 2 — CLIENTE ÚNICO POR COMUNA
Cada cliente debe pertenecer a una sola comuna al momento de calcular distribución geográfica.

Esto se logra cruzando:
- `fact_pedidos.cliente_id`
con
- `dim_clientes.cliente_id`

### REGLA 3 — CÁLCULO CORRECTO DE CLIENTES POR COMUNA
El patrón correcto es:

SELECT
    dc.comuna_principal,
    COUNT(DISTINCT fp.cliente_id) clientes
FROM `{MAPA_VERDAD["fact_pedidos"]}` fp
JOIN `{MAPA_VERDAD["dim_clientes"]}` dc
ON fp.cliente_id = dc.cliente_id
GROUP BY dc.comuna_principal

### REGLA 4 — CÁLCULO DE PORCENTAJES
El denominador debe ser el total de clientes únicos:
- `COUNT(DISTINCT cliente_id)`

Nunca sumar porcentajes si un cliente aparece en más de una comuna.

### REGLA 5 — USO DE TABLAS
- `dim_clientes` → geografía del cliente
- `fact_pedidos` → ventas / actividad / fechas / recurrencia
- `fact_pedidos_limpios` → análisis de productos

### REGLA 6 — DETECCIÓN DE ERRORES
Si los porcentajes por comuna superan el 100% total o varias comunas muestran porcentajes cercanos al 50% simultáneamente, debes asumir duplicidad geográfica y recalcular utilizando únicamente `dim_clientes`.

### REGLA 7 — MODELO MENTAL
La geografía pertenece al cliente.
Las compras pertenecen al pedido.

Nunca inferir geografía desde pedidos.
"""


def obtener_bloque_geografico_dinamico(pregunta_usuario: str) -> str:
    if es_pregunta_geografica_de_clientes(pregunta_usuario):
        return construir_bloque_reglas_geograficas_criticas()

    if es_pregunta_geografica(pregunta_usuario):
        return f"""
## REGLA GEOGRÁFICA DE PRECAUCIÓN
Si la pregunta busca distribución, porcentaje, concentración o conteo de clientes por geografía, debes usar obligatoriamente:
- `{MAPA_VERDAD["dim_clientes"]}` como fuente de geografía del cliente
- `{MAPA_VERDAD["fact_pedidos"]}` solo para actividad, ventas, fechas y recurrencia

Nunca asignes comuna del cliente desde tablas de pedidos ni desde tablas raw geográficas.
"""
    return ""


def validar_sql_geografica_critica(query: str, pregunta_usuario: str) -> Tuple[bool, str]:
    if not es_pregunta_geografica_de_clientes(pregunta_usuario):
        return True, ""

    sql = limpiar_sql(query).lower()

    forbidden_patterns = [
        "fp.comuna",
        "fp.comuna_principal",
        "fact_pedidos.comuna",
        "fact_pedidos_limpios.comuna",
        "geo_fuentes_raw",
        "geo_fuentes_limpias",
        "geo_inteligencia_base",
    ]
    for pat in forbidden_patterns:
        if pat in sql:
            return False, (
                "Para análisis geográfico de clientes no se puede inferir la geografía desde pedidos ni desde tablas geográficas raw/inteligencia. "
                "La comuna debe salir de dim_clientes.comuna_principal."
            )

    usa_dim_clientes = f"`{MAPA_VERDAD['dim_clientes']}`".lower() in sql
    usa_fact_pedidos = f"`{MAPA_VERDAD['fact_pedidos']}`".lower() in sql

    if not usa_dim_clientes:
        return False, "Falta usar dim_clientes como fuente exclusiva de geografía del cliente."

    if not usa_fact_pedidos:
        return False, "Falta usar fact_pedidos para calcular clientes activos/reales sobre actividad."

    if "comuna_principal" not in sql:
        return False, "La asignación geográfica debe usar dim_clientes.comuna_principal."

    if "count(distinct" not in sql:
        return False, "El cálculo geográfico de clientes debe usar COUNT(DISTINCT cliente_id)."

    if "join" not in sql:
        return False, "La consulta debe cruzar fact_pedidos.cliente_id con dim_clientes.cliente_id."

    return True, ""


def detectar_followup_semantico_con_modelo(
    pregunta_usuario: str,
    ultima_pregunta: str,
    ultima_respuesta: str,
    ultimo_sql: str = "",
    ultimo_df_texto: str = "",
) -> bool:
    if not ultima_pregunta and not ultima_respuesta:
        return False

    prompt = f"""
Evalúa si la NUEVA_PREGUNTA depende del contexto anterior o si puede entenderse totalmente sola.

Responde únicamente con una de estas dos opciones:
- FOLLOWUP
- NUEVA

Criterio:
- Responde FOLLOWUP si la nueva pregunta se refiere, aunque sea de forma implícita, a rankings, tablas, hallazgos, comunas, clientes, períodos, comparaciones, conclusiones o resultados mencionados antes.
- Responde FOLLOWUP aunque no use pronombres obvios como "eso", "ese ranking" o "lo anterior", si semánticamente está continuando la conversación.
- Responde NUEVA solo si cambia claramente de tema o puede resolverse sin depender del contexto previo.

ULTIMA_PREGUNTA:
{ultima_pregunta}

ULTIMA_RESPUESTA:
{ultima_respuesta}

ULTIMO_SQL:
{ultimo_sql}

ULTIMOS_DATOS:
{ultimo_df_texto}

NUEVA_PREGUNTA:
{pregunta_usuario}
"""
    try:
        response = generar_contenido_con_fallback(prompt, MODEL_RESPONSE)
        texto = obtener_texto_modelo(response).strip().upper()
        return "FOLLOWUP" in texto
    except Exception:
        return False


def debe_usar_contexto_previo(pregunta_usuario: str) -> bool:
    if es_followup_de_respuesta_anterior(pregunta_usuario):
        return True

    last_user_question = st.session_state.get("last_user_question", "")
    last_answer = st.session_state.get("last_answer", "")
    last_sql = st.session_state.get("last_sql", "")
    last_df = st.session_state.get("last_df", None)

    last_df_texto = ""
    if last_df is not None and not last_df.empty:
        last_df_texto = last_df.head(20).to_string(index=False)

    return detectar_followup_semantico_con_modelo(
        pregunta_usuario=pregunta_usuario,
        ultima_pregunta=last_user_question,
        ultima_respuesta=last_answer,
        ultimo_sql=last_sql,
        ultimo_df_texto=last_df_texto,
    )


def resumir_contexto_anterior_para_followup() -> str:
    partes = []

    last_user_question = st.session_state.get("last_user_question", "")
    last_sql = st.session_state.get("last_sql", "")
    last_answer = st.session_state.get("last_answer", "")
    last_df = st.session_state.get("last_df", None)

    if last_user_question:
        partes.append(f"ULTIMA_PREGUNTA_USUARIO:\n{last_user_question}")

    if last_answer:
        partes.append(f"ULTIMA_RESPUESTA_ASISTENTE:\n{last_answer}")

    if last_sql:
        partes.append(f"ULTIMO_SQL_USADO:\n{last_sql}")

    if last_df is not None and not last_df.empty:
        partes.append("ULTIMOS_DATOS_TABULARES:\n" + last_df.head(20).to_string(index=False))

    return "\n\n".join(partes)


def construir_contexto_historial(
    messages: List[Dict[str, str]],
    pregunta_actual: str,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    recientes = messages[-max_messages:]
    historial_chat = "\n".join([f"{m['role']}: {m['content']}" for m in recientes])

    contexto_extra = ""
    if debe_usar_contexto_previo(pregunta_actual):
        contexto_extra = resumir_contexto_anterior_para_followup()

    if contexto_extra:
        return f"{historial_chat}\n\nCONTEXTO_ESTRUCTURADO_PREVIO:\n{contexto_extra}"

    return historial_chat


# ============================================================
# TABLAS PRIORITARIAS
# ============================================================

def obtener_tablas_prioritarias(pregunta_usuario: str) -> List[str]:
    q = pregunta_usuario.lower()
    prioridades: List[str] = []

    if any(x in q for x in ["venta", "ventas", "mes", "año", "anio", "comparar", "facturación", "facturacion", "ticket", "ingresos"]):
        prioridades += [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["fact_pedidos"],
            MAPA_VERDAD["geo_inteligencia_base"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["dim_clientes"],
        ]

    if any(x in q for x in ["cliente", "clientes", "ltv", "segmento", "recurrente", "nuevo", "riesgo", "inactividad", "churn", "reactivar"]):
        prioridades += [
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["fact_pedidos"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["geo_filtros_expandibles"],
        ]

    if any(x in q for x in ["producto", "productos", "retención", "retencion", "recompra", "gancho", "afinidad", "bundle", "mix", "canasta"]):
        prioridades += [
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["afinidad_productos"],
            MAPA_VERDAD["fact_pedido_productos"],
            MAPA_VERDAD["fact_pedidos_limpios"],
        ]

    if es_pregunta_geografica_de_clientes(pregunta_usuario):
        prioridades += [
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["fact_pedidos"],
            MAPA_VERDAD["geo_filtros_expandibles"],
            MAPA_VERDAD["geo_catalogo_chile_normalizado"],
        ]
    elif any(x in q for x in ["comuna", "comunas", "ciudad", "ciudades", "provincia", "provincias", "región", "region", "geo", "geografia", "geografía"]):
        prioridades += [
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["geo_inteligencia_base"],
            MAPA_VERDAD["geo_filtros_expandibles"],
            MAPA_VERDAD["geo_catalogo_chile_normalizado"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if any(x in q for x in ["calidad", "auditoría", "auditoria", "error", "cobertura"]):
        prioridades += [
            MAPA_VERDAD["auditoria_datos"],
            MAPA_VERDAD["geo_no_resueltos"],
        ]

    if any(x in q for x in [
        "inversión", "inversion", "medios", "paid media", "google ads", "meta ads",
        "presupuesto", "budget", "roas", "cpa", "cac", "performance",
        "retargeting", "prospecting", "audiencia", "audiencias", "escalar"
    ]):
        prioridades += [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["afinidad_productos"],
            MAPA_VERDAD["geo_inteligencia_base"],
            MAPA_VERDAD["dim_clientes"],
        ]

    if es_pregunta_informe(pregunta_usuario):
        prioridades += [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["auditoria_datos"],
            MAPA_VERDAD["geo_inteligencia_base"],
            MAPA_VERDAD["dim_clientes"],
        ]

    if not prioridades:
        prioridades = [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    return list(dict.fromkeys(prioridades))


def filtrar_tablas_existentes(tablas_objetivo: List[str]) -> List[str]:
    disponibles = []
    for table_fq in tablas_objetivo:
        try:
            bq_client.get_table(table_fq)
            disponibles.append(table_fq)
        except Exception:
            continue
    return disponibles


def obtener_esquemas_tablas(tablas_objetivo: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    esquemas: Dict[str, List[Tuple[str, str]]] = {}
    for table_fq in tablas_objetivo:
        try:
            table = bq_client.get_table(table_fq)
            if table.schema:
                esquemas[table_fq] = [(field.name, field.field_type) for field in table.schema]
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

    bloqueadas = [
        "INSERT ", "UPDATE ", "DELETE ", "MERGE ", "DROP ", "ALTER ",
        "TRUNCATE ", "CREATE ", "CALL ", "EXECUTE IMMEDIATE ", "EXPORT DATA "
    ]
    for palabra in bloqueadas:
        if palabra in upper:
            return False, f"Se detectó una operación no permitida: {palabra.strip()}."

    if re.search(r"SELECT\s+\*", query_limpio, re.IGNORECASE):
        return False, "No se permite SELECT *."

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


def diagnosticar_tablas() -> Dict[str, bool]:
    estado = {}
    for alias, table_fq in MAPA_VERDAD.items():
        try:
            bq_client.get_table(table_fq)
            estado[alias] = True
        except Exception:
            estado[alias] = False
    return estado


# ============================================================
# PROMPTS
# ============================================================

def construir_prompt_sql(
    pregunta_usuario: str,
    historial_contexto: str,
    tablas_prioritarias: List[str],
    esquemas_texto: str,
    error_previo: str = "",
) -> str:
    tablas_texto = "\n".join([f"- {t}" for t in tablas_prioritarias])
    bloque_geo = obtener_bloque_geografico_dinamico(pregunta_usuario)

    return f"""
Eres NobleBotAI, motor de consulta SQL de BigQuery para inteligencia comercial, performance y marketing 360.

## OBJETIVO
Debes escribir una única consulta SQL válida, segura y eficiente para responder la pregunta del usuario usando exclusivamente el Mapa de Verdad oficial.

## MAPA DE VERDAD OFICIAL
{chr(10).join([f"- {v}" for v in MAPA_VERDAD.values()])}

## TABLAS PRIORITARIAS PARA ESTA PREGUNTA
{tablas_texto}

## ESQUEMAS REALES DISPONIBLES
{esquemas_texto}

## MEMORIA OPERATIVA
{historial_contexto}

## PREGUNTA DEL USUARIO
{pregunta_usuario}

## ERROR PREVIO A CORREGIR
{error_previo}

{bloque_geo}

## REGLAS DE NEGOCIO
- prioriza tablas AI cuando resuelvan la pregunta con claridad.
- usa CORE solo si la pregunta requiere granularidad real.
- si la pregunta es geográfica y trata sobre clientes o distribución geográfica de clientes, la geografía del cliente debe salir exclusivamente de `dim_clientes.comuna_principal`.
- si la pregunta es geográfica y trata sobre actividad, ventas o recurrencia, puedes usar `fact_pedidos` para medir la actividad y `dim_clientes` para fijar la geografía del cliente.
- si el usuario pregunta por región o provincia, no filtres texto libremente en las tablas core.
- si el usuario pregunta por región, expande a comunas usando `geo_filtros_expandibles` y luego cruza esas comunas contra `dim_clientes.comuna_principal`.
- si el usuario pregunta por provincia, expande a comunas usando `geo_filtros_expandibles` y luego cruza esas comunas contra `dim_clientes.comuna_principal`.
- si el usuario pregunta por comuna en contexto de clientes, filtra usando `dim_clientes.comuna_principal`.
- para rankings y agregados rápidos por ventas/geografía puedes usar tablas AI, pero no para asignar la geografía de un cliente.
- nunca vuelvas a usar columnas geográficas crudas del core como lógica final de asignación de geografía del cliente si existe equivalente canónico en `dim_clientes`.
- si el follow-up depende de una respuesta anterior, continúa sobre ese contexto.
- si la pregunta parece ambigua, elige la ruta más segura y ejecutiva.
- si la pregunta es distribución geográfica de clientes, cada cliente debe pertenecer a una sola comuna y el conteo debe ser COUNT(DISTINCT cliente_id).

## REGLAS TECNICAS OBLIGATORIAS
1. Responde únicamente con SQL.
2. La consulta debe comenzar con SELECT o WITH.
3. Usa backticks en todas las tablas.
4. No inventes tablas, columnas, joins ni métricas.
5. No uses columnas fuera de los esquemas entregados.
6. No uses CREATE, INSERT, UPDATE, DELETE, MERGE, DROP, ALTER, TRUNCATE, CALL ni EXECUTE IMMEDIATE.
7. No uses SELECT *.
8. Usa alias legibles.
9. Si la pregunta pide ranking, incluye ORDER BY.
10. Si la pregunta pide top o detalle abierto, aplica un LIMIT razonable.
11. Si del historial existe una referencia clara y útil, puedes reutilizarla.
12. Si una tabla agregada resuelve la pregunta, no escales a una tabla más pesada.
13. Si la pregunta del usuario hace referencia a una respuesta anterior como "ese ranking", "eso", "lo anterior" o similares, debes usar el CONTEXTO_ESTRUCTURADO_PREVIO para inferir correctamente a qué resultado se refiere.
14. Si la pregunta es por región o provincia y trata de clientes, la unidad analítica base sigue siendo la comuna del cliente en `dim_clientes`, y luego agregas.
15. Evita scans innecesarios.
16. Si la pregunta es distribución geográfica de clientes, no puedes usar `geo_inteligencia_base`, `geo_fuentes_raw`, `geo_fuentes_limpias`, `fact_pedidos.comuna` ni otras comunas de pedido como fuente geográfica principal.
17. Si calculas porcentajes por comuna, el denominador debe ser el total de clientes únicos y los porcentajes no deben duplicar clientes en varias comunas.

## FORMATO DE SALIDA
- Entrega solo SQL puro.
- No expliques nada.
- No uses markdown.
"""


def construir_prompt_respuesta(
    pregunta_usuario: str,
    historial_contexto: str,
    sql_usado: str,
    datos_texto: str,
    num_filas: int,
) -> str:
    return f"""
Eres NobleBotAI, asistente de inteligencia comercial, performance y marketing 360.

## OBJETIVO
Transforma resultados tabulares en una respuesta ejecutiva, clara, precisa y accionable.

## TONO
- Ejecutivo
- Estratégico
- Directo
- Profesional
- Humano
- Basado en evidencia

## REGLAS
1. Responde siempre en español.
2. Empieza con: "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽."
3. Luego sigue con: "Consulté en las tablas oficiales y aquí tienes el resultado..."
4. Abre con el hallazgo principal.
5. Si corresponde, interpreta implicancias de negocio.
6. Si hay cifras monetarias, preséntalas como CLP.
7. No inventes métricas.
8. Si la evidencia es limitada, dilo.
9. Cierra con una recomendación concreta.
10. Cuando sea útil, estructura la respuesta en:
   - Diagnóstico
   - Qué significa
   - Riesgo u oportunidad
   - Próximo paso
11. Si la pregunta es geográfica y trata sobre clientes, recuerda que la geografía válida del cliente proviene de `dim_clientes.comuna_principal`.
12. Si detectas porcentajes sospechosos o que suman más de 100%, dilo explícitamente como riesgo metodológico.

## CONTEXTO DE CONVERSACION
{historial_contexto}

## PREGUNTA
{pregunta_usuario}

## SQL USADO
{sql_usado}

## FILAS DEVUELTAS
{num_filas}

## DATOS
{datos_texto}
"""


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def generar_sql_con_reintentos(pregunta_usuario: str, historial_contexto: str) -> Tuple[str, pd.DataFrame]:
    tablas_prioritarias = obtener_tablas_prioritarias(pregunta_usuario)
    tablas_disponibles = filtrar_tablas_existentes(tablas_prioritarias)

    if not tablas_disponibles:
        fallback_core = [
            MAPA_VERDAD["fact_pedidos"],
            MAPA_VERDAD["fact_pedido_productos"],
            MAPA_VERDAD["fact_pedidos_limpios"],
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["bridge_cliente_identidades"],
        ]
        tablas_disponibles = filtrar_tablas_existentes(fallback_core)

    if not tablas_disponibles:
        raise ValueError("No pude encontrar tablas disponibles. Revisa el proyecto, datasets y permisos.")

    esquemas = obtener_esquemas_tablas(tablas_disponibles)
    if not esquemas:
        raise ValueError(f"No pude obtener esquemas de las tablas disponibles. Tablas detectadas: {tablas_disponibles}")

    esquemas_texto = formatear_esquemas_para_prompt(esquemas)

    error_previo = ""
    ultimo_error = ""
    ultimo_sql_generado = ""

    for intento in range(MAX_SQL_RETRIES):
        prompt_sql = construir_prompt_sql(
            pregunta_usuario=pregunta_usuario,
            historial_contexto=historial_contexto,
            tablas_prioritarias=tablas_disponibles,
            esquemas_texto=esquemas_texto,
            error_previo=error_previo,
        )

        respuesta_sql = generar_contenido_con_fallback(prompt_sql, MODEL_SQL)

        query_limpio = limpiar_sql(obtener_texto_modelo(respuesta_sql))
        ultimo_sql_generado = query_limpio

        es_valida, motivo = validar_sql(query_limpio)
        if not es_valida:
            ultimo_error = f"[Intento {intento + 1}] SQL inválido: {motivo}\n\nSQL generado:\n{query_limpio}"
            error_previo = f"La consulta previa fue inválida. Motivo: {motivo}"
            continue

        es_valida_geo, motivo_geo = validar_sql_geografica_critica(query_limpio, pregunta_usuario)
        if not es_valida_geo:
            ultimo_error = f"[Intento {intento + 1}] SQL geográfico inválido: {motivo_geo}\n\nSQL generado:\n{query_limpio}"
            error_previo = f"La consulta previa incumplió reglas geográficas críticas. Motivo: {motivo_geo}"
            continue

        try:
            df = ejecutar_query_segura(query_limpio)
            return query_limpio, df
        except Exception as e:
            ultimo_error = f"[Intento {intento + 1}] Error BigQuery: {str(e)}\n\nSQL generado:\n{query_limpio}"
            error_previo = f"La consulta previa falló en BigQuery. Error: {str(e)}"

    raise ValueError(ultimo_error or f"No pude construir una consulta válida. Último SQL:\n{ultimo_sql_generado}")


def responder_como_noblebot(
    pregunta_usuario: str,
    historial_contexto: str,
    sql_usado: str,
    df_datos: pd.DataFrame,
) -> str:
    if df_datos.empty:
        return (
            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
            "Consulté en las tablas oficiales y aquí tienes el resultado: no encontré filas que calcen exactamente con esa búsqueda.\n\n"
            "La lectura de negocio no es que no exista oportunidad, sino que el filtro actual está demasiado estrecho o no coincide con la forma en que la señal está guardada.\n\n"
            "Recomendación inmediata: conviene reformular la búsqueda por período, comuna, región, provincia, cliente, producto o segmento."
        )

    datos_texto = resumir_dataframe_para_prompt(df_datos)

    prompt_final = construir_prompt_respuesta(
        pregunta_usuario=pregunta_usuario,
        historial_contexto=historial_contexto,
        sql_usado=sql_usado,
        datos_texto=datos_texto,
        num_filas=len(df_datos),
    )

    respuesta_final = generar_contenido_con_fallback(prompt_final, MODEL_RESPONSE)
    texto = obtener_texto_modelo(respuesta_final).strip()

    if not texto:
        texto = (
            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
            "Consulté en las tablas oficiales y aquí tienes el resultado.\n\n"
            "Recuperé correctamente los datos, pero la redacción final salió vacía. El hallazgo sigue disponible en el SQL y en la tabla."
        )

    return texto


# ============================================================
# GRAFICOS ON-DEMAND
# ============================================================

def elegir_columnas_para_grafico(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], str]:
    if df is None or df.empty:
        return None, None, "bar"

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    preferencias_y = [
        "ventas_totales_clp", "venta_clp", "ventas", "total", "ingresos",
        "clientes_unicos", "clientes", "pedidos_totales", "pedidos",
        "ticket_promedio_clp", "ticket"
    ]
    y_col = None
    for pref in preferencias_y:
        for col in numeric_cols:
            if pref in col.lower():
                y_col = col
                break
        if y_col:
            break
    if not y_col and numeric_cols:
        y_col = numeric_cols[0]

    x_col = None
    preferencias_x = [
        "comuna", "provincia", "region", "anio", "año", "mes",
        "fecha", "producto", "segmento", "cliente"
    ]
    for pref in preferencias_x:
        for col in all_cols:
            if pref in col.lower() and col != y_col:
                x_col = col
                break
        if x_col:
            break

    if not x_col:
        non_numeric = [c for c in all_cols if c not in numeric_cols]
        if non_numeric:
            x_col = non_numeric[0]
        elif len(all_cols) >= 2:
            x_col = all_cols[0] if all_cols[0] != y_col else all_cols[1]

    chart_type = "bar"
    if x_col and any(k in x_col.lower() for k in ["fecha", "anio", "año", "mes"]):
        chart_type = "line"

    return x_col, y_col, chart_type


def generar_grafico_desde_dataframe(df: pd.DataFrame, titulo: str):
    x_col, y_col, chart_type = elegir_columnas_para_grafico(df)

    if not x_col or not y_col:
        return None, "No encontré columnas adecuadas para graficar."

    df_plot = df.copy().head(50)

    if chart_type == "line":
        fig = px.line(df_plot, x=x_col, y=y_col, markers=True, title=titulo)
    else:
        fig = px.bar(df_plot, x=x_col, y=y_col, title=titulo)

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig, ""


# ============================================================
# PDF EJECUTIVO
# ============================================================

def dataframe_to_table_data(df: pd.DataFrame, max_rows: int = 20) -> List[List[str]]:
    if df is None or df.empty:
        return [["Sin datos"]]

    df2 = df.head(max_rows).copy()
    headers = [str(c) for c in df2.columns.tolist()]
    rows = [[str(v) for v in row] for row in df2.fillna("").astype(str).values.tolist()]
    return [headers] + rows


def build_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()
    style_h1 = styles["Heading1"]
    style_h2 = styles["Heading2"]
    style_body = styles["BodyText"]
    style_small = ParagraphStyle(
        "small",
        parent=style_body,
        fontSize=8,
        leading=10,
    )

    story = []
    story.append(Paragraph(f"Informe Ejecutivo - {BRAND_ID}", style_h1))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), style_small))
    story.append(Spacer(1, 12))

    report_items = st.session_state.get("report_items", [])
    charts_for_pdf = st.session_state.get("generated_charts", [])

    if not report_items:
        story.append(Paragraph("No hay contenido suficiente en la sesión para generar el informe.", style_body))
    else:
        for idx, item in enumerate(report_items, start=1):
            story.append(Paragraph(f"Sección {idx}", style_h2))
            story.append(Paragraph(f"<b>Pregunta:</b> {item.get('question', '')}", style_body))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<b>Respuesta ejecutiva:</b><br/>{item.get('answer', '').replace(chr(10), '<br/>')}", style_body))
            story.append(Spacer(1, 8))

            sql_text = item.get("sql", "")
            if sql_text:
                story.append(Paragraph("<b>SQL usado</b>", style_body))
                story.append(Paragraph(sql_text.replace("\n", "<br/>"), style_small))
                story.append(Spacer(1, 8))

            df = item.get("df")
            if df is not None and not df.empty:
                story.append(Paragraph("<b>Tabla de evidencia</b>", style_body))
                table_data = dataframe_to_table_data(df, max_rows=12)
                table = Table(table_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(table)
                story.append(Spacer(1, 10))

        if charts_for_pdf:
            story.append(PageBreak())
            story.append(Paragraph("Gráficos generados durante la sesión", style_h2))
            story.append(Spacer(1, 8))

            for chart_item in charts_for_pdf:
                fig = chart_item.get("fig")
                title = chart_item.get("title", "Gráfico")
                if fig is None:
                    continue

                try:
                    image_bytes = pio.to_image(fig, format="png", width=1200, height=700, scale=2)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(image_bytes)
                        img_path = tmp.name

                    story.append(Paragraph(title, style_body))
                    story.append(Spacer(1, 6))
                    story.append(Image(img_path, width=17 * cm, height=10 * cm))
                    story.append(Spacer(1, 12))
                except Exception:
                    story.append(Paragraph(f"{title} (no se pudo incrustar imagen del gráfico en PDF)", style_small))
                    story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

st.set_page_config(page_title="NobleBotAI 🐷🐽", layout="wide")
st.title("NobleBotAI 🐷🐽")
st.caption("Inteligencia comercial, performance y marketing 360 para agencia")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

if "last_df" not in st.session_state:
    st.session_state.last_df = None

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "report_items" not in st.session_state:
    st.session_state.report_items = []

if "generated_charts" not in st.session_state:
    st.session_state.generated_charts = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Configuración")
    st.markdown(f"**Proyecto:** `{PROJECT_ID}`")
    st.markdown(f"**Location:** `{LOCATION}`")
    st.markdown(f"**Marca base:** `{BRAND_ID}`")
    st.markdown(f"**Modelo SQL:** `{MODEL_SQL}`")
    st.markdown(f"**Modelo respuesta:** `{MODEL_RESPONSE}`")
    st.markdown(f"**Modelo contraste externo:** `{MODEL_MEDIA}`")

    st.subheader("Mapa de Verdad")
    for alias, fq_table in MAPA_VERDAD.items():
        st.markdown(f"- **{alias}** → `{fq_table}`")

    st.subheader("Estado tablas oficiales")
    estado_tablas = diagnosticar_tablas()
    for alias, existe in estado_tablas.items():
        icono = "✅" if existe else "❌"
        st.markdown(f"{icono} **{alias}**")

    if st.button("Limpiar memoria"):
        st.session_state.messages = []
        st.session_state.last_sql = ""
        st.session_state.last_df = None
        st.session_state.last_user_question = ""
        st.session_state.last_answer = ""
        st.session_state.report_items = []
        st.session_state.generated_charts = []
        st.rerun()


# ============================================================
# RENDER HISTORIAL
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# FLUJO CHAT
# ============================================================

if prompt := st.chat_input("Pregúntame por clientes, ventas, productos, comunas, regiones, provincias o inversión en medios..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # --------------------------------------------------------
            # GRAFICO ON-DEMAND
            # --------------------------------------------------------
            if es_solicitud_grafico(prompt):
                df_chart = st.session_state.get("last_df", None)

                if df_chart is None or df_chart.empty:
                    respuesta = (
                        "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                        "No tengo datos previos en memoria para graficar todavía. Primero necesito una respuesta tabular o analítica en esta sesión."
                    )
                    st.markdown(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                else:
                    fig, error_chart = generar_grafico_desde_dataframe(
                        df_chart,
                        f"Gráfico on-demand - {prompt}"
                    )

                    if fig is None:
                        respuesta = (
                            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                            f"No pude construir el gráfico con los datos actuales. Motivo: {error_chart}"
                        )
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    else:
                        respuesta = (
                            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                            "Usé los datos que ya estaban en memoria de la aplicación, sin volver a consultar BigQuery, para que el gráfico sea idéntico a la respuesta anterior."
                        )
                        st.markdown(respuesta)
                        st.plotly_chart(fig, use_container_width=True)

                        st.session_state.generated_charts.append({
                            "title": f"Gráfico on-demand - {prompt}",
                            "fig": fig,
                        })

                        st.session_state.messages.append({"role": "assistant", "content": respuesta})

            # --------------------------------------------------------
            # PDF ON-DEMAND
            # --------------------------------------------------------
            elif es_solicitud_pdf(prompt):
                pdf_bytes = build_pdf_bytes()
                respuesta = (
                    "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                    "Compilé el contenido ya generado durante esta sesión en un informe PDF ejecutivo, sin volver a consultar BigQuery."
                )
                st.markdown(respuesta)
                st.download_button(
                    label="Descargar informe PDF",
                    data=pdf_bytes,
                    file_name=f"informe_{BRAND_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
                st.session_state.messages.append({"role": "assistant", "content": respuesta})

            # --------------------------------------------------------
            # FLUJO ANALITICO NORMAL
            # --------------------------------------------------------
            else:
                with st.spinner("NobleBotAI está consultando las tablas oficiales..."):
                    memoria_reciente = construir_contexto_historial(
                        st.session_state.messages,
                        prompt
                    )

                    sql_usado, df_datos = generar_sql_con_reintentos(prompt, memoria_reciente)

                    respuesta_final = responder_como_noblebot(
                        prompt,
                        memoria_reciente,
                        sql_usado,
                        df_datos,
                    )

                    st.markdown(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})

                    st.session_state.last_user_question = prompt
                    st.session_state.last_answer = respuesta_final
                    st.session_state.last_sql = sql_usado
                    st.session_state.last_df = df_datos.copy()

                    st.session_state.report_items.append({
                        "question": prompt,
                        "answer": respuesta_final,
                        "sql": sql_usado,
                        "df": df_datos.copy(),
                    })

                    with st.expander("Ver SQL usado"):
                        st.code(sql_usado, language="sql")

                    with st.expander("Ver datos tabulares"):
                        st.dataframe(df_datos.head(MAX_ROWS_RESULT), use_container_width=True)

        except Exception as e:
            respuesta_error = (
                "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                "Consulté las rutas oficiales, pero hubo un problema técnico al construir o ejecutar la consulta.\n\n"
                "La oportunidad sigue ahí; lo que falló fue la ruta de consulta, no la lógica comercial. "
                "Prueba con una pregunta más específica por comuna, provincia, región, producto, período, cliente o segmento."
            )
            st.markdown(respuesta_error)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_error})

            with st.expander("Detalle técnico"):
                st.code(str(e))
