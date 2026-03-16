import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from google import genai
from google.cloud import bigquery
from google.oauth2 import service_account
from google.genai import types


# ============================================================
# CONFIG
# ============================================================
PROJECT_ID = st.secrets.get("PROJECT_ID", "data-marketing-360")
LOCATION = st.secrets.get("GOOGLE_CLOUD_LOCATION", "global")
BRAND_ID = st.secrets.get("BRAND_ID", "campo_noble")

CORE_DATASET = f"{BRAND_ID}_core"
AI_DATASET = f"{BRAND_ID}_ai"

# Gemini 3.1 en Vertex AI con cuenta de servicio
MODEL_SQL = st.secrets.get("MODEL_SQL", "gemini-3.1-pro")
MODEL_ROUTER = st.secrets.get("MODEL_ROUTER", "gemini-3.1-flash-lite")
MODEL_CHAT = st.secrets.get("MODEL_CHAT", "gemini-3.1-flash-lite")
MODEL_REPORT = st.secrets.get("MODEL_REPORT", "gemini-3.1-flash")

MAX_ROWS_RESULT = int(st.secrets.get("MAX_ROWS_RESULT", 200))
MAX_HISTORY_MESSAGES = int(st.secrets.get("MAX_HISTORY_MESSAGES", 8))
MAX_SQL_RETRIES = int(st.secrets.get("MAX_SQL_RETRIES", 2))
MAX_BYTES_BILLED = int(st.secrets.get("MAX_BYTES_BILLED", 5 * 1024 * 1024 * 1024))
MAX_SCHEMA_COLUMNS = int(st.secrets.get("MAX_SCHEMA_COLUMNS", 80))
ENABLE_SQL_CACHE = bool(st.secrets.get("ENABLE_SQL_CACHE", True))


# ============================================================
# MAPA DE VERDAD
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

    # AI geografia nueva
    "geo_maestro_comunas": f"{PROJECT_ID}.{AI_DATASET}.geo_maestro_comunas",
    "geo_no_match": f"{PROJECT_ID}.{AI_DATASET}.geo_no_match",
    "geo_pedidos_base": f"{PROJECT_ID}.{AI_DATASET}.geo_pedidos_base",
    "resumen_geografia": f"{PROJECT_ID}.{AI_DATASET}.resumen_geografia",
}
ALLOWED_TABLES = set(MAPA_VERDAD.values())


# ============================================================
# CLIENTES
# ============================================================
creds_dict = dict(st.secrets["gcp_service_account"])
creds = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

bq_client = bigquery.Client(credentials=creds, project=PROJECT_ID)

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=creds,
)


# ============================================================
# ESTADO
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_route" not in st.session_state:
    st.session_state.last_route = ""
if "discussed_topics" not in st.session_state:
    st.session_state.discussed_topics = []
if "pending_report" not in st.session_state:
    st.session_state.pending_report = False
if "report_mode" not in st.session_state:
    st.session_state.report_mode = False
if "report_items" not in st.session_state:
    st.session_state.report_items = []
if "report_goal" not in st.session_state:
    st.session_state.report_goal = ""
if "sql_cache" not in st.session_state:
    st.session_state.sql_cache = {}


# ============================================================
# DATOS / MODELOS AUXILIARES
# ============================================================
@dataclass
class RouteDecision:
    route: str
    intent: str
    response_mode: str
    needs_bq: bool
    followup: bool
    asks_report_scope: bool
    topic: str
    reason: str


SMALLTALK_PATTERNS = [
    r"^hola+$", r"^holi+$", r"^buenas$", r"^buenos dias$", r"^buenas tardes$",
    r"^buenas noches$", r"^gracias$", r"^ok+$", r"^dale$", r"^perfecto$",
    r"^entiendo$", r"^listo$", r"^genial$",
]

GREETING_REPLY = (
    "Hola. Soy NobleBotAI 🐷🐽. Estoy listo para ayudarte con ventas, clientes, productos, geografía, "
    "medios e informes. También puedo seguir el contexto de esta conversación sin consultar BigQuery si no hace falta."
)


# ============================================================
# UTILIDADES
# ============================================================
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def lower_clean(text: str) -> str:
    return clean_text(text).lower()


def obtener_texto_modelo(response) -> str:
    return getattr(response, "text", "") or ""


def limpiar_sql(texto: str) -> str:
    return (texto or "").replace("```sql", "").replace("```", "").strip()


def es_smalltalk(pregunta: str) -> bool:
    q = lower_clean(pregunta)
    return any(re.match(p, q) for p in SMALLTALK_PATTERNS)


def es_followup_heuristico(pregunta: str) -> bool:
    q = lower_clean(pregunta)
    patrones = [
        "eso", "ese", "esa", "esas", "esos", "de eso", "de ese", "de esa", "de esos",
        "lo anterior", "respuesta anterior", "ranking anterior", "top anterior", "profundiza",
        "compara con", "y ahora", "y en", "y de", "ese ranking", "esa tabla", "esos resultados",
        "hazme un informe", "haz un informe", "quiero un informe", "reporte de eso", "sobre eso",
    ]
    if any(x in q for x in patrones):
        return True
    return len(q.split()) <= 8 and any(x in q.split() for x in ["eso", "ese", "esa", "y", "tambien", "también"])


def compact_dataframe(df: Optional[pd.DataFrame], max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "Sin datos tabulares previos."
    return df.head(max_rows).fillna("").astype(str).to_string(index=False)


def add_topic(topic: str):
    topic = clean_text(topic)
    if not topic:
        return
    if topic not in st.session_state.discussed_topics:
        st.session_state.discussed_topics.append(topic)
    st.session_state.discussed_topics = st.session_state.discussed_topics[-8:]


def recent_topics_text() -> str:
    topics = st.session_state.get("discussed_topics", [])
    if not topics:
        return "Sin temas conversados aún."
    return " | ".join(topics[-6:])


def summarize_memory() -> str:
    last_df = st.session_state.get("last_df")
    blocks = []
    if st.session_state.get("last_route"):
        blocks.append(f"ULTIMA_RUTA: {st.session_state['last_route']}")
    if st.session_state.get("last_sql"):
        blocks.append(f"ULTIMO_SQL:
{st.session_state['last_sql']}")
    if st.session_state.get("last_answer"):
        blocks.append(f"ULTIMA_RESPUESTA:
{st.session_state['last_answer']}")
    if last_df is not None and not last_df.empty:
        blocks.append("ULTIMOS_DATOS:
" + compact_dataframe(last_df, 15))
    if st.session_state.get("report_mode"):
        blocks.append(f"MODO_INFORME_ACTIVO: True")
    if st.session_state.get("report_goal"):
        blocks.append(f"OBJETIVO_INFORME: {st.session_state['report_goal']}")
    if st.session_state.get("report_items"):
        blocks.append(f"PUNTOS_EN_INFORME: {len(st.session_state['report_items'])}")
    blocks.append(f"TEMAS_CONVERSADOS: {recent_topics_text()}")
    return "

".join(blocks)


def build_chat_history(max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    recent = st.session_state.messages[-max_messages:]
    if not recent:
        return ""
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent])


def report_mode_requested(q: str) -> bool:
    triggers = [
        "al final quiero un informe",
        "quiero crear un informe al final",
        "vamos armando un informe",
        "quiero ir armando un informe",
        "guarda esto para el informe",
        "ten en cuenta esto para el informe",
        "esto va al informe",
    ]
    q2 = lower_clean(q)
    return any(t in q2 for t in triggers)


def add_to_report_requested(q: str) -> bool:
    triggers = [
        "agrega esto al informe",
        "sumalo al informe",
        "súmalo al informe",
        "incluye esto en el informe",
        "esto agregalo al informe",
        "esto agrégalo al informe",
        "guarda esto en el informe",
    ]
    q2 = lower_clean(q)
    return any(t in q2 for t in triggers)


def final_report_requested(q: str) -> bool:
    triggers = [
        "dame el informe",
        "entregame el informe",
        "entrégame el informe",
        "genera el informe final",
        "haz el informe final",
        "crea el informe final",
        "ahora si dame el informe",
        "ya dame el informe",
    ]
    q2 = lower_clean(q)
    return any(t in q2 for t in triggers)


def save_current_point_to_report(question: str, answer: str, sql: str = "", df: Optional[pd.DataFrame] = None, intent: str = ""):
    item = {
        "question": question,
        "answer": answer,
        "sql": sql,
        "intent": intent,
        "data_preview": compact_dataframe(df, 25) if df is not None else "Sin tabla asociada.",
    }
    st.session_state.report_items.append(item)
    st.session_state.report_items = st.session_state.report_items[-20:]


def llm_call(
    model: str,
    contents: str,
    low_latency: bool = True,
    system_instruction: Optional[str] = None,
    higher_reasoning: bool = False,
) -> str:
    config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=2048,
    )
    if low_latency and not higher_reasoning:
        config.thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    if higher_reasoning:
        config.thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM)
    if system_instruction:
        config.system_instruction = system_instruction

    response = genai_client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return obtener_texto_modelo(response).strip()


# ============================================================
# ROUTER INTELIGENTE
# ============================================================
def classify_route(pregunta: str) -> RouteDecision:
    q = lower_clean(pregunta)

    if es_smalltalk(q):
        return RouteDecision(
            route="smalltalk",
            intent="conversacion",
            response_mode="short",
            needs_bq=False,
            followup=False,
            asks_report_scope=False,
            topic="saludo",
            reason="smalltalk heurístico",
        )

    if any(x in q for x in ["hazme un informe", "haz un informe", "quiero un informe", "reporte", "informe ejecutivo"]):
        topics = st.session_state.get("discussed_topics", [])
        if len(topics) >= 2 and not any(x in q for x in ["ventas", "clientes", "productos", "geograf", "medios", "retencion", "retención"]):
            return RouteDecision(
                route="clarify_report_scope",
                intent="informe",
                response_mode="clarify",
                needs_bq=False,
                followup=True,
                asks_report_scope=True,
                topic="informe",
                reason="hay varios temas previos y el alcance del informe no está definido",
            )

    # Heurística rápida primero
    if any(x in q for x in ["comuna", "comunas", "region", "región", "provincia", "provincias", "talca", "maule", "geografia", "geografía"]):
        return RouteDecision("analytics", "geografia", "data", True, es_followup_heuristico(q), False, "geografía", "heurística geografía")
    if any(x in q for x in ["venta", "ventas", "ticket", "ingresos", "facturacion", "facturación", "mes", "año", "anio", "periodo", "período"]):
        return RouteDecision("analytics", "ventas", "data", True, es_followup_heuristico(q), False, "ventas", "heurística ventas")
    if any(x in q for x in ["cliente", "clientes", "ltv", "churn", "riesgo", "recurrente", "nuevo", "segmento"]):
        return RouteDecision("analytics", "clientes", "data", True, es_followup_heuristico(q), False, "clientes", "heurística clientes")
    if any(x in q for x in ["producto", "productos", "sku", "mix", "canasta", "afinidad", "retencion", "retención", "recompra"]):
        return RouteDecision("analytics", "productos", "data", True, es_followup_heuristico(q), False, "productos", "heurística productos")
    if any(x in q for x in ["meta ads", "google ads", "roas", "cac", "cpa", "paid media", "presupuesto", "inversion", "inversión", "campaña", "campana"]):
        return RouteDecision("analytics", "medios", "strategic", True, es_followup_heuristico(q), False, "medios", "heurística medios")

    # Fallback barato con LLM router
    router_prompt = f"""
Clasifica la consulta del usuario.
Devuelve SOLO JSON válido con esta forma exacta:
{{
  "route": "smalltalk|analytics|clarify_report_scope|chat",
  "intent": "ventas|clientes|productos|geografia|medios|auditoria|informe|conversacion|otro",
  "response_mode": "short|data|strategic|clarify",
  "needs_bq": true,
  "followup": false,
  "asks_report_scope": false,
  "topic": "texto corto",
  "reason": "texto corto"
}}

Reglas:
- route=smalltalk si el usuario solo saluda o conversa.
- route=chat si puede responderse sin BigQuery.
- route=analytics si necesita datos reales del warehouse.
- route=clarify_report_scope si pide un informe pero no queda claro de cuál tema conversado.
- needs_bq=false para saludos, agradecimientos, aclaraciones simples o coordinación.
- response_mode=data para respuestas directas con datos duros.
- response_mode=strategic solo cuando el usuario pida lectura estratégica, inversión, recomendación o informe.
- followup=true si depende del contexto anterior.

TEMAS_CONVERSADOS:
{recent_topics_text()}

ULTIMA_MEMORIA:
{summarize_memory()}

PREGUNTA:
{pregunta}
"""
    raw = llm_call(MODEL_ROUTER, router_prompt, low_latency=True)
    try:
        parsed = json.loads(raw)
        return RouteDecision(**parsed)
    except Exception:
        return RouteDecision("analytics", "otro", "data", True, es_followup_heuristico(q), False, "análisis", "fallback seguro")


# ============================================================
# BIGQUERY / ESQUEMAS
# ============================================================
def diagnosticar_tablas() -> Dict[str, bool]:
    status = {}
    for alias, table_fq in MAPA_VERDAD.items():
        try:
            bq_client.get_table(table_fq)
            status[alias] = True
        except Exception:
            status[alias] = False
    return status


def filtrar_tablas_existentes(tablas_objetivo: List[str]) -> List[str]:
    ok = []
    for table_fq in tablas_objetivo:
        try:
            bq_client.get_table(table_fq)
            ok.append(table_fq)
        except Exception:
            continue
    return ok


def obtener_esquema_limitado(table_fq: str) -> List[Tuple[str, str]]:
    table = bq_client.get_table(table_fq)
    schema = [(f.name, f.field_type) for f in table.schema[:MAX_SCHEMA_COLUMNS]]
    return schema


def obtener_esquemas(tablas_objetivo: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for t in tablas_objetivo:
        try:
            out[t] = obtener_esquema_limitado(t)
        except Exception:
            continue
    return out


def format_schemas(esquemas: Dict[str, List[Tuple[str, str]]]) -> str:
    blocks = []
    for tabla, cols in esquemas.items():
        blocks.append(f"- {tabla}: " + ", ".join([f"{c} ({d})" for c, d in cols]))
    return "\n".join(blocks)


def obtener_tablas_prioritarias(pregunta: str, intent: str) -> List[str]:
    q = lower_clean(pregunta)

    if intent == "geografia":
        if any(x in q for x in ["top", "ranking", "más", "mas", "resumen ejecutivo", "ejecutivo"]):
            return [
                MAPA_VERDAD["resumen_geografia"],
                MAPA_VERDAD["geo_pedidos_base"],
                MAPA_VERDAD["geo_no_match"],
            ]
        if any(x in q for x in ["match", "alias", "no coincide", "error geográfico", "error geografico", "depura", "depurar"]):
            return [
                MAPA_VERDAD["geo_no_match"],
                MAPA_VERDAD["geo_maestro_comunas"],
            ]
        return [
            MAPA_VERDAD["geo_pedidos_base"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["geo_maestro_comunas"],
        ]

    if intent == "ventas":
        return [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if intent == "clientes":
        return [
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["dim_clientes"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if intent == "productos":
        return [
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["afinidad_productos"],
            MAPA_VERDAD["fact_pedido_productos"],
        ]

    if intent == "medios":
        return [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["resumen_geografia"],
        ]

    if intent == "auditoria":
        return [
            MAPA_VERDAD["auditoria_datos"],
            MAPA_VERDAD["geo_no_match"],
        ]

    return [
        MAPA_VERDAD["resumen_ventas_periodo"],
        MAPA_VERDAD["perfil_clientes_360"],
        MAPA_VERDAD["resumen_productos_ventas"],
    ]


def validar_sql(query: str) -> Tuple[bool, str]:
    q = limpiar_sql(query)
    if not q:
        return False, "SQL vacío"

    upper = q.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "La consulta debe iniciar con SELECT o WITH"

    blocked = [
        "INSERT ", "UPDATE ", "DELETE ", "MERGE ", "DROP ", "ALTER ",
        "TRUNCATE ", "CREATE ", "CALL ", "EXECUTE IMMEDIATE ", "EXPORT DATA "
    ]
    for word in blocked:
        if word in upper:
            return False, f"Operación no permitida: {word.strip()}"

    if re.search(r"SELECT\s+\*", q, flags=re.IGNORECASE):
        return False, "SELECT * no permitido"

    tablas = set(re.findall(r"`([^`]+)`", q))
    if not tablas:
        return False, "Debes usar tablas oficiales con backticks"

    invalid = [t for t in tablas if t not in ALLOWED_TABLES]
    if invalid:
        return False, f"Tablas fuera del mapa: {invalid}"

    return True, ""


def ejecutar_query(query: str) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        maximum_bytes_billed=MAX_BYTES_BILLED,
    )
    return bq_client.query(query, job_config=job_config).result().to_dataframe()


# ============================================================
# SQL GENERATION
# ============================================================
def build_sql_prompt(pregunta: str, route: RouteDecision, tablas: List[str], esquemas_texto: str, error_previo: str = "") -> str:
    tablas_txt = "\n".join([f"- {t}" for t in tablas])
    return f"""
Eres NobleBotAI SQL Engine. Devuelve SOLO SQL puro para BigQuery.

MAPA_OFICIAL:
{chr(10).join([f'- {v}' for v in MAPA_VERDAD.values()])}

TABLAS_PRIORITARIAS:
{tablas_txt}

ESQUEMAS_REALES:
{esquemas_texto}

MEMORIA_RECIENTE:
{build_chat_history()}

CONTEXTO_ESTRUCTURADO:
{summarize_memory()}

PREGUNTA:
{pregunta}

INTENT:
{route.intent}

ERROR_PREVIO:
{error_previo}

REGLAS DE NEGOCIO:
- Prioriza tablas AI; baja a CORE solo si la granularidad lo exige.
- Responde con datos duros y evita sobrecomplicar.
- Si la pregunta es geográfica, usa el flujo nuevo:
  - detalle geográfico -> `geo_pedidos_base`
  - resumen ejecutivo -> `resumen_geografia`
  - depuración de match -> `geo_no_match`
- Si piden Talca, Maule, comunas, provincias o regiones, usa la lógica geográfica oficial, no texto libre en core como ruta principal.
- Si es follow-up, usa la memoria previa para entender a qué se refiere.
- Si la pregunta puede resolverse con una tabla agregada, no escales a una tabla pesada.
- Si piden top/ranking, ordena y limita.
- Si piden detalle abierto, limita razonablemente.

REGLAS TÉCNICAS:
1. Solo SQL.
2. Empieza con SELECT o WITH.
3. Usa backticks en tablas.
4. No inventes columnas.
5. No uses SELECT *.
6. No uses DDL ni DML.
7. Evita scans innecesarios.
8. Si el usuario pide un dato puntual, no generes una consulta gigantesca.
"""


def generate_sql_with_retries(pregunta: str, route: RouteDecision) -> Tuple[str, pd.DataFrame]:
    cache_key = f"{route.intent}::{pregunta.strip().lower()}::{st.session_state.get('last_sql','')}"
    if ENABLE_SQL_CACHE and cache_key in st.session_state.sql_cache:
        sql_cached = st.session_state.sql_cache[cache_key]
        return sql_cached, ejecutar_query(sql_cached)

    tablas = filtrar_tablas_existentes(obtener_tablas_prioritarias(pregunta, route.intent))
    if not tablas:
        raise ValueError("No encontré tablas disponibles del mapa oficial.")

    esquemas = obtener_esquemas(tablas)
    if not esquemas:
        raise ValueError("No pude leer esquemas de las tablas disponibles.")

    prompt_base = format_schemas(esquemas)
    error_previo = ""
    ultimo_error = ""

    for _ in range(MAX_SQL_RETRIES):
        sql = limpiar_sql(llm_call(
            MODEL_SQL,
            build_sql_prompt(pregunta, route, tablas, prompt_base, error_previo),
            low_latency=False,
            higher_reasoning=True,
        ))

        ok, motivo = validar_sql(sql)
        if not ok:
            ultimo_error = motivo
            error_previo = motivo
            continue

        try:
            df = ejecutar_query(sql)
            if ENABLE_SQL_CACHE:
                st.session_state.sql_cache[cache_key] = sql
            return sql, df
        except Exception as e:
            ultimo_error = str(e)
            error_previo = str(e)

    raise ValueError(f"No pude generar SQL válido. Último error: {ultimo_error}")


# ============================================================
# RESPONSE LAYER
# ============================================================
def suggested_next_steps(route: RouteDecision, pregunta: str, has_data: bool) -> List[str]:
    if not has_data:
        return [
            "Puedo reformular la búsqueda con otro período o nivel geográfico.",
            "Puedo revisar si el problema está en match geográfico o en un filtro demasiado estrecho.",
        ]
    if route.intent == "ventas":
        return [
            "También puedo comparar este resultado contra el período anterior.",
            "También puedo cruzarlo con productos líderes o ticket promedio.",
        ]
    if route.intent == "clientes":
        return [
            "También puedo segmentar esos clientes por recurrencia, riesgo o valor.",
            "También puedo convertir esto en un informe ejecutivo de clientes.",
        ]
    if route.intent == "productos":
        return [
            "También puedo mostrar afinidad entre productos o patrones de recompra.",
            "También puedo bajar esto a SKU, mix o retención.",
        ]
    if route.intent == "geografia":
        return [
            "También puedo hacer el ranking territorial del mismo tema.",
            "También puedo revisar si hay comunas o pedidos que no hicieron match.",
        ]
    if route.intent == "medios":
        return [
            "También puedo convertir esto en una propuesta de inversión en marketing.",
            "También puedo hacer una lectura ejecutiva para performance, CRM o retención.",
        ]
    return [
        "También puedo profundizar en el mismo tema con otra dimensión.",
        "También puedo convertir esta conversación en un informe ejecutivo cuando tú quieras.",
    ]


def build_answer_prompt(pregunta: str, route: RouteDecision, sql: str, df: pd.DataFrame) -> str:
    data_text = compact_dataframe(df, 60)
    return f"""
Eres NobleBotAI, un analista conversacional para ecommerce, marketing, medios y agencia.

OBJETIVO:
Responder entendiendo exactamente lo que pidió el usuario.
No conviertas cada respuesta en un informe.
Primero responde con precisión y datos duros.
Solo agrega interpretación o propuesta si aporta de verdad o si el usuario la pidió.

ESTILO:
- español
- claro
- ejecutivo
- rápido
- sin relleno
- sin sonar flojo
- sin inventar datos

MODO_DE_RESPUESTA: {route.response_mode}
INTENT: {route.intent}
PREGUNTA: {pregunta}
MEMORIA: {summarize_memory()}
SQL: {sql}
FILAS: {len(df)}
DATOS: {data_text}

REGLAS:
1. No digas siempre diagnóstico + plan de acción.
2. Si el usuario pidió un dato puntual, responde puntual.
3. Si pidió análisis, agrega lectura breve y útil.
4. Si pidió informe, ahí sí responde amplio, completo y estructurado.
5. Usa CLP cuando haya montos.
6. Si no hay evidencia suficiente, dilo.
7. Cierra con 2 sugerencias breves y útiles, estilo asistente inteligente.
"""


def render_answer(pregunta: str, route: RouteDecision, sql: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        if route.intent == "geografia":
            base = (
                "No encontré filas que calcen exactamente con esa búsqueda en las tablas oficiales. "
                "Eso normalmente significa que el filtro está muy estrecho, el período no tiene registros o el match geográfico no coincide."
            )
        elif route.intent == "ventas":
            base = (
                "No encontré filas que calcen exactamente con esa búsqueda en las tablas oficiales. "
                "Eso normalmente significa que el período consultado no tiene registros, la métrica pedida no está en esa tabla o la consulta quedó demasiado restringida."
            )
        else:
            base = (
                "No encontré filas que calcen exactamente con esa búsqueda en las tablas oficiales. "
                "Eso normalmente significa que el filtro quedó demasiado estrecho o faltó precisar mejor la consulta."
            )
        sugg = suggested_next_steps(route, pregunta, has_data=False)
        return base + "

" + "
".join([f"- {s}" for s in sugg])

    raw = llm_call(MODEL_CHAT, build_answer_prompt(pregunta, route, sql, df), low_latency=True)
    raw = raw.strip() or "Consulté las tablas oficiales y obtuve el resultado, pero la redacción final salió vacía."
    if st.session_state.get("report_mode"):
        raw += "

Puedo guardar este hallazgo para el informe final si me dices: agrega esto al informe."
    return raw


def build_report_prompt(user_request: str, scope: str) -> str:
    items = st.session_state.get("report_items", [])
    compiled_points = []
    for i, item in enumerate(items, start=1):
        compiled_points.append(
            f"PUNTO {i}
"
            f"PREGUNTA: {item.get('question', '')}
"
            f"INTENT: {item.get('intent', '')}
"
            f"RESPUESTA: {item.get('answer', '')}
"
            f"SQL: {item.get('sql', '')}
"
            f"DATOS: {item.get('data_preview', '')}"
        )

    return f"""
Eres NobleBotAI.
Genera un informe final EXTENSO, profundo y bien conectado para una agencia / marca ecommerce.

ALCANCE DEL INFORME: {scope}
PEDIDO ORIGINAL: {user_request}
OBJETIVO_GENERAL: {st.session_state.get('report_goal', '')}

MEMORIA DEL CHAT:
{summarize_memory()}

PUNTOS ACUMULADOS PARA EL INFORME:
{'

'.join(compiled_points) if compiled_points else 'No hay puntos acumulados.'}

REGLAS:
- Usa solo lo conversado y lo consultado en esta sesión.
- Consolida todos los puntos guardados en un solo documento coherente.
- No inventes datos faltantes.
- Cruza hallazgos entre secciones, no repitas como bloques aislados.
- Debe ser bastante más extenso que una respuesta normal.
- Extrae conclusiones, patrones, comparaciones y tensiones de negocio.
- Si hay años o períodos, compáralos con claridad.
- Si hay geografía, interpreta territorios, concentración y cambios.
- Si hay clientes/productos/ventas, conecta impacto comercial.
- Adáptalo a marketing, medios, ecommerce, CRM y agencia si aplica.
- Debe sentirse sólido, no genérico.

ESTRUCTURA OBLIGATORIA:
1. Resumen ejecutivo
2. Contexto y alcance del análisis
3. Hallazgos clave consolidados
4. Análisis detallado por tema
5. Comparaciones relevantes y qué cambió
6. Conclusiones estratégicas
7. Riesgos y oportunidades
8. Recomendaciones priorizadas
9. Próximos análisis sugeridos
"""


def handle_report_request(prompt: str) -> str:
    topics = st.session_state.get("discussed_topics", [])
    q = lower_clean(prompt)

    if report_mode_requested(prompt):
        st.session_state.report_mode = True
        st.session_state.report_goal = prompt
        return (
            "Perfecto. Desde ahora iré armando el informe en segundo plano durante esta conversación. "
            "Cuando una respuesta quieras guardarla, puedes decirme 'agrega esto al informe'. "
            "Y cuando quieras el documento final, pídeme 'dame el informe final'."
        )

    if add_to_report_requested(prompt):
        if st.session_state.get("last_answer"):
            save_current_point_to_report(
                question=st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else prompt,
                answer=st.session_state.get("last_answer", ""),
                sql=st.session_state.get("last_sql", ""),
                df=st.session_state.get("last_df", None),
                intent=st.session_state.get("last_route", ""),
            )
            return f"Listo. Ya agregué ese punto al informe. Puntos acumulados: {len(st.session_state.get('report_items', []))}."
        return "Todavía no tengo una respuesta previa sólida para agregar al informe."

    if final_report_requested(prompt):
        if not st.session_state.get("report_items"):
            return "Aún no tengo puntos guardados para construir el informe final. Primero activa el modo informe o pídeme agregar respuestas específicas."
        scope = st.session_state.get("report_goal", prompt)
        report = llm_call(MODEL_REPORT, build_report_prompt(prompt, scope), low_latency=False, higher_reasoning=True)
        add_topic("informe")
        st.session_state.pending_report = False
        return report

    if len(topics) >= 2 and not any(x in q for x in ["ventas", "clientes", "productos", "geograf", "medios", "retencion", "retención"]):
        options = "
".join([f"- {t}" for t in topics[-6:]])
        st.session_state.pending_report = True
        return (
            "Puedo hacerlo. Antes de generar el informe, dime sobre cuál de estos temas conversados lo quieres:

"
            f"{options}

"
            "También puedes pedírmelo cruzado, por ejemplo: ventas + productos, geografía + clientes, o medios + ventas."
        )

    scope = prompt
    report = llm_call(MODEL_REPORT, build_report_prompt(prompt, scope), low_latency=False, higher_reasoning=True)
    add_topic("informe")
    st.session_state.pending_report = False
    return report


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="NobleBotAI 🐷🐽", layout="wide")
st.title("NobleBotAI 🐷🐽")
st.caption("Inteligencia comercial, performance y marketing 360 para agencia")

with st.sidebar:
    st.header("Configuración")
    st.markdown(f"**Proyecto:** `{PROJECT_ID}`")
    st.markdown(f"**Location:** `{LOCATION}`")
    st.markdown(f"**Marca base:** `{BRAND_ID}`")
    st.markdown(f"**Modelo SQL:** `{MODEL_SQL}`")
    st.markdown(f"**Modelo Router:** `{MODEL_ROUTER}`")
    st.markdown(f"**Modelo Chat:** `{MODEL_CHAT}`")
    st.markdown(f"**Modelo Report:** `{MODEL_REPORT}`")

    st.subheader("Mapa de Verdad")
    for alias, fq_table in MAPA_VERDAD.items():
        st.markdown(f"- **{alias}** → `{fq_table}`")

    st.subheader("Estado tablas oficiales")
    for alias, existe in diagnosticar_tablas().items():
        st.markdown(f"{'✅' if existe else '❌'} **{alias}**")

    if st.button("Limpiar memoria"):
        for key in [
            "messages", "chat_memory", "last_sql", "last_df", "last_answer",
            "last_route", "discussed_topics", "pending_report", "report_mode",
            "report_items", "report_goal", "sql_cache"
        ]:
            st.session_state.pop(key, None)
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Pregúntame por clientes, ventas, productos, geografía, medios o informes...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            route = classify_route(prompt)
            st.session_state.last_route = route.intent
            add_topic(route.topic)

            if route.route == "smalltalk":
                answer = GREETING_REPLY
                st.markdown(answer)

            elif route.route == "clarify_report_scope":
                answer = handle_report_request(prompt)
                st.markdown(answer)

            elif route.route == "chat" and not route.needs_bq:
                answer = llm_call(
                    MODEL_CHAT,
                    f"Responde de forma breve y útil. Mantén contexto del chat.\n\nMEMORIA:\n{summarize_memory()}\n\nPREGUNTA:\n{prompt}",
                    low_latency=True,
                )
                st.markdown(answer)

            elif route.intent == "informe" or st.session_state.get("pending_report") or report_mode_requested(prompt) or add_to_report_requested(prompt) or final_report_requested(prompt):
                answer = handle_report_request(prompt)
                st.markdown(answer)

            else:
                with st.spinner("Consultando tablas oficiales..."):
                    sql, df = generate_sql_with_retries(prompt, route)
                    answer = render_answer(prompt, route, sql, df)
                    st.markdown(answer)

                    st.session_state.last_sql = sql
                    st.session_state.last_df = df.copy()

                    if add_to_report_requested(prompt):
                        save_current_point_to_report(
                            question=prompt,
                            answer=answer,
                            sql=sql,
                            df=df,
                            intent=route.intent,
                        )

                    with st.expander("Ver SQL usado"):
                        st.code(sql, language="sql")

                    with st.expander("Ver datos tabulares"):
                        st.dataframe(df.head(MAX_ROWS_RESULT), use_container_width=True)

            st.session_state.last_answer = answer
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_message = (
                "Hubo un problema técnico al construir o ejecutar la respuesta. "
                "La lógica del agente sigue intacta, pero falló la ruta de ejecución. "
                "Prueba con una pregunta más específica o revisa si el modelo y la ubicación de Vertex AI están configurados correctamente."
            )
            st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.expander("Detalle técnico"):
                st.code(str(e))

