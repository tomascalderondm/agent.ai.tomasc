import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
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

MODEL_SQL = st.secrets.get("MODEL_SQL", "gemini-3.1-pro-preview")
MODEL_ROUTER = st.secrets.get("MODEL_ROUTER", "gemini-3.1-flash-lite-preview")
MODEL_CHAT = st.secrets.get("MODEL_CHAT", "gemini-3.1-flash-lite-preview")
MODEL_REPORT = st.secrets.get("MODEL_REPORT", "gemini-3.1-pro-preview")

MAX_ROWS_RESULT = int(st.secrets.get("MAX_ROWS_RESULT", 200))
MAX_HISTORY_MESSAGES = int(st.secrets.get("MAX_HISTORY_MESSAGES", 8))
MAX_SQL_RETRIES = int(st.secrets.get("MAX_SQL_RETRIES", 2))
MAX_BYTES_BILLED = int(st.secrets.get("MAX_BYTES_BILLED", 5 * 1024 * 1024 * 1024))
MAX_SCHEMA_COLUMNS = int(st.secrets.get("MAX_SCHEMA_COLUMNS", 80))


def to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


ENABLE_SQL_CACHE = to_bool(st.secrets.get("ENABLE_SQL_CACHE", True), default=True)

# Búsqueda externa opcional
ENABLE_EXTERNAL_RESEARCH = to_bool(st.secrets.get("ENABLE_EXTERNAL_RESEARCH", False), default=False)
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
EXTERNAL_RESEARCH_MAX_RESULTS = int(st.secrets.get("EXTERNAL_RESEARCH_MAX_RESULTS", 8))


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
# CONTEXTO DE ENTIDADES CONOCIDAS
# ============================================================
KNOWN_ENTITIES = {
    "campo noble": {
        "type": "marca",
        "description": "Campo Noble es una marca chilena de productos de cerdo vinculada a Coexca.",
        "aliases": ["camponoble", "campo noble", "coexca campo noble"],
    },
    "coexca": {
        "type": "empresa",
        "description": "Coexca S.A. es una empresa chilena de carne de cerdo con integración vertical y presencia exportadora.",
        "aliases": ["coexca", "coexca s.a.", "coexca sa"],
    },
}


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
if "last_intent" not in st.session_state:
    st.session_state.last_intent = ""
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
if "last_research_context" not in st.session_state:
    st.session_state.last_research_context = ""


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
    "medios, informes e investigación externa si me la pides."
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
    return len(q.split()) <= 8 and any(x in q.split() for x in ["eso", "ese", "esa", "tambien", "también"])


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
    if st.session_state.get("last_intent"):
        blocks.append(f"ULTIMO_INTENT: {st.session_state['last_intent']}")
    if st.session_state.get("last_sql"):
        blocks.append(f"ULTIMO_SQL: {st.session_state['last_sql']}")
    if st.session_state.get("last_answer"):
        blocks.append(f"ULTIMA_RESPUESTA: {st.session_state['last_answer']}")
    if last_df is not None and not last_df.empty:
        blocks.append("ULTIMOS_DATOS: " + compact_dataframe(last_df, 15))
    if st.session_state.get("report_mode"):
        blocks.append("MODO_INFORME_ACTIVO: True")
    if st.session_state.get("report_goal"):
        blocks.append(f"OBJETIVO_INFORME: {st.session_state['report_goal']}")
    if st.session_state.get("report_items"):
        blocks.append(f"PUNTOS_EN_INFORME: {len(st.session_state['report_items'])}")
    if st.session_state.get("last_research_context"):
        blocks.append("ULTIMO_CONTEXTO_EXTERN0: disponible")
    blocks.append(f"TEMAS_CONVERSADOS: {recent_topics_text()}")
    return "  ".join(blocks)


def build_chat_history(max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    recent = st.session_state.messages[-max_messages:]
    if not recent:
        return ""
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent])


def known_entities_text() -> str:
    bloques = []
    for nombre, info in KNOWN_ENTITIES.items():
        bloques.append(f"- {nombre}: {info.get('description', '')}")
    return "\n".join(bloques)


def report_mode_requested(q: str) -> bool:
    q2 = lower_clean(q)

    triggers = [
        "al final quiero un informe",
        "quiero crear un informe al final",
        "vamos armando un informe",
        "quiero ir armando un informe",
        "ten en cuenta esto para el informe",
        "esto va al informe",
        "desde ahora arma un informe",
        "quiero construir un informe durante la conversación",
        "anda guardando para el informe",
    ]

    if any(t in q2 for t in triggers):
        return True

    return bool(
        re.search(r"\b(informe|reporte)\b.*\b(final|después|despues|al final)\b", q2)
        or re.search(r"\b(armando|construyendo|guardando)\b.*\b(informe|reporte)\b", q2)
    )


def add_to_report_requested(q: str) -> bool:
    q2 = lower_clean(q)

    triggers_exactos = [
        "agrega esto al informe",
        "sumalo al informe",
        "súmalo al informe",
        "incluye esto en el informe",
        "guarda esto en el informe",
        "mete esto en el informe",
        "pon esto en el informe",
        "incorpora esto al informe",
        "anota esto para el informe",
    ]

    if any(t in q2 for t in triggers_exactos):
        return True

    patrones_flexibles = [
        r"\bagrega\b.*\binforme\b",
        r"\bincluye\b.*\binforme\b",
        r"\bguarda\b.*\binforme\b",
        r"\bincorpora\b.*\binforme\b",
        r"\bmete\b.*\binforme\b",
        r"\bpon\b.*\binforme\b",
        r"\banota\b.*\binforme\b",
        r"\bsuma\b.*\binforme\b",
        r"\besto\b.*\binforme\b",
    ]

    return any(re.search(p, q2) for p in patrones_flexibles)


def final_report_requested(q: str) -> bool:
    q2 = lower_clean(q)

    triggers = [
        "dame el informe",
        "entregame el informe",
        "entrégame el informe",
        "genera el informe final",
        "haz el informe final",
        "crea el informe final",
        "ahora si dame el informe",
        "ya dame el informe",
        "muéstrame el informe final",
        "quiero ver el informe final",
        "redacta el informe final",
        "arma el informe final",
    ]

    if any(t in q2 for t in triggers):
        return True

    return bool(
        re.search(r"\b(informe final|reporte final)\b", q2)
        or re.search(r"\b(dame|haz|crea|genera|arma|entrega|muestra|redacta)\b.*\b(informe|reporte)\b", q2)
    )


def research_mode_requested(q: str) -> bool:
    q2 = lower_clean(q)
    return any(x in q2 for x in [
        "investiga sobre",
        "busca en google",
        "busca noticias",
        "noticias relevantes",
        "último mes",
        "ultimo mes",
        "qué está pasando con",
        "que está pasando con",
        "que esta pasando con",
        "quien es",
        "quién es",
        "empresa",
        "competencia",
        "benchmark",
        "como lo hace",
        "como lo haría",
        "como lo haria",
        "estudio de mercado",
        "estudio de comportamiento de mercado",
        "consultora",
        "al estilo de",
    ])


def save_current_point_to_report(question: str, answer: str, sql: str = "", df: Optional[pd.DataFrame] = None, intent: str = ""):
    item = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sql": sql,
        "intent": intent,
        "data_preview": compact_dataframe(df, 25) if df is not None else "Sin tabla asociada.",
    }
    st.session_state.report_items.append(item)
    st.session_state.report_items = st.session_state.report_items[-30:]


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
        max_output_tokens=4096,
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

    if research_mode_requested(q):
        return RouteDecision(
            route="research",
            intent="investigacion",
            response_mode="strategic",
            needs_bq=False,
            followup=es_followup_heuristico(q),
            asks_report_scope=False,
            topic="investigación externa",
            reason="heurística investigación web/noticias/benchmark",
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

    router_prompt = f"""
Clasifica la consulta del usuario. Devuelve SOLO JSON válido con esta forma exacta:
{{
  "route": "smalltalk|analytics|clarify_report_scope|chat|research",
  "intent": "ventas|clientes|productos|geografia|medios|auditoria|informe|conversacion|investigacion|benchmark|otro",
  "response_mode": "short|data|strategic|clarify",
  "needs_bq": true,
  "followup": false,
  "asks_report_scope": false,
  "topic": "texto corto",
  "reason": "texto corto"
}}

Reglas:
- route=smalltalk si el usuario solo saluda o conversa.
- route=chat si puede responderse sin BigQuery ni búsqueda externa.
- route=analytics si necesita datos reales del warehouse.
- route=research si pide investigar empresas, noticias, benchmark, estilo de consultora o búsqueda web.
- route=clarify_report_scope si pide un informe pero no queda claro de cuál tema conversado.
- needs_bq=false para saludos, agradecimientos, aclaraciones simples, benchmark, investigación externa o coordinación.
- response_mode=data para respuestas directas con datos duros.
- response_mode=strategic cuando el usuario pida lectura estratégica, investigación, benchmarking, inversión, recomendación o informe.
- followup=true si depende del contexto anterior.

TEMAS_CONVERSADOS: {recent_topics_text()}
ULTIMA_MEMORIA: {summarize_memory()}
ENTIDADES_CONOCIDAS:
{known_entities_text()}

PREGUNTA: {pregunta}
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
@st.cache_data(ttl=300)
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

ENTIDADES_CONOCIDAS:
{known_entities_text()}

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
    cache_key = f"{route.intent}::{pregunta.strip().lower()}"

    if ENABLE_SQL_CACHE and cache_key in st.session_state.sql_cache:
        sql_cached = st.session_state.sql_cache[cache_key]
        try:
            return sql_cached, ejecutar_query(sql_cached)
        except Exception:
            st.session_state.sql_cache.pop(cache_key, None)

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
# INVESTIGACIÓN EXTERNA
# ============================================================
def resolve_known_entity_context(query: str) -> str:
    q = lower_clean(query)
    matches = []
    for _, info in KNOWN_ENTITIES.items():
        aliases = info.get("aliases", [])
        if any(alias in q for alias in aliases):
            matches.append(info.get("description", ""))
    return " | ".join(matches) if matches else ""


def external_search_serpapi(query: str, num_results: int = EXTERNAL_RESEARCH_MAX_RESULTS) -> List[Dict]:
    if not SERPAPI_KEY:
        return []

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "hl": "es",
                "gl": "cl",
                "num": num_results,
                "api_key": SERPAPI_KEY,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", "Google"),
                "date": item.get("date", ""),
            })
        return results
    except Exception:
        return []


def external_news_newsapi(query: str, days: int = 30, page_size: int = EXTERNAL_RESEARCH_MAX_RESULTS) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "es",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("articles", [])[:page_size]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "date": item.get("publishedAt", ""),
            })
        return results
    except Exception:
        return []


def build_external_queries(prompt: str) -> Tuple[List[str], List[str]]:
    q = clean_text(prompt)
    q_lower = lower_clean(prompt)

    entity_context = resolve_known_entity_context(prompt)
    news_queries = []
    web_queries = []

    if entity_context:
        web_queries.append(q)

    if "último mes" in q_lower or "ultimo mes" in q_lower or "noticias" in q_lower:
        news_queries.append(q)

    if "campo noble" in q_lower and ("noticias" in q_lower or "último mes" in q_lower or "ultimo mes" in q_lower):
        news_queries.append("Campo Noble Coexca noticias último mes")
        web_queries.append("Campo Noble Coexca empresa marca cerdo Chile")

    if "como lo hace" in q_lower or "al estilo de" in q_lower or "consultora" in q_lower:
        web_queries.append(q)
        web_queries.append(q + " metodología informe")
        web_queries.append(q + " estudio comportamiento mercado")

    if not news_queries and not web_queries:
        web_queries.append(q)

    return news_queries[:4], web_queries[:4]


def format_external_results(news_results: List[Dict], web_results: List[Dict], entity_context: str) -> str:
    bloques = []

    if entity_context:
        bloques.append(f"CONTEXTO_ENTIDAD_CONOCIDA: {entity_context}")

    if news_results:
        bloques.append("NOTICIAS_RELEVANTES:")
        for i, item in enumerate(news_results, start=1):
            bloques.append(
                f"[NEWS {i}] TITULO: {item.get('title','')} | FUENTE: {item.get('source','')} | FECHA: {item.get('date','')} | LINK: {item.get('link','')} | RESUMEN: {item.get('snippet','')}"
            )

    if web_results:
        bloques.append("RESULTADOS_WEB:")
        for i, item in enumerate(web_results, start=1):
            bloques.append(
                f"[WEB {i}] TITULO: {item.get('title','')} | FUENTE: {item.get('source','')} | FECHA: {item.get('date','')} | LINK: {item.get('link','')} | RESUMEN: {item.get('snippet','')}"
            )

    if not bloques:
        bloques.append("SIN_RESULTADOS_EXTERNOS_CONFIGURADOS")

    return "\n".join(bloques)


def run_external_research(prompt: str) -> str:
    entity_context = resolve_known_entity_context(prompt)

    if not ENABLE_EXTERNAL_RESEARCH:
        return (
            "INVESTIGACION_EXTERNA_NO_CONFIGURADA\n"
            f"CONTEXTO_CONOCIDO: {entity_context if entity_context else 'Sin contexto conocido adicional.'}\n"
            "Para activar investigación externa real, configura ENABLE_EXTERNAL_RESEARCH y una fuente como SERPAPI_KEY o NEWSAPI_KEY."
        )

    news_queries, web_queries = build_external_queries(prompt)

    all_news_results: List[Dict] = []
    all_web_results: List[Dict] = []

    for q in news_queries:
        all_news_results.extend(external_news_newsapi(q))

    for q in web_queries:
        all_web_results.extend(external_search_serpapi(q))

    unique_news = []
    seen_news = set()
    for item in all_news_results:
        key = (item.get("title", ""), item.get("link", ""))
        if key not in seen_news:
            seen_news.add(key)
            unique_news.append(item)

    unique_web = []
    seen_web = set()
    for item in all_web_results:
        key = (item.get("title", ""), item.get("link", ""))
        if key not in seen_web:
            seen_web.add(key)
            unique_web.append(item)

    context = format_external_results(
        unique_news[:EXTERNAL_RESEARCH_MAX_RESULTS],
        unique_web[:EXTERNAL_RESEARCH_MAX_RESULTS],
        entity_context,
    )
    st.session_state.last_research_context = context
    return context


def build_research_prompt(pregunta: str, external_context: str) -> str:
    return f"""
Eres NobleBotAI en modo investigación externa.

PREGUNTA:
{pregunta}

CONTEXTO EXTERNO:
{external_context}

MEMORIA:
{summarize_memory()}

ENTIDADES_CONOCIDAS:
{known_entities_text()}

REGLAS:
- Usa primero la evidencia entregada en el contexto externo.
- Si el contexto externo no trae resultados reales, dilo explícitamente.
- Si el usuario pidió noticias del último mes, prioriza recencia.
- Si el usuario pidió empresa, resume quién es, qué hace, posicionamiento, señales recientes y contexto competitivo si existe.
- Si el usuario pidió benchmark o "hazme un informe como lo hace tal empresa", identifica el estilo o metodología observable y adapta la estructura sin copiar textos.
- Si el usuario pidió Campo Noble, entiende que es una marca ligada a Coexca.
- No inventes.
- Estructura la respuesta con:
  1. Resumen
  2. Hallazgos clave
  3. Noticias o señales recientes
  4. Lectura estratégica
  5. Riesgos y oportunidades
  6. Próximos pasos sugeridos
"""


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
    if route.intent == "investigacion":
        return [
            "También puedo convertir esta investigación en un informe ejecutivo.",
            "También puedo cruzar esto con tus datos internos si me pides una lectura combinada.",
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
        return base + "  " + " ".join([f"- {s}" for s in sugg])

    raw = llm_call(MODEL_CHAT, build_answer_prompt(pregunta, route, sql, df), low_latency=True)
    raw = raw.strip() or "Consulté las tablas oficiales y obtuve el resultado, pero la redacción final salió vacía."
    if st.session_state.get("report_mode"):
        raw += "  Este hallazgo quedó disponible para el informe final."
    return raw


# ============================================================
# CONSOLIDACIÓN DE INFORME
# ============================================================
def build_report_consolidation_prompt(items: List[Dict], user_request: str, scope: str) -> str:
    bloques = []
    for i, item in enumerate(items, start=1):
        bloques.append(
            f"""
PUNTO {i}
TIMESTAMP: {item.get('timestamp', '')}
INTENT: {item.get('intent', '')}
QUESTION: {item.get('question', '')}
ANSWER: {item.get('answer', '')}
SQL: {item.get('sql', '')}
DATA_PREVIEW: {item.get('data_preview', '')}
""".strip()
        )

    return f"""
Eres NobleBotAI. Debes consolidar puntos analíticos antes de redactar un informe final.

PEDIDO_USUARIO: {user_request}
ALCANCE: {scope}

PUNTOS_ORIGINALES:
{'  '.join(bloques) if bloques else 'No hay puntos.'}

TAREA:
1. Reordena los puntos cronológicamente usando TIMESTAMP.
2. Detecta duplicados exactos o semánticos.
3. Fusiona hallazgos repetidos o muy similares en un solo hallazgo consolidado.
4. Prioriza los hallazgos más fuertes según:
   - evidencia numérica,
   - impacto comercial,
   - recurrencia,
   - claridad estratégica.
5. Separa:
   - hallazgos principales,
   - hallazgos secundarios,
   - conclusiones,
   - riesgos,
   - oportunidades.
6. No inventes datos.
7. Si dos puntos se contradicen, indícalo.
8. Devuelve SOLO JSON válido con esta estructura:

{{
  "ordered_points": [
    {{
      "timestamp": "iso",
      "question": "texto",
      "intent": "texto",
      "summary": "texto corto"
    }}
  ],
  "merged_findings": [
    {{
      "title": "texto corto",
      "priority": "alta|media|baja",
      "strength_score": 1,
      "evidence": "texto",
      "merged_from": [1,2],
      "business_impact": "texto",
      "strategic_read": "texto"
    }}
  ],
  "conclusions": ["texto", "texto"],
  "risks": ["texto", "texto"],
  "opportunities": ["texto", "texto"]
}}
"""


def consolidate_report_items(user_request: str, scope: str) -> Dict:
    items = st.session_state.get("report_items", [])
    if not items:
        return {}

    items_sorted = sorted(items, key=lambda x: x.get("timestamp", ""))

    raw = llm_call(
        MODEL_REPORT,
        build_report_consolidation_prompt(items_sorted, user_request, scope),
        low_latency=False,
        higher_reasoning=True,
    )

    try:
        return json.loads(raw)
    except Exception:
        return {
            "ordered_points": [],
            "merged_findings": [],
            "conclusions": [],
            "risks": [],
            "opportunities": [],
        }


def build_report_prompt(user_request: str, scope: str) -> str:
    consolidated = consolidate_report_items(user_request, scope)

    return f"""
Eres NobleBotAI. Genera un informe final EXTENSO, sólido y ejecutivo.

ALCANCE DEL INFORME: {scope}
PEDIDO ORIGINAL: {user_request}
OBJETIVO_GENERAL: {st.session_state.get('report_goal', '')}

MEMORIA DEL CHAT: {summarize_memory()}

ENTIDADES_CONOCIDAS:
{known_entities_text()}

CONSOLIDACION_PREVIA:
{json.dumps(consolidated, ensure_ascii=False, indent=2)}

REGLAS:
- Usa primero la evidencia interna de esta sesión.
- Si el usuario pidió explícitamente investigación externa, integra también señales externas verificadas si existen.
- Si el usuario pidió benchmark de consultora o empresa, adapta el enfoque, el tono y la estructura al estilo observado, sin copiar textos.
- Usa la consolidación previa como base principal.
- Respeta el orden cronológico de evolución del análisis.
- No repitas hallazgos duplicados.
- Fusiona hallazgos repetidos.
- Prioriza hallazgos de mayor fuerza e impacto.
- Debes inferir conclusiones, riesgos y oportunidades SOLO cuando haya base suficiente en los datos.
- Si la evidencia es insuficiente, dilo.
- Cruza ventas, clientes, productos, geografía, medios y negocio cuando corresponda.
- No hagas un texto plano interminable: estructura, jerarquiza y sintetiza.
- Debe sentirse como un informe serio de análisis de mercado/comercial, no como una respuesta casual.

ESTRUCTURA OBLIGATORIA:
1. Resumen ejecutivo
2. Evolución cronológica del análisis
3. Hallazgos clave priorizados
4. Hallazgos consolidados por tema
5. Conclusiones basadas en evidencia
6. Riesgos
7. Oportunidades
8. Recomendaciones priorizadas
9. Próximos análisis sugeridos
"""


def handle_report_request(prompt: str) -> str:
    topics = st.session_state.get("discussed_topics", [])
    q = lower_clean(prompt)

    if report_mode_requested(prompt):
        st.session_state.report_mode = True
        st.session_state.report_goal = prompt
        st.session_state.pending_report = False
        return (
            "Perfecto. Activé el modo informe. "
            "Desde ahora iré guardando automáticamente los hallazgos analíticos de esta conversación para el informe final. "
            "Si quieres, igual puedes reforzarlo con frases como 'agrega esto', 'incluye esto en el informe' o 'guárdalo para el reporte'. "
            "Cuando quieras el documento final, pídeme el informe final."
        )

    if add_to_report_requested(prompt):
        if st.session_state.get("last_answer"):
            save_current_point_to_report(
                question=st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else prompt,
                answer=st.session_state.get("last_answer", ""),
                sql=st.session_state.get("last_sql", ""),
                df=st.session_state.get("last_df", None),
                intent=st.session_state.get("last_intent", ""),
            )
            st.session_state.pending_report = False
            return f"Listo. Ya agregué ese punto al informe. Puntos acumulados: {len(st.session_state.get('report_items', []))}."
        return "Todavía no tengo una respuesta previa sólida para agregar al informe."

    if final_report_requested(prompt):
        if not st.session_state.get("report_items"):
            return "Aún no tengo puntos guardados para construir el informe final. Primero activa el modo informe o agrégame respuestas específicas."
        scope = st.session_state.get("report_goal", prompt)
        report = llm_call(
            MODEL_REPORT,
            build_report_prompt(prompt, scope),
            low_latency=False,
            higher_reasoning=True,
        )
        add_topic("informe")
        st.session_state.pending_report = False
        return report

    if len(topics) >= 2 and not any(x in q for x in ["ventas", "clientes", "productos", "geograf", "medios", "retencion", "retención", "investig"]):
        options = " ".join([f"- {t}" for t in topics[-6:]])
        st.session_state.pending_report = True
        return (
            "Puedo hacerlo. Antes de generar el informe, dime sobre cuál de estos temas conversados lo quieres: "
            f"{options}  "
            "También puedes pedírmelo cruzado, por ejemplo: ventas + productos, geografía + clientes, medios + ventas o investigación externa + datos internos."
        )

    st.session_state.pending_report = False
    scope = prompt
    report = llm_call(
        MODEL_REPORT,
        build_report_prompt(prompt, scope),
        low_latency=False,
        higher_reasoning=True,
    )
    add_topic("informe")
    return report


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="NobleBotAI 🐷🐽", layout="wide")
st.title("NobleBotAI 🐷🐽")
st.caption("Inteligencia comercial, performance, investigación y marketing 360 para agencia")

with st.sidebar:
    st.header("Configuración")
    st.markdown(f"**Proyecto:** `{PROJECT_ID}`")
    st.markdown(f"**Location:** `{LOCATION}`")
    st.markdown(f"**Marca base:** `{BRAND_ID}`")
    st.markdown(f"**Modelo SQL:** `{MODEL_SQL}`")
    st.markdown(f"**Modelo Router:** `{MODEL_ROUTER}`")
    st.markdown(f"**Modelo Chat:** `{MODEL_CHAT}`")
    st.markdown(f"**Modelo Report:** `{MODEL_REPORT}`")
    st.markdown(f"**Research externo:** `{'ON' if ENABLE_EXTERNAL_RESEARCH else 'OFF'}`")

    st.subheader("Mapa de Verdad")
    for alias, fq_table in MAPA_VERDAD.items():
        st.markdown(f"- **{alias}** → `{fq_table}`")

    st.subheader("Entidades conocidas")
    for alias, info in KNOWN_ENTITIES.items():
        st.markdown(f"- **{alias}** → {info.get('description', '')}")

    st.subheader("Estado tablas oficiales")
    for alias, existe in diagnosticar_tablas().items():
        st.markdown(f"{'✅' if existe else '❌'} **{alias}**")

    if st.button("Limpiar memoria"):
        for key in [
            "messages", "chat_memory", "last_sql", "last_df", "last_answer",
            "last_route", "last_intent", "discussed_topics", "pending_report", "report_mode",
            "report_items", "report_goal", "sql_cache", "last_research_context"
        ]:
            st.session_state.pop(key, None)
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Pregúntame por clientes, ventas, productos, geografía, medios, investigación o informes...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            route = classify_route(prompt)
            st.session_state.last_route = route.route
            st.session_state.last_intent = route.intent
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
                    f"""
Responde de forma breve y útil. Mantén contexto del chat.

MEMORIA:
{summarize_memory()}

ENTIDADES_CONOCIDAS:
{known_entities_text()}

PREGUNTA:
{prompt}
""",
                    low_latency=True,
                )
                st.markdown(answer)

            elif (
                route.intent == "informe"
                or report_mode_requested(prompt)
                or add_to_report_requested(prompt)
                or final_report_requested(prompt)
                or (st.session_state.get("pending_report") and "informe" in lower_clean(prompt))
            ):
                answer = handle_report_request(prompt)
                st.markdown(answer)

            elif route.route == "research":
                with st.spinner("Investigando fuentes externas..."):
                    external_context = run_external_research(prompt)
                    answer = llm_call(
                        MODEL_CHAT,
                        build_research_prompt(prompt, external_context),
                        low_latency=False,
                        higher_reasoning=True,
                    )
                    st.markdown(answer)

                    st.session_state.last_sql = ""
                    st.session_state.last_df = None

                    if st.session_state.get("report_mode"):
                        save_current_point_to_report(
                            question=prompt,
                            answer=answer,
                            sql="",
                            df=None,
                            intent=route.intent,
                        )

                    with st.expander("Ver contexto externo usado"):
                        st.text(external_context)

            else:
                with st.spinner("Consultando tablas oficiales..."):
                    sql, df = generate_sql_with_retries(prompt, route)
                    answer = render_answer(prompt, route, sql, df)
                    st.markdown(answer)

                    st.session_state.last_sql = sql
                    st.session_state.last_df = df.copy()

                    if st.session_state.get("report_mode"):
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
                "Prueba con una pregunta más específica o revisa si el modelo, la ubicación de Vertex AI y las claves opcionales de investigación externa están configuradas correctamente."
            )
            st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.expander("Detalle técnico"):
                st.code(str(e))

