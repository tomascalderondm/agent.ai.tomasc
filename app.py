import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account


# ============================================================
# CONFIGURACION BASE
# ============================================================

PROJECT_ID = "data-marketing-360"
BRAND_ID = st.secrets.get("BRAND_ID", "campo_noble")

CORE_DATASET = f"{BRAND_ID}_core"
AI_DATASET = f"{BRAND_ID}_ai"

MAX_ROWS_RESULT = 200
MAX_HISTORY_MESSAGES = 6
MAX_SQL_RETRIES = 2
MAX_BYTES_BILLED = 5 * 1024 * 1024 * 1024  # 5 GB

MODEL_SQL = st.secrets.get("MODEL_SQL", "gemini-2.5-flash")
MODEL_RESPONSE = st.secrets.get("MODEL_RESPONSE", "gemini-2.5-flash")
MODEL_MEDIA = st.secrets.get("MODEL_MEDIA", "gemini-2.5-flash")

ENABLE_MEDIA_GROUNDING = st.secrets.get("ENABLE_MEDIA_GROUNDING", True)
ENABLE_EXTERNAL_CORROBORATION = st.secrets.get("ENABLE_EXTERNAL_CORROBORATION", True)


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


def resumir_dataframe_para_prompt(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "Sin filas."
    return df.head(max_rows).to_string(index=False)


def formatear_moneda_clp(valor) -> str:
    try:
        return f"${valor:,.0f} CLP".replace(",", ".")
    except Exception:
        return str(valor)


def es_pregunta_medios_o_inversion(pregunta_usuario: str) -> bool:
    q = pregunta_usuario.lower()
    keywords = [
        "inversión", "inversion", "medios", "paid media", "google ads", "meta ads",
        "presupuesto", "budget", "roas", "cpa", "cac", "campaña", "campana",
        "performance", "escalar", "escalamiento", "retargeting", "prospecting",
        "audiencia", "audiencias", "funnel", "embudo", "bidding", "puja"
    ]
    return any(x in q for x in keywords)


def obtener_tablas_prioritarias(pregunta_usuario: str) -> List[str]:
    q = pregunta_usuario.lower()

    prioridades = []

    if any(x in q for x in ["venta", "ventas", "mes", "año", "anio", "comparar", "facturación", "facturacion", "ticket"]):
        prioridades += [
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["fact_pedidos"],
        ]

    if any(x in q for x in ["cliente", "clientes", "ltv", "segmento", "recurrente", "nuevo", "riesgo", "inactividad", "churn", "reactivar", "winback"]):
        prioridades += [
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_clientes"],
            MAPA_VERDAD["dim_clientes"],
        ]

    if any(x in q for x in ["producto", "productos", "retención", "retencion", "recompra", "gancho", "afinidad", "bundle", "mix", "canasta"]):
        prioridades += [
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["resumen_cliente_producto"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["afinidad_productos"],
            MAPA_VERDAD["fact_pedido_productos"],
        ]

    if any(x in q for x in ["comuna", "geo", "geografía", "geografia", "ciudad", "zona", "territorio", "region", "región"]):
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

    if any(x in q for x in [
        "inversión", "inversion", "medios", "paid media", "google ads", "presupuesto",
        "budget", "roas", "cpa", "cac", "performance", "retargeting", "prospecting",
        "audiencia", "audiencias", "escalar", "bidding"
    ]):
        prioridades += [
            MAPA_VERDAD["action_center"],
            MAPA_VERDAD["resumen_ventas_periodo"],
            MAPA_VERDAD["perfil_clientes_360"],
            MAPA_VERDAD["resumen_productos_ventas"],
            MAPA_VERDAD["resumen_productos_retencion"],
            MAPA_VERDAD["resumen_geografia"],
            MAPA_VERDAD["resumen_clientes"],
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


def extraer_urls_grounding(response) -> List[str]:
    urls: List[str] = []

    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            if grounding_metadata is None:
                continue

            grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            for chunk in grounding_chunks:
                web_info = getattr(chunk, "web", None)
                uri = getattr(web_info, "uri", None) if web_info else None
                if uri and uri not in urls:
                    urls.append(uri)
    except Exception:
        pass

    return urls[:8]


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
Eres NobleBotAI, motor de consulta SQL de BigQuery para inteligencia comercial, performance y marketing 360 de una agencia de marketing y medios.

## OBJETIVO
Debes escribir una única consulta SQL válida, segura y eficiente para responder la pregunta del usuario usando exclusivamente el Mapa de Verdad oficial.

Tu trabajo es responder preguntas de negocio con foco en:
- inversión
- performance comercial
- clientes
- recurrencia
- recompra
- churn
- ticket
- LTV
- mix de productos
- geografía
- audiencias
- oportunidades de activación, retención y win-back

## MAPA DE VERDAD OFICIAL
Puedes usar únicamente estas tablas:

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

## CRITERIO DE NEGOCIO
Piensa como un director de estrategia, medios y growth:
- prioriza tablas que permitan responder con la mayor claridad ejecutiva y el menor costo de consulta.
- si la pregunta es ejecutiva, devuelve agregados accionables antes que detalle transaccional.
- si la pregunta es operativa, entrega el detalle mínimo útil.
- si la pregunta apunta a inversión o medios, busca señales de eficiencia, concentración de valor, fuga, recurrencia, potencial de reactivación y redistribución presupuestaria.
- no compliques la consulta si una tabla AI ya responde la pregunta.
- si no existen métricas de paid media en las tablas oficiales, no las inventes.

## REGLAS DE PRIORIZACION
1. Usa primero tablas resumen AI cuando sean suficientes para responder.
2. Usa tablas CORE solo cuando la pregunta requiera granularidad histórica o transaccional.
3. Si la pregunta es sobre ventas por período, evolución, estacionalidad, ticket o comparación temporal, prioriza `resumen_ventas_periodo`.
4. Si la pregunta es sobre productos que más venden, aportan o concentran ingresos, prioriza `resumen_productos_ventas`.
5. Si la pregunta es sobre recompra, retención, producto ancla, producto gancho o canasta, prioriza `resumen_productos_retencion`, `resumen_cliente_producto` y `afinidad_productos`.
6. Si la pregunta es sobre clientes, segmentos, LTV, recurrencia, reactivación, riesgo o churn, prioriza `perfil_clientes_360`, `resumen_clientes` y `dim_clientes`.
7. Si la pregunta es sobre comportamiento de compra detallado, puedes usar `fact_pedidos` y `fact_pedido_productos`.
8. Si la pregunta es sobre geografía comercial, comunas o regiones, prioriza `resumen_geografia`.
9. Si la pregunta es sobre calidad de datos o cobertura, prioriza `auditoria_datos`.
10. Si la pregunta es sobre oportunidades accionables o lectura ejecutiva, prioriza `action_center`.
11. Si la pregunta es ambigua, elige la ruta más segura, más liviana y más ejecutiva.

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
11. Si la pregunta pide una búsqueda por nombre de cliente sin email exacto, puedes usar LIKE sobre nombre.
12. Si del historial existe una referencia clara y útil, puedes reutilizarla.
13. Si una tabla agregada resuelve la pregunta, no escales a una tabla más pesada.
14. Evita scans innecesarios.

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
Eres NobleBotAI, asistente de inteligencia comercial, performance y marketing 360 para una agencia de marketing y medios.

## OBJETIVO
Debes transformar resultados tabulares en una respuesta ejecutiva, clara, precisa, accionable y orientada a negocio.

Debes pensar como un estratega senior de agencia:
- performance
- inversión
- retención
- recompra
- mix de productos
- segmentación
- geografía comercial
- reactivación
- oportunidades de medios
- eficiencia de crecimiento

## TONO
- Ejecutivo
- Estratégico
- Directo
- Profesional
- Humano
- Basado en evidencia
- Nunca robótico
- Nunca inflado
- Nunca vendedor de humo

## REGLAS
1. Responde siempre en español.
2. Empieza con: "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽."
3. Luego sigue con: "Consulté en las tablas oficiales y aquí tienes el resultado..."
4. Abre con el hallazgo principal.
5. Luego explica por qué ese hallazgo importa para negocio, inversión o performance.
6. Si corresponde, interpreta en lógica de marketing 360:
   - adquisición
   - conversión
   - recurrencia
   - recompra
   - retención
   - riesgo
   - geografía
   - mix de productos
   - concentración de ventas
   - oportunidad de reactivación
7. Si hay cifras monetarias, preséntalas como CLP.
8. Si hay pocos datos, dilo con naturalidad, sin sonar defensivo.
9. No inventes información ni métricas que no estén respaldadas por los datos.
10. Si la evidencia es limitada, dilo explícitamente.
11. Cierra siempre con una recomendación concreta, priorizada y accionable.
12. Cuando sea útil, estructura la respuesta en:
   - Diagnóstico
   - Qué significa
   - Riesgo u oportunidad
   - Próximo paso
13. Distingue claramente entre:
   - evidencia interna,
   - benchmark o lineamiento externo,
   - recomendación estratégica.

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


def construir_prompt_contraste_medios(
    pregunta_usuario: str,
    historial_contexto: str,
    resumen_diagnostico_interno: str,
    datos_texto: str,
) -> str:
    return f"""
Eres NobleBotAI, estratega senior de marketing y medios para una agencia de performance.

## OBJETIVO
Debes contrastar un diagnóstico interno de negocio con buenas prácticas recientes y lineamientos públicos de marketing digital, inversión y medios.

## REGLAS CRITICAS
1. No contradigas los datos internos sin evidencia.
2. Distingue siempre entre:
   - lo demostrado por los datos internos,
   - lo respaldado por fuentes externas recientes,
   - y lo recomendado estratégicamente.
3. No inventes métricas de canal que no existen en la data interna.
4. Si faltan métricas de paid media, habla de hipótesis y criterios de asignación, no de certezas operativas.
5. Prioriza temas de:
   - inversión
   - bidding
   - valor de conversión
   - retención
   - audiencias
   - expansión geográfica
   - concentración de ventas
   - riesgo de dependencia
   - reactivación
   - eficiencia comercial
6. Si el contraste externo no agrega valor, dilo.
7. Responde en español.
8. Sé ejecutivo, concreto y útil para una agencia.

## ESTRUCTURA OBLIGATORIA
1. Contraste externo
2. Qué valida o tensiona del diagnóstico interno
3. Recomendación de inversión / medios priorizada
4. Advertencias o límites de evidencia

## CONTEXTO
{historial_contexto}

## PREGUNTA DEL USUARIO
{pregunta_usuario}

## DIAGNOSTICO INTERNO
{resumen_diagnostico_interno}

## DATOS RESUMIDOS
{datos_texto}
"""


def construir_prompt_fusion_final(
    pregunta_usuario: str,
    respuesta_interna: str,
    contraste_externo: str,
) -> str:
    return f"""
Eres NobleBotAI, asistente de inteligencia comercial, performance y marketing 360 para una agencia de marketing y medios.

## OBJETIVO
Debes fusionar un diagnóstico interno basado en datos propios con un contraste externo basado en fuentes públicas recientes.

## REGLAS
1. Responde siempre en español.
2. Empieza con: "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽."
3. Luego sigue con: "Consulté en las tablas oficiales y aquí tienes el resultado..."
4. Separa con claridad:
   - Diagnóstico interno
   - Contraste externo
   - Recomendación
5. Si el contraste externo valida el diagnóstico, dilo.
6. Si lo tensiona, dilo con precisión.
7. No inventes métricas.
8. Si faltan datos de medios reales, aclara que la recomendación es estratégica y no una auditoría de canal completa.
9. Mantén tono ejecutivo, accionable y orientado a inversión.

## PREGUNTA
{pregunta_usuario}

## RESPUESTA INTERNA
{respuesta_interna}

## CONTRASTE EXTERNO
{contraste_externo}
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
            model=MODEL_SQL,
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
            "Consulté en las tablas oficiales y aquí tienes el resultado: no encontré filas que calcen exactamente con esa búsqueda.\n\n"
            "La lectura de negocio no es que no exista oportunidad, sino que el filtro actual está demasiado estrecho o no coincide con la forma en que la señal está guardada en el modelo.\n\n"
            "Recomendación inmediata: conviene reformular la búsqueda por período, cliente, producto, comuna, segmento o una variante del nombre para capturar mejor la oportunidad comercial."
        )

    datos_texto = resumir_dataframe_para_prompt(df_datos)

    prompt_final = construir_prompt_respuesta(
        pregunta_usuario=pregunta_usuario,
        historial_contexto=historial_contexto,
        sql_usado=sql_usado,
        datos_texto=datos_texto,
        num_filas=len(df_datos),
    )

    respuesta_final = genai_client.models.generate_content(
        model=MODEL_RESPONSE,
        contents=prompt_final,
    )
    texto = obtener_texto_modelo(respuesta_final).strip()

    if not texto:
        texto = (
            "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
            "Consulté en las tablas oficiales y aquí tienes el resultado.\n\n"
            "Ya recuperé correctamente los datos, pero la redacción final salió vacía. "
            "El hallazgo sigue disponible en el SQL y en la tabla de resultados."
        )

    return texto


def generar_contraste_externo_medios(
    pregunta_usuario: str,
    historial_contexto: str,
    respuesta_interna: str,
    df_datos: pd.DataFrame,
) -> Tuple[str, List[str]]:
    if not ENABLE_EXTERNAL_CORROBORATION:
        return "", []

    datos_texto = resumir_dataframe_para_prompt(df_datos, max_rows=40)
    prompt_media = construir_prompt_contraste_medios(
        pregunta_usuario=pregunta_usuario,
        historial_contexto=historial_contexto,
        resumen_diagnostico_interno=respuesta_interna,
        datos_texto=datos_texto,
    )

    if not ENABLE_MEDIA_GROUNDING:
        response = genai_client.models.generate_content(
            model=MODEL_MEDIA,
            contents=prompt_media,
        )
        return obtener_texto_modelo(response).strip(), []

    try:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )

        response = genai_client.models.generate_content(
            model=MODEL_MEDIA,
            contents=prompt_media,
            config=config,
        )

        texto = obtener_texto_modelo(response).strip()
        urls = extraer_urls_grounding(response)
        return texto, urls

    except Exception:
        # Fallback silencioso si el modelo/config grounding no está disponible
        response = genai_client.models.generate_content(
            model=MODEL_MEDIA,
            contents=prompt_media,
        )
        return obtener_texto_modelo(response).strip(), []


def fusionar_respuesta_final(
    pregunta_usuario: str,
    respuesta_interna: str,
    contraste_externo: str,
) -> str:
    if not contraste_externo.strip():
        return respuesta_interna

    prompt_fusion = construir_prompt_fusion_final(
        pregunta_usuario=pregunta_usuario,
        respuesta_interna=respuesta_interna,
        contraste_externo=contraste_externo,
    )

    response = genai_client.models.generate_content(
        model=MODEL_RESPONSE,
        contents=prompt_fusion,
    )

    texto = obtener_texto_modelo(response).strip()
    return texto or respuesta_interna


# ============================================================
# INTERFAZ STREAMLIT
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

if "last_external_contrast" not in st.session_state:
    st.session_state.last_external_contrast = ""

if "last_external_urls" not in st.session_state:
    st.session_state.last_external_urls = []

with st.sidebar:
    st.header("Configuración")
    st.markdown(f"**Proyecto:** `{PROJECT_ID}`")
    st.markdown(f"**Marca base:** `{BRAND_ID}`")
    st.markdown(f"**Modelo SQL:** `{MODEL_SQL}`")
    st.markdown(f"**Modelo respuesta:** `{MODEL_RESPONSE}`")
    st.markdown(f"**Modelo contraste externo:** `{MODEL_MEDIA}`")
    st.markdown(f"**Contraste externo:** `{ENABLE_EXTERNAL_CORROBORATION}`")
    st.markdown(f"**Grounding Google Search:** `{ENABLE_MEDIA_GROUNDING}`")

    st.subheader("Mapa de Verdad")
    for alias, fq_table in MAPA_VERDAD.items():
        st.markdown(f"- **{alias}** → `{fq_table}`")

    if st.button("Limpiar memoria"):
        st.session_state.messages = []
        st.session_state.last_sql = ""
        st.session_state.last_df = None
        st.session_state.last_external_contrast = ""
        st.session_state.last_external_urls = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregúntame por clientes, ventas, productos, comunas, retención o inversión en medios..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("NobleBotAI está consultando las tablas oficiales..."):
            try:
                memoria_reciente = construir_contexto_historial(st.session_state.messages)
                sql_usado, df_datos = generar_sql_con_reintentos(prompt, memoria_reciente)

                respuesta_interna = responder_como_noblebot(
                    prompt,
                    memoria_reciente,
                    sql_usado,
                    df_datos,
                )

                contraste_externo = ""
                external_urls: List[str] = []

                if es_pregunta_medios_o_inversion(prompt):
                    contraste_externo, external_urls = generar_contraste_externo_medios(
                        pregunta_usuario=prompt,
                        historial_contexto=memoria_reciente,
                        respuesta_interna=respuesta_interna,
                        df_datos=df_datos,
                    )

                respuesta_final = fusionar_respuesta_final(
                    pregunta_usuario=prompt,
                    respuesta_interna=respuesta_interna,
                    contraste_externo=contraste_externo,
                )

                st.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                st.session_state.last_sql = sql_usado
                st.session_state.last_df = df_datos
                st.session_state.last_external_contrast = contraste_externo
                st.session_state.last_external_urls = external_urls

                with st.expander("Ver SQL usado"):
                    st.code(sql_usado, language="sql")

                with st.expander("Ver datos tabulares"):
                    st.dataframe(df_datos.head(MAX_ROWS_RESULT), use_container_width=True)

                if contraste_externo:
                    with st.expander("Ver contraste externo de medios"):
                        st.markdown(contraste_externo)

                        if external_urls:
                            st.markdown("**Fuentes públicas detectadas:**")
                            for url in external_urls:
                                st.markdown(f"- {url}")

            except Exception as e:
                respuesta_error = (
                    "Hola, soy NobleBotAI, tu asistente IA chancho🐷🐽.\n\n"
                    "Consulté las rutas oficiales, pero hubo un problema técnico al construir o ejecutar la consulta.\n\n"
                    "La oportunidad sigue ahí; lo que falló fue la ruta de consulta, no la lógica comercial. "
                    "Prueba con una pregunta más específica por cliente, producto, período, comuna, segmento o tipo de oportunidad."
                )
                st.markdown(respuesta_error)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_error})

                with st.expander("Detalle técnico"):
                    st.code(str(e))
