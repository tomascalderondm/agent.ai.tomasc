import streamlit as st
from google.cloud import bigquery
from google import genai
from google.oauth2 import service_account

# --- CONFIGURACIÓN ESTRATÉGICA ---
PROJECT_ID = "base-de-datos-489323"
DATASET_VISTA = "campo_noble_ai"

# 1. Cargar Credenciales
creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)

# 2. Inicializar Clientes
bq_client = bigquery.Client(credentials=creds, project=PROJECT_ID)
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def obtener_esquema_real():
    query = f"SELECT column_name FROM `{PROJECT_ID}.{DATASET_VISTA}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'vista_maestra_agente_360'"
    df = bq_client.query(query).to_dataframe()
    return df['column_name'].tolist()

def chatbot_eterno(pregunta_usuario, historial_contexto):
    columnas_actuales = obtener_esquema_real()
    
    # PROMPT DE SQL CON MEMORIA Y PROACTIVIDAD
    prompt_sql = f"""
    Eres NobleBotAI 🐷🐽, experto en SQL para BigQuery. Tu misión es generar el query perfecto para la tabla `{PROJECT_ID}.{DATASET_VISTA}.vista_maestra_agente_360`.
    
    COLUMNAS DISPONIBLES: {columnas_actuales}
    
    CONTEXTO DE LA CONVERSACIÓN (MEMORIA):
    {historial_contexto}
    
    PREGUNTA DEL USUARIO: {pregunta_usuario}
    
    REGLAS DE PROACTIVIDAD:
    1. Si el usuario pregunta por una persona (ej: "Patricio") y no tienes su email, usa: `WHERE nombre_cliente LIKE '%Patricio%'`.
    2. MEMORIA OPERATIVA: Si en el contexto ya se mencionó un email o un dato específico, úsalo automáticamente para filtrar este nuevo query.
    3. Si piden "detalles", selecciona todas las columnas.
    4. Responde ÚNICAMENTE con el código SQL.
    """
    
    intentos = 0
    while intentos < 2:
        try:
            res_sql = genai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt_sql)
            query_limpio = res_sql.text.strip().replace('```sql', '').replace('```', '')
            resultado_df = bq_client.query(query_limpio).to_dataframe()
            return resultado_df, query_limpio
        except Exception as e:
            intentos += 1
            prompt_sql += f"\n\nERROR PREVIO: {str(e)}. Intenta una ruta distinta."
            if intentos == 2: return None, f"Error tras reintentos: {e}"

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="NobleBotAI 🐷🐽", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🤖 Configuración")
    if st.button("Limpiar Memoria"):
        st.session_state.messages = []
        st.rerun()

st.title("Consultor IA Campo Noble 🐷🐽")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber de tus datos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("NobleBotAI está analizando..."):
            # Pasamos los últimos mensajes como memoria para el SQL
            memoria_reciente = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
            
            df_datos, sql_usado = chatbot_eterno(prompt, memoria_reciente)
            
            if df_datos is not None:
                # --- PERSONALIDAD DE NOBLEBOT (Basada en tu informe estratégico) ---
                instrucciones_ia = """
                Eres NobleBotAI, el asistente IA chancho 🐷🐽 y Director de Inteligencia de Campo Noble.
                Tu tono es ejecutivo, rápido, proactivo y estratégico. NO uses la palabra "Jefe".
                
                INTELIGENCIA ESTRATÉGICA (Basada en reporte Abril 2026):
                - Un cliente recurrente es quien tiene múltiples compras y BAJA RECENCIA (compró hace < 120 días)[cite: 7, 10].
                - Si un cliente tiene mucho gasto pero está inactivo (ej: Patricio Arriagada con 182 días o Carolina Gonzalez con 258 días), trátalo como "Fuga de Valor"[cite: 8, 17, 18].
                - Si los datos son pocos, no digas que es un fallo; explícalo como una "oportunidad quirúrgica" o "punta de lanza".
                - Sugiere tácticas proactivas: "Operación de Reconexión", "Bundles" de alta fidelidad (Loin Kaburi + Punta Costillita), o pases a Meta Ads[cite: 25, 29, 35].
                """
                
                final_prompt = f"Instrucciones: {instrucciones_ia}\nDatos: {df_datos.to_string()}\nPregunta: {prompt}"
                res_final = genai_client.models.generate_content(model='gemini-2.5-flash', contents=final_prompt)
                
                respuesta = res_final.text
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                with st.expander("Ver Mapa de Verdad (SQL)"):
                    st.code(sql_usado)
            else:
                st.error("Hubo un problema en la conexión, pero soy un chancho persistente. ¿Podrías reformular la pregunta?")
