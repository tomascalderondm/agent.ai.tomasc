import streamlit as st
from google.cloud import bigquery
from google import genai
from google.oauth2 import service_account

# --- CONFIGURACIÓN DE SEGURIDAD (LEER DESDE STREAMLIT SECRETS) ---
PROJECT_ID = "data-marketing-360"
DATASET_VISTA = "campo_noble_ai"

# 1. Cargar Credenciales de Google Cloud
creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)

# 2. Inicializar Clientes
bq_client = bigquery.Client(credentials=creds, project=PROJECT_ID)
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def obtener_esquema_real():
    query = f"SELECT column_name FROM `{PROJECT_ID}.{DATASET_VISTA}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'vista_maestra_agente_360'"
    df = bq_client.query(query).to_dataframe()
    return df['column_name'].tolist()

def chatbot_eterno(pregunta_usuario):
    columnas_actuales = obtener_esquema_real()
    prompt_sql = f"""
    Eres un experto en BigQuery. Genera un query SQL para la tabla `{PROJECT_ID}.{DATASET_VISTA}.vista_maestra_agente_360`.
    COLUMNAS DISPONIBLES HOY: {columnas_actuales}
    PREGUNTA DEL USUARIO: {pregunta_usuario}
    REGLA: Responde SOLO con el código SQL. Si la columna no está, usa la que más se parezca.
    """
    intentos = 0
    while intentos < 2:
        try:
            res_sql = genai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt_sql)
            query_limpio = res_sql.text.strip().replace('```sql', '').replace('```', '')
            resultado_df = bq_client.query(query_limpio).to_dataframe()
            return resultado_df, query_limpio
        except Exception as e:
            intentos += 1
            prompt_sql += f"\n\nERROR PREVIO: {str(e)}. Corrige el SQL."
            if intentos == 2: return None, f"Error tras reintentos: {e}"

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="IA Agencia 360", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🤖 Configuración")
    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Consultor IA Campo Noble")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber de tus datos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando BigQuery..."):
            df_datos, sql_usado = chatbot_eterno(prompt)
            if df_datos is not None:
                historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                final_prompt = f"Historial: {historial}\nPregunta: {prompt}\nDatos: {df_datos.to_string()}\nResponde como consultor senior."
                res_final = genai_client.models.generate_content(model='gemini-2.0-flash', contents=final_prompt)
                respuesta = res_final.text
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                with st.expander("Ver SQL y Datos"):
                    st.code(sql_usado)
                    st.dataframe(df_datos)
            else:
                st.error(sql_usado)
