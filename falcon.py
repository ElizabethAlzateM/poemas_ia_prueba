
import os
import pandas as pd
import streamlit as st
import requests
import traceback

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="Generador de Poemas con IA",
    page_icon="✍️",
    layout="wide",
)

# =========================
# DIAGNÓSTICO Y CARGA INICIAL
# =========================
csv_path = "poems_clean.csv"
df = None
try:
    df = pd.read_csv(csv_path)
except Exception:
    st.sidebar.error("Error: No se pudo cargar poems_clean.csv. Verifica que esté en la raíz.")
    df = None

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.sidebar.warning("⚠️ HF_TOKEN no encontrado. Configúralo como variable de entorno o en st.secrets['HF_TOKEN'].")

# =========================
# CONFIGURACIÓN DEL MODELO Y ROUTER
# =========================
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"  # principal
FALLBACK_MODEL_ID = "gpt2"                  # respaldo

def router_url(model_id: str) -> str:
    return f"https://router.huggingface.co/models/{model_id}"

def hf_generate(prompt, model_id=MODEL_ID, max_tokens=300, temperature=0.9, return_full_text=False):
    """
    Cliente HTTP para Hugging Face Router con manejo de errores y fallback.
    - Payload: 'inputs' + 'parameters' (kwargs de pipelines).
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": return_full_text,
            # opcionales: top_p, top_k, repetition_penalty, etc.
        }
    }

    try:
        resp = requests.post(router_url(model_id), headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # El Router mantiene 'generated_text' para text-generation
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "generated_text" in data[0]:
                return data[0]["generated_text"], model_id, False
            # intento genérico por si el proveedor devuelve clave distinta
            for k in ("text", "output_text"):
                if k in data[0]:
                    return data[0][k], model_id, False

        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"], model_id, False

        return "Error: Respuesta inesperada del Router.", model_id, False

    except requests.HTTPError as e:
        status_code = e.response.status_code

        # 404: el modelo no está disponible -> fallback
        if status_code == 404 and model_id != FALLBACK_MODEL_ID:
            # Avisamos que activamos fallback
            st.info(f"ℹ️ 404 con **{model_id}**. Cambiando a modelo de respaldo: **{FALLBACK_MODEL_ID}**.")
            return hf_generate(prompt, model_id=FALLBACK_MODEL_ID,
                               max_tokens=max_tokens, temperature=temperature,
                               return_full_text=return_full_text)

        # 503: servicio no disponible (cold start)
        if status_code == 503:
            return "💔 **503**: el modelo está cargando o no acepta tráfico ahora (Cold Start).", model_id, False

        # 410: si alguna llamada antigua llega a api-inference (deprecado)
        if status_code == 410:
            return ("⚠️ El endpoint api-inference.huggingface.co fue deprecado. "
                    "Usa https://router.huggingface.co/models/{model_id}."), model_id, False

        # Otros errores HTTP
        return f"🚨 Error HTTP del Router: {status_code} - {e.response.text}", model_id, False

    except requests.exceptions.Timeout:
        return "⏰ **Timeout**: el modelo tardó demasiado en responder.", model_id, False
    except Exception as e:
        return "🚨 Error inesperado durante la generación.\n" + "".join(traceback.format_exception(e)), model_id, False

# =========================
# INTERFAZ STREAMLIT
# =========================
st.title("✍️ IA Generativa de Poemas en Español")

st.markdown(f"""
Aplicación generativa de poemas en español usando **{MODEL_ID}** (vía Hugging Face Router).
Si no está disponible, se aplicará fallback automático a **{FALLBACK_MODEL_ID}**.
""")

st.subheader("Configuración de la Generación")

col1, col2 = st.columns(2)
with col1:
    tema = st.text_input("Tema del poema", placeholder="Ej: La melancolía del otoño")
with col2:
    estilo = st.selectbox(
        "Estilo",
        ["Verso libre", "Soneto", "Haiku", "Romance", "Décima", "Oda",
         "Copla", "Elegía", "Égloga", "Lira", "Redondilla"]
    )

if st.button("✨ Generar Poema", type="primary"):
    if not tema or len(tema.strip()) < 3:
        st.error("Por favor, ingresa un tema válido para la generación.")
    elif not HF_TOKEN:
        st.error("El token de Hugging Face (HF_TOKEN) es necesario.")
    elif df is None or df.empty:
        st.error("El dataset de poemas no se cargó correctamente.")
    else:
        # 1) Preparar ejemplos y prompt
        ejemplos = df['content'].dropna().sample(min(3, len(df))).tolist()
        ejemplos_texto = "\n".join([f"- {e.strip()[:200]}..." for e in ejemplos])

        prompt = f"""
Eres un poeta experto en español.
Escribe un poema sobre el tema: "{tema}".
Estilo: {estilo}.
Inspírate en el estilo (sin copiar) de estos ejemplos:
{ejemplos_texto}

Ahora escribe el poema:
""".strip()

        # 2) Generar con spinner
        st.subheader(f"Resultado: Poema '{estilo}' sobre '{tema}'")
        with st.spinner("⏳ La IA está escribiendo... Esto puede tardar varios segundos."):
            poem, used_model, used_fallback_flag = hf_generate(
                prompt,
                model_id=MODEL_ID,
                max_tokens=300,
                temperature=0.9,
                return_full_text=False
            )

        # 3) Mostrar resultado y avisos
        st.success(f"✅ Generación completada con **{used_model}**.")
        if used_model == FALLBACK_MODEL_ID:
            st.warning("🔁 Se activó el **fallback** al modelo de respaldo (gpt2) porque el principal no estaba disponible.")

        st.markdown("---")
        st.markdown(poem)
        st.markdown("---")

# =========================
# AYUDA SOBRE ESTILOS
# =========================
st.markdown("""
---
### Estilos Disponibles:
* **Verso libre**: Poema sin rima ni métrica fija.
* **Soneto**: 14 versos endecasílabos con rima organizada.
* **Haiku**: Tres versos breves inspirados en la naturaleza.
* **Romance**: Versos octosílabos con rima asonante en pares.
* **Décima**: 10 versos octosílabos con rima ABBAACCDDC.
* **Oda**: Poema solemne y reflexivo.
* **Copla**: Estrofa de 4 versos octosílabos con rima en pares.
* **Elegía**: Poema melancólico sobre la pérdida.
* **Égloga**: Diálogo bucólico entre pastores.
* **Lira**: Estrofa de 5 versos con métrica 7-11-7-7-11.
* **Redondilla**: Estrofa de 4 versos octosílabos con rima ABBA.
""")