
# streamlit_app.py
# App de Streamlit que genera poemas llamando a un Inference Endpoint dedicado de Hugging Face.
# Solo necesitas definir dos secretos en el Space:
#   - HF_TOKEN: tu token de Hugging Face
#   - HF_ENDPOINT_URL: la URL del endpoint (https://<tu-endpoint>.endpoints.huggingface.cloud)

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
# LECTURA DE SECRETS / ENTORNO
# =========================
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL") or st.secrets.get("HF_ENDPOINT_URL")

# Validaciones básicas
if not HF_TOKEN:
    st.sidebar.error("❌ Falta HF_TOKEN. Agrega tu token en Settings → Secrets del Space.")
if not ENDPOINT_URL:
    st.sidebar.error("❌ Falta HF_ENDPOINT_URL. Agrega la URL del Inference Endpoint en Settings → Secrets del Space.")

# =========================
# CARGA DEL DATASET
# =========================
csv_path = "poems_clean.csv"
df = None
try:
    df = pd.read_csv(csv_path)
except Exception:
    st.sidebar.error("⚠️ No se pudo cargar 'poems_clean.csv'. Verifica que esté en la raíz del Space.")
    df = None

# =========================
# CLIENTE DEL INFERENCE ENDPOINT
# =========================
def hf_generate_via_endpoint(prompt: str, max_tokens: int = 300, temperature: float = 0.9):
    """
    Llama a tu Inference Endpoint dedicado de Hugging Face.
    Debes haber creado el endpoint y puesto su URL en HF_ENDPOINT_URL.
    El endpoint de TGI/Inference API suele responder con una lista de dicts que incluyen 'generated_text'.
    """
    if not HF_TOKEN or not ENDPOINT_URL:
        return False, "Configuración incompleta: faltan secretos HF_TOKEN o HF_ENDPOINT_URL."

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
            # Añade según necesidad:
            # "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15,
            # "return_full_text": False  # si tu servidor lo soporta
        },
        # Algunos endpoints (según cómo los configures) aceptan "stream": False/True.
        # "stream": False,
    }

    try:
        resp = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # Casos típicos: lista con {"generated_text": "..."} o dict con "generated_text"
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "generated_text" in data[0]:
                return True, data[0]["generated_text"]
            # Intento genérico por si el servidor devuelve otra clave
            for k in ("text", "output_text"):
                if k in data[0]:
                    return True, data[0][k]

        if isinstance(data, dict) and "generated_text" in data:
            return True, data["generated_text"]

        # Si llega aquí, el formato no fue el esperado:
        return False, f"Respuesta inesperada del endpoint: {str(data)[:400]}"

    except requests.HTTPError as e:
        status = e.response.status_code
        # Mensajes claros por códigos comunes
        if status == 401:
            return False, "401 No autorizado: revisa tu HF_TOKEN o permisos."
        if status == 403:
            return False, "403 Prohibido: el endpoint o el modelo requieren permisos/licencia."
        if status == 404:
            return False, "404 No encontrado: revisa la URL HF_ENDPOINT_URL."
        if status == 503:
            return False, "503 Servicio no disponible: el endpoint está arrancando (cold start) o saturado."
        return False, f"HTTP {status}: {e.response.text}"

    except requests.Timeout:
        return False, "⏰ Timeout: el endpoint tardó demasiado en responder."
    except Exception as ex:
        return False, "Excepción: " + "".join(traceback.format_exception(ex))[:800]

# =========================
# UI
# =========================
st.title("✍️ IA Generativa de Poemas en Español (Endpoint dedicado)")

st.markdown("""
Esta app usa un **Inference Endpoint** de Hugging Face para generar poemas.
Asegúrate de haber configurado los **Secrets** del Space: `HF_TOKEN` y `HF_ENDPOINT_URL`.
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

# Parámetros (opcionales) en la barra lateral
st.sidebar.header("Parámetros de Decodificación")
max_new_tokens = st.sidebar.slider("Máx. tokens nuevos", 64, 512, 256, step=32)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.9, step=0.05)
top_p = st.sidebar.slider("Top-p (opcional)", 0.0, 1.0, 0.9, step=0.05)
top_k = st.sidebar.slider("Top-k (opcional)", 0, 200, 50, step=5)
repetition_penalty = st.sidebar.slider("Penalización de repetición (opc.)", 1.0, 2.0, 1.15, step=0.05)

if st.button("✨ Generar Poema", type="primary"):
    if not tema or len(tema.strip()) < 3:
        st.error("Por favor, ingresa un tema válido para la generación.")
    elif not HF_TOKEN or not ENDPOINT_URL:
        st.error("Faltan secretos HF_TOKEN y/o HF_ENDPOINT_URL en el Space.")
    elif df is None or df.empty:
        st.error("El dataset de poemas no se cargó correctamente (poems_clean.csv).")
    else:
        # 1) Preparar ejemplos y prompt
        ejemplos = df['content'].dropna().sample(min(3, len(df))).tolist()
        ejemplos_texto = "\n".join([f"- {e.strip()[:200]}..." for e in ejemplos])

        # Prompt
        prompt = f"""
Eres un poeta experto en español.
Escribe un poema sobre el tema: "{tema}".
Estilo: {estilo}.
Inspírate en el estilo (sin copiar) de estos ejemplos:
{ejemplos_texto}

Ahora escribe el poema:
""".strip()

        # 2) Parametrización adicional (si tu servidor soporta estos kwargs)
        #   No todos los servidores aceptan 'top_p', 'top_k', 'repetition_penalty' directamente;
        #   si tu endpoint usa TGI, normalmente sí. Puedes pasarlos añadiéndolos en hf_generate_via_endpoint.
        #   Para mantener compatibilidad, lo haremos así:
        def _merge_extra_params(base_params: dict) -> dict:
            # Solo añade si no son valores por defecto
            extras = {}
            if top_p != 0.9:
                extras["top_p"] = top_p
            if top_k != 50:
                extras["top_k"] = top_k
            if repetition_penalty != 1.15:
                extras["repetition_penalty"] = repetition_penalty
            # Inserta en "parameters"
            base_params = dict(base_params)  # copia
            base_params.setdefault("parameters", {}).update(extras)
            return base_params

        st.subheader(f"Resultado: Poema '{{estilo}}' sobre '{tema}'")
        with st.spinner("⏳ El endpoint está generando el poema..."):
            # Construimos un payload para pasar los extras; reusamos la función principal
            # para evitar duplicar código: hacemos una pequeña variante ad hoc:
            base_payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                }
            }

            # Enviamos manualmente la petición aquí para poder inyectar los extras:
            if not HF_TOKEN or not ENDPOINT_URL:
                ok, poem = False, "Config del endpoint incompleta."
            else:
                headers = {
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
                payload = _merge_extra_params(base_payload)
                try:
                    resp = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        poem = data[0].get("generated_text") or data[0].get("text") or data[0].get("output_text")
                        ok = poem is not None
                        if not ok:
                            poem = f"Respuesta inesperada del endpoint: {str(data)[:400]}"
                    elif isinstance(data, dict) and "generated_text" in data:
                        poem = data["generated_text"]
                        ok = True
                    else:
                        ok = False
                        poem = f"Respuesta inesperada del endpoint: {str(data)[:400]}"
                except requests.HTTPError as e:
                    ok = False
                    status = e.response.status_code
                    if status == 401:
                        poem = "401 No autorizado: revisa HF_TOKEN o permisos."
                    elif status == 403:
                        poem = "403 Prohibido: el endpoint o el modelo requieren permisos/licencia."
                    elif status == 404:
                        poem = "404 No encontrado: revisa la URL HF_ENDPOINT_URL."
                    elif status == 503:
                        poem = "503 Servicio no disponible: el endpoint está arrancando (cold start) o saturado."
                    else:
                        poem = f"HTTP {status}: {e.response.text}"
                except requests.Timeout:
                    ok, poem = False, "⏰ Timeout: el endpoint tardó demasiado en responder."
                except Exception as ex:
                    ok, poem = False, "Excepción: " + "".join(traceback.format_exception(ex))[:800]

        if ok:
            st.success("✅ Generación completada desde tu Inference Endpoint.")
            st.markdown("---")
            # Muchos servidores devuelven el prompt + la continuación; si deseas, puedes limpiar:
            split_key = "Ahora escribe el poema:"
            if split_key in poem:
                poem = poem.split(split_key, 1)[-1].strip()
            st.markdown(poem)
            st.markdown("---")
        else:
            st.error("🚨 No se pudo generar el poema.")
            st.code(poem)
            st.info("Revisa los Secrets (HF_TOKEN, HF_ENDPOINT_URL), licencias del modelo y el estado del Endpoint.")
        

# =========================
# AYUDA SOBRE ESTILOS
# =========================
st.markdown("""
---
### Estilos Disponibles:
* **Verso libre**: Poema sin rima ni métrica fija.
* **Soneto**: 14 versos endecasílabos con rima.
* **Haiku**: Tres versos breves inspirados en la naturaleza.
* **Romance**: Versos octosílabos con rima asonante en pares.
* **Décima**: 10 versos octosílabos con rima ABBAACCDDC.
* **Oda**: Poema solemne y reflexivo.
* **Copla**: Estrofa de 4 versos octosílabos con rima en pares.
* **Elegía**: Poema melancólico sobre la pérdida.
* **Égloga**: Diálogo bucólico entre pastores.
* **Lira**: Estrofa 7-11-7-7-11.
* **Redondilla**: 4 versos octosílabos con rima ABBA.
""")