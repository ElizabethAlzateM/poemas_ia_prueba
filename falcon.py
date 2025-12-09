
import os
import pandas as pd
import streamlit as st
import requests
import traceback
from typing import List, Optional, Tuple

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
# MODELOS Y PROVEEDORES A PROBAR (EN ORDEN)
# =========================
# Incluimos varios instruct de buen rendimiento y opciones livianas.
CANDIDATE_MODELS: List[str] = [
    "HuggingFaceH4/zephyr-7b-beta",           # suele ir bien, pero puede no estar activo en algunos providers
    "mistralai/Mistral-7B-Instruct-v0.2",     # muy sólido para ES
    "google/gemma-2-2b-it",                   # más ligero, rápido para demo
    "Qwen/Qwen2.5-7B-Instruct",               # buen desempeño general
    "gpt2"                                    # último recurso
]

# Proveedores del Router a intentar. None -> enrutado automático.
CANDIDATE_PROVIDERS: List[Optional[str]] = [None, "hf-inference", "together", "fireworks", "perplexity"]

ROUTER_BASE = "https://router.huggingface.co/models"

def router_url(model_id: str) -> str:
    return f"{ROUTER_BASE}/{model_id}"

def call_router(prompt: str,
                model_id: str,
                provider: Optional[str],
                max_tokens: int = 300,
                temperature: float = 0.9,
                return_full_text: bool = False) -> Tuple[bool, str, Optional[str]]:
    """
    Llama al Router para (modelo, proveedor). Devuelve:
    - ok: bool
    - texto_o_error: poema si ok=True, en caso contrario, mensaje de error legible
    - used_key: clave indicativa del proveedor realmente usado (o el solicitado)
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
        }
    }
    # Si el usuario solicita forzar un provider específico
    if provider:
        payload["provider"] = provider

    try:
        resp = requests.post(router_url(model_id), headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # Formatos habituales: lista con dict {'generated_text': ...}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "generated_text" in data[0]:
                return True, data[0]["generated_text"], provider or "auto"
            # Intento genérico por si el proveedor devuelve otra clave:
            for k in ("text", "output_text"):
                if k in data[0]:
                    return True, data[0][k], provider or "auto"

        # También hay casos con dict con 'generated_text'
        if isinstance(data, dict) and "generated_text" in data:
            return True, data["generated_text"], provider or "auto"

        # Respuesta inesperada pero con 2xx
        return False, f"Respuesta inesperada del Router para {model_id} (provider={provider}). Payload: {str(data)[:400]}", provider or "auto"

    except requests.HTTPError as e:
        status = e.response.status_code
        # Mensajes claros para los códigos más comunes
        if status == 404:
            return False, f"404 Not Found para {model_id} (provider={provider}).", provider or "auto"
        if status == 503:
            return False, f"503 Servicio no disponible (cold start) para {model_id} (provider={provider}).", provider or "auto"
        if status == 410:
            return False, ("410: api-inference.huggingface.co fue deprecado; usa router.huggingface.co. "
                           f"(Modelo={model_id}, provider={provider})"), provider or "auto"
        # Otros HTTP
        return False, f"HTTP {status} para {model_id} (provider={provider}). Body: {e.response.text}", provider or "auto"

    except requests.Timeout:
        return False, f"Timeout para {model_id} (provider={provider}).", provider or "auto"
    except Exception as ex:
        return False, "Excepción: " + "".join(traceback.format_exception(ex))[:800], provider or "auto"

def robust_generate(prompt: str,
                    models: List[str],
                    providers: List[Optional[str]],
                    max_tokens: int = 300,
                    temperature: float = 0.9) -> Tuple[bool, str, Optional[str], Optional[str], list]:
    """
    Intenta (modelo x proveedor) en orden hasta que uno funcione.
    Devuelve:
    - ok: bool
    - text_or_error: poema o mensaje final
    - used_model: modelo que funcionó (o None)
    - used_provider: proveedor usado (o None/auto)
    - tried_messages: lista de causas de fallo para diagnóstico
    """
    tried_msgs = []
    for m in models:
        for p in providers:
            ok, out, used_provider = call_router(
                prompt, m, provider=p, max_tokens=max_tokens, temperature=temperature, return_full_text=False
            )
            if ok:
                return True, out, m, used_provider, tried_msgs
            else:
                tried_msgs.append(out)
    return False, "Ninguna combinación (modelo/proveedor) respondió correctamente.", None, None, tried_msgs

# =========================
# INTERFAZ STREAMLIT
# =========================
st.title("✍️ IA Generativa de Poemas en Español")

st.markdown("""
Esta app usa el **Hugging Face Router** y probará automáticamente varias combinaciones de **modelo** y **proveedor**
hasta encontrar una disponible. El payload usa `inputs` + `parameters`, como en los *pipelines* de Hugging Face.
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

        # 2) Generar con robustez
        st.subheader(f"Resultado: Poema '{estilo}' sobre '{tema}'")
        with st.spinner("⏳ Buscando un modelo/proveedor disponible y generando..."):
            ok, text_or_error, used_model, used_provider, tried_msgs = robust_generate(
                prompt,
                models=CANDIDATE_MODELS,
                providers=CANDIDATE_PROVIDERS,
                max_tokens=300,
                temperature=0.9
            )

        if ok:
            prov_label = used_provider if used_provider else "auto"
            st.success(f"✅ Generación completada con **{used_model}** (provider: **{prov_label}**).")
            st.markdown("---")
            st.markdown(text_or_error)
            st.markdown("---")
        else:
            st.error("🚨 No se pudo generar el poema: ninguna combinación respondió correctamente.")
            with st.expander("Detalles de diagnóstico (intentos y causas)"):
                for msg in tried_msgs:
                    st.write(f"- {msg}")
            st.info("Sugerencias: prueba nuevamente (cold start), verifica permisos/licencias del modelo o cambia tu token a uno con 'Inference Providers' habilitado.")

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