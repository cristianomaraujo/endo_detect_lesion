import os, io, base64, hashlib
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper
from openai import OpenAI

# Carrega a variável da API
load_dotenv()
SENHA_OPEN_AI = os.getenv("SENHA_OPEN_AI") or st.secrets["SENHA_OPEN_AI"]

# Instancia o cliente OpenAI corretamente
client = OpenAI(api_key=SENHA_OPEN_AI)


# ---------- Utilitários ----------
def compress_image(img: Image.Image, max_px=900, quality=75) -> bytes:
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()

def enhance_xray(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(1.15)
    return gray.convert("RGB")

def build_message(texto: str) -> dict:
    data_url = to_data_url(st.session_state.jpeg_bytes)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": texto},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
        ]
    }

# ---------- Prompt ----------
SYSTEM_PROMPT = (
    "Você é o EndoDetectBot, um assistente virtual especializado em radiologia odontológica. "
    "Sua tarefa é avaliar radiografias periapicais fornecidas pelo usuário e fornecer uma interpretação sugestiva com base em padrões radiográficos visuais. "
    "(1) 👁️‍🗨️ **Indício ou ausência de lesão periapical**; "
    "(2) 📊 **Classificação sugestiva da severidade** (leve, moderada ou severa); "
    "(3) 🧾 **Orientação clínica geral** com base em boas práticas. "
    "Use termos como 'sugere-se', 'indicativos de', evitando afirmações absolutas. "
    "Sempre verifique antes de analisar, se o dente a ser analisado aparece por completo o seu periápice."
    "A ponta do ápice radicular precisa aparecer na imagem, senão ela precisa ser reinserida pelo usuário"
    "Sempre forneça uma avaliação. Caso a imagem não esteja completa do ápice, solicite o reenvio da imagem, solicitando que o usuário aumente o campo selecionado durante o recorte da imagem."
    "Se o periápice não estiver visível, oriente o reenvio da imagem. "
    "Finalize com a ressalva de que o laudo é sugestivo, podendo conter erros, e que o diagnóstico definitivo deve ser feito por um cirurgião-dentista."
)

# ---------- Estado inicial ----------
st.session_state.setdefault("history", [{"role": "system", "content": SYSTEM_PROMPT}])
st.session_state.setdefault("jpeg_bytes", None)
st.session_state.setdefault("crop_orig", None)
st.session_state.setdefault("crop_enh", None)
st.session_state.setdefault("await_choice", False)
st.session_state.setdefault("laudo_pronto", False)

# ---------- Exibir logo ----------
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 10px;'>
        <img src='https://raw.githubusercontent.com/cristianomaraujo/endo_detect_lesion/e6fb421c7054094bff961079834e4c9828ddbde2/Logo.png' width='280'/>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Interface ----------
# (Removido título)

uploaded = st.file_uploader("Radiografia periapical (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ---------- Etapa A: recorte ----------
if uploaded and not st.session_state.await_choice and st.session_state.jpeg_bytes is None:
    img_orig = Image.open(uploaded).convert("RGB")
    st.write("🔍 Ajuste o retângulo vertical e clique em **Confirmar recorte**.")
    cropped_img = st_cropper(
        img_orig,
        aspect_ratio=(2, 3),
        box_color="#27AE60",
        realtime_update=True,
        return_type="image",
        key="cropper"
    )

    if st.button("Confirmar recorte"):
        st.session_state.crop_orig = cropped_img
        st.session_state.crop_enh = enhance_xray(cropped_img)
        st.session_state.await_choice = True
        st.experimental_rerun()

# ---------- Etapa B: escolha da imagem ----------
if st.session_state.await_choice:
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.crop_orig, caption="Original", use_column_width=True)
    with col2:
        st.image(st.session_state.crop_enh, caption="P&B + Contraste", use_column_width=True)

    choice = st.radio("Qual versão deseja enviar?", ("Original", "P&B + Contraste"), key="radio_choice")
    if st.button("Usar esta imagem"):
        final_img = st.session_state.crop_enh if choice == "P&B + Contraste" else st.session_state.crop_orig
        jpeg = compress_image(final_img)
        size_mb = len(jpeg) / (1024 * 1024)

        if size_mb > 4:
            st.error("❌ Arquivo > 4 MB. Refaça o recorte ou envie uma imagem menor.")
        else:
            st.session_state.jpeg_bytes = jpeg
            st.session_state.await_choice = False
            st.success("✅ Imagem salva! Agora clique em **Gerar avaliação**.")
            st.experimental_rerun()

# ---------- Gerar laudo ----------
if st.session_state.jpeg_bytes and not st.session_state.laudo_pronto:
    if st.button("Gerar laudo"):
        texto = "Por favor, avalie a radiografia."
        st.session_state.history.append(build_message(texto))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.history,
            max_tokens=700,
            temperature=0.2
        )
        st.session_state.history.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        st.session_state.laudo_pronto = True
        st.experimental_rerun()

# ---------- Renderização após laudo ----------
if st.session_state.laudo_pronto:
    for msg in st.session_state.history[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])
        else:
            if isinstance(msg["content"], list):
                with st.chat_message("user"):
                    st.write(msg["content"][0]["text"])
                    st.image(msg["content"][1]["image_url"]["url"], caption="Radiografia enviada")
            else:
                st.chat_message("user").write(msg["content"])

    # Campo de perguntas pós-laudo
    nova_pergunta = st.chat_input("❓ Tirar dúvidas sobre o laudo ou a lesão:")
    if nova_pergunta:
        st.session_state.history.append({"role": "user", "content": nova_pergunta})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.history,
            max_tokens=700,
            temperature=0.2
        )
        st.session_state.history.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        st.experimental_rerun()

    # Botão para reiniciar
    if st.button("📤 Enviar nova imagem"):
        for key in ["jpeg_bytes", "crop_orig", "crop_enh", "await_choice", "laudo_pronto"]:
            st.session_state[key] = None
        st.experimental_rerun()
