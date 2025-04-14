import os, io, base64
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageDraw
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Carrega a variável da API
load_dotenv()
SENHA_OPEN_AI = os.getenv("SENHA_OPEN_AI") or st.secrets["SENHA_OPEN_AI"]

# Instancia o cliente OpenAI corretamente
client = OpenAI(api_key=SENHA_OPEN_AI)


# ---------- Webcam ----------
class CameraTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

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

def enhance_clahe(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    return Image.fromarray(cl).convert("RGB")

def enhance_laplacian(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return Image.fromarray(lap).convert("RGB")

def enhance_sobel(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    return Image.fromarray(sobel_combined).convert("RGB")

def draw_custom_box(img: Image.Image, box_coords, color="red") -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle(box_coords, outline=color, width=4)
    return img

def build_message(texto: str) -> list:
    user_data_url = to_data_url(st.session_state.jpeg_bytes)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": texto},
                {"type": "image_url", "image_url": {"url": user_data_url, "detail": "auto"}}
            ]
        }
    ]

# ---------- Prompt ----------
SYSTEM_PROMPT = (
    "Você é o EndoDetectBot, um assistente virtual especializado em radiologia odontológica. "
    "Sua tarefa é avaliar radiografias periapicais fornecidas pelo usuário e fornecer uma interpretação sugestiva com base em padrões radiográficos visuais. "
    "A imagem recebida é composta por quatro variações da mesma radiografia, dispostas lado a lado: "
    "(1) imagem original, "
    "(2) versão com realce adaptativo por CLAHE, "
    "(3) versão com realce de bordas (Sobel), "
    "(4) versão com filtro de textura (Gabor). "
    "Utilize essas versões de forma complementar para auxiliar na detecção de sinais sutis de lesão periapical. "
    "A área delimitada por um retângulo vermelho indica a região de interesse selecionada manualmente pelo usuário, geralmente envolvendo o periápice do dente. Concentre sua avaliação nessa área. "
    "Sua resposta deve conter: "
    "\n🦷**(1) Classificação Final:** indique se é normal, espessamento (leve ou moderado) ou lesão periapical. "
    "\n🔍**(2) Grau de confiança:** alto, moderado ou baixo, com justificativa. "
    "\n🧾**(3) Descrição da imagem:** localização e características observadas. "
    "\n⚠️**(4) Achados adicionais (se houver):** como reabsorções, fraturas, extravasamento de material. "
    "\n✅**(5) Orientação clínica:** recomendação baseada no achado. Sempre reforçando a necessidade da avaliação por um endodontista "
    "Utilize linguagem técnica (por exemplo: 'sugere-se', 'há indicativos de') e evite afirmações absolutas. "
    "Mesmo que a imagem não esteja perfeita, tente fornecer uma avaliação sugestiva com base nas evidências visuais disponíveis. "
    "Finalize com a ressalva de que a análise é sugestiva, feita por uma inteligência artificial, podendo conter erros, "
    "e que o diagnóstico definitivo deve ser realizado por um cirurgião-dentista após avaliação clínica completa."
    "Lembre sempre dos exemplos que vou te passar para definir a resposta final, entre presença de lesão, espessamento (mesmo que leve) ou normalidade."
    "Lesão periapical apresenta área radiolúcida bem delimitada na região apical; espessamento do ligamento periodontal mostra alargamento linear do espaço periodontal sem destruição óssea; já a normalidade do ligamento exibe espaço periodontal fino, contínuo e uniforme ao redor da raiz."
    "A avaliação deve compreender a avaliação do ápice dentário (no caso de lesão periapical é onde estará localizada), e também de todo ligamento (para diferenciar espessamento de normalidade)."
    "Você deve avaliar o ápice primeiro, para depois avaliar o restante do ligamento, antes da tomada de decisão."
    "Realize uma análise reflexiva, como se você estivesse avaliando essa imagem três vezes em momentos diferentes. Reflita brevemente sobre possíveis variações nas interpretações, depois integre as conclusões em um único parecer com grau de confiança."
)

EXEMPLOS_FEWSHOT = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Exemplo de lesão periapical: radiolucidez apical com margens mal definidas, geralmente associada à necrose pulpar.Se houver presença de rarefação na raiz, é uma provável lesão periapical. Estou enviando 6 exemplos, eles estão demarcados no retângulo vermelho. Aprenda esse padrão"},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/cristianomaraujo/endo_detect_lesion/main/exemplo_lesao.jpg", "detail": "low"}}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Exemplos de espessamento do ligamento periodontal: faixa radiolúcida fina e uniforme, com margens definidas, podendo estar relacionada a trauma oclusal ou mobilidade dentária. Espaço do ligamento aumentado, mas sem rarefação, é indício de espessamento. Só fale que é espessamento, somente se o espaço estiver aumentado, caso contrário, indique normalidade. Estou enviando 5 exemplos, eles estão demarcados no retângulo vermelho. Aprenda esse padrão"},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/cristianomaraujo/endo_detect_lesion/main/exemplo_espessamento.jpg", "detail": "low"}}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Exemplo de estrutura normal: espaço do ligamento periodontal bem definido, de espessura regular e sem radiolucidez patológica. Estou enviando 6 exemplos, eles estão demarcados no retângulo vermelho. Aprenda esse padrão"},
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/cristianomaraujo/endo_detect_lesion/main/exemplo_normal.jpg", "detail": "low"}}
        ]
    }
]

# ---------- Estado inicial ----------
st.session_state.setdefault("history", [{"role": "system", "content": SYSTEM_PROMPT}])
st.session_state.setdefault("jpeg_bytes", None)
st.session_state.setdefault("crop_image", None)
st.session_state.setdefault("canvas_box", None)
st.session_state.setdefault("laudo_pronto", False)

# ---------- Logo ----------
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 10px;'>
        <img src='https://raw.githubusercontent.com/cristianomaraujo/endo_detect_lesion/e6fb421c7054094bff961079834e4c9828ddbde2/Logo.png' width='280'/>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Câmera ----------
st.subheader("📷 Tirar foto com a câmera")
camera_ctx = webrtc_streamer(
    key="camera",
    video_transformer_factory=CameraTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)

if camera_ctx and camera_ctx.video_transformer:
    frame = camera_ctx.video_transformer.frame
    if frame is not None:
        st.image(frame, caption="Imagem capturada", channels="BGR")
        if st.button("📸 Usar esta imagem"):
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.session_state.crop_image = img_pil
            st.experimental_rerun()

# ---------- Upload ----------
uploaded = st.file_uploader("📁 Ou envie uma imagem de radiografia (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded and st.session_state.jpeg_bytes is None:
    img_orig = Image.open(uploaded).convert("RGB")
    st.write("🔍 Ajuste o recorte ao dente de interesse e clique em **Confirmar recorte**.")
    cropped_img = st_cropper(img_orig, aspect_ratio=(2, 3), box_color="#27AE60", realtime_update=True, return_type="image", key="cropper")
    if st.button("Confirmar recorte"):
        st.session_state.crop_image = cropped_img
        st.experimental_rerun()

if st.session_state.crop_image and st.session_state.canvas_box is None:
    st.subheader("🦷 Marque a região do periápice")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=st.session_state.crop_image,
        update_streamlit=True,
        height=st.session_state.crop_image.height,
        width=st.session_state.crop_image.width,
        drawing_mode="rect",
        key="canvas"
    )
    if st.button("Usar imagem com demarcação"):
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            obj = canvas_result.json_data["objects"][0]
            x, y, w, h = obj["left"], obj["top"], obj["width"], obj["height"]
            box_coords = [(x, y), (x + w, y + h)]
            clahe_img = enhance_clahe(st.session_state.crop_image.copy())
            lap_img = enhance_laplacian(st.session_state.crop_image.copy())
            sobel_img = enhance_sobel(st.session_state.crop_image.copy())
            orig_boxed = draw_custom_box(st.session_state.crop_image.copy(), box_coords, color="red")
            lap_boxed = draw_custom_box(lap_img.copy(), box_coords, color="yellow")
            clahe_boxed = draw_custom_box(clahe_img.copy(), box_coords, color="blue")
            sobel_boxed = draw_custom_box(sobel_img.copy(), box_coords, color="green")
            total_width = orig_boxed.width * 4
            new_img = Image.new("RGB", (total_width, orig_boxed.height))
            new_img.paste(orig_boxed, (0, 0))
            new_img.paste(lap_boxed, (orig_boxed.width, 0))
            new_img.paste(clahe_boxed, (orig_boxed.width * 2, 0))
            new_img.paste(sobel_boxed, (orig_boxed.width * 3, 0))
            st.session_state.jpeg_bytes = compress_image(new_img)
            st.session_state.canvas_box = new_img
            st.success("✅ Imagem salva! Agora clique em **Gerar laudo**.")
        else:
            st.warning("⚠️ Marque uma região antes de continuar.")

# ---------- Gerar laudo ----------
if st.session_state.jpeg_bytes and not st.session_state.laudo_pronto:
    if st.button("Gerar laudo"):
        texto = "Por favor, avalie a radiografia. Siga a estrutura solicitada, incluindo grau de confiança e achados adicionais."
        mensagens = [st.session_state.history[0], *EXEMPLOS_FEWSHOT, *build_message(texto)]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=mensagens,
            max_tokens=700,
            temperature=0.2
        )
        st.session_state.history.append(build_message(texto)[0])
        st.session_state.history.append({"role": "assistant", "content": response.choices[0].message.content})
        st.session_state.laudo_pronto = True
        st.experimental_rerun()

# ---------- Exibição do laudo ----------
if st.session_state.laudo_pronto:
    for msg in st.session_state.history[1:]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])
        elif isinstance(msg["content"], list):
            with st.chat_message("user"):
                for part in msg["content"]:
                    if part["type"] == "text":
                        st.markdown(part["text"])
                    elif part["type"] == "image_url" and "exemplo_" not in part["image_url"]["url"]:
                        st.image(part["image_url"]["url"], caption="Radiografia enviada")

    nova_pergunta = st.chat_input("❓ Tirar dúvidas sobre o laudo ou a lesão:")
    if nova_pergunta:
        st.session_state.history.append({"role": "user", "content": nova_pergunta})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.history,
            max_tokens=700,
            temperature=0.2
        )
        st.session_state.history.append({"role": "assistant", "content": response.choices[0].message.content})
        st.experimental_rerun()

    if st.button("📤 Enviar nova imagem"):
        for key in ["jpeg_bytes", "crop_image", "canvas_box", "laudo_pronto"]:
            st.session_state[key] = None
        st.experimental_rerun()
