from __future__ import annotations

import io
import hashlib
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

st.set_page_config(page_title="SEIF PHOTO STUDIO PRO MAX", page_icon="📸", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:2rem; max-width:1500px;}
.hero {padding:1.1rem 1.4rem; border-radius:18px; background:linear-gradient(120deg,#171923,#34384a); color:#fff; margin-bottom:1rem;}
.hero h1 {margin:0;font-size:2rem;}
.hero p {margin:.45rem 0 0 0;color:#e2e4eb;}
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {border-radius:10px; min-height:2.6rem;}
.small-note {color:#777;font-size:.9rem;}
</style>
<div class="hero">
<h1>📸 SEIF PHOTO STUDIO PRO MAX</h1>
<p>تغيير الخلفيات • الصور الرسمية • Fond neutre • تعديل الإضاءة • تنزيل سريع</p>
</div>
""", unsafe_allow_html=True)

MODEL_OPTIONS = {
    "BirefNet Portrait — جودة أعلى للبورتريه": "birefnet-portrait",
    "U²-Net Human Seg — الأشخاص": "u2net_human_seg",
    "U²-Net — عام": "u2net",
}

NEUTRAL_COLORS = {
    "Blanc pur / أبيض": "#FFFFFF",
    "Gris clair / رمادي فاتح": "#D9D9D9",
    "Gris moyen / رمادي": "#92969C",
    "Gris foncé / رمادي داكن": "#50545B",
    "Beige / بيج": "#D9C5A3",
    "Ivoire / عاجي": "#F2EBDD",
    "Taupe / تاوب": "#8B8178",
    "Noir / أسود": "#202124",
    "Bleu doux / أزرق هادئ": "#AFC5D6",
    "Vert sauge / أخضر هادئ": "#B8C5B1",
}

@st.cache_resource(show_spinner=False)
def get_rembg_session(model_name):
    from rembg import new_session
    return new_session(model_name)

@st.cache_data(show_spinner=False, max_entries=8)
def remove_background_cached(image_bytes: bytes, model_name: str) -> bytes:
    from rembg import remove
    source = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result = remove(source, session=get_rembg_session(model_name),
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10)
    out = io.BytesIO()
    result.save(out, format="PNG", optimize=True)
    return out.getvalue()

def open_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    return im

def make_background(size, color_hex: str, gradient: bool = True) -> Image.Image:
    w, h = size
    rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
    if not gradient:
        return Image.new("RGB", (w, h), rgb)
    # Subtle studio gradient, brighter near the upper center.
    y, x = np.mgrid[0:h, 0:w]
    nx = (x - w * 0.5) / max(w * 0.75, 1)
    ny = (y - h * 0.35) / max(h * 0.8, 1)
    light = np.clip(1.07 - 0.16 * (nx * nx + ny * ny), 0.82, 1.07)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = np.clip(rgb[c] * light, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def composite_subject(subject: Image.Image, bg: Image.Image, scale: float, x_percent: int, y_percent: int, rotation: float) -> Image.Image:
    subject = subject.convert("RGBA")
    bg = bg.convert("RGBA")
    if rotation:
        subject = subject.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    target_h = max(1, int(bg.height * 0.78 * scale))
    ratio = target_h / max(subject.height, 1)
    target_w = max(1, int(subject.width * ratio))
    subject = subject.resize((target_w, target_h), Image.Resampling.LANCZOS)
    x = int((bg.width - target_w) * x_percent / 100)
    y = int((bg.height - target_h) * y_percent / 100)
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.alpha_composite(subject, (x, y))
    return Image.alpha_composite(bg, layer).convert("RGB")

def enhance_image(im: Image.Image, brightness: float, contrast: float, saturation: float, sharpness: float) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Color(im).enhance(saturation)
    im = ImageEnhance.Sharpness(im).enhance(sharpness)
    return im

if "original_bytes" not in st.session_state:
    st.session_state.original_bytes = None
if "cutout_bytes" not in st.session_state:
    st.session_state.cutout_bytes = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_result_key" not in st.session_state:
    st.session_state.last_result_key = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ أدوات الاستوديو")
    uploaded = st.file_uploader("1. ارفع الصورة", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        incoming = uploaded.getvalue()
        digest = hashlib.sha256(incoming).hexdigest()
        if st.session_state.get("image_digest") != digest:
            st.session_state.original_bytes = incoming
            st.session_state.image_digest = digest
            st.session_state.cutout_bytes = None
            st.session_state.last_result = None
            st.session_state.last_result_key = None
    model_label = st.selectbox("محرك قص الخلفية", list(MODEL_OPTIONS.keys()), index=0)
    model_name = MODEL_OPTIONS[model_label]
    st.caption("BirefNet Portrait قد يعطي نتائج أفضل للبورتريه، لكنه أثقل وقد يستغرق وقتًا أطول.")
    st.divider()
    st.subheader("Fond neutre")
    selected_color_name = st.selectbox("لون الخلفية", list(NEUTRAL_COLORS.keys()))
    selected_color = NEUTRAL_COLORS[selected_color_name]
    gradient = st.toggle("تدرّج استوديو ناعم", value=True)
    custom_bg_file = st.file_uploader("أو ارفع خلفية خاصة", type=["jpg", "jpeg", "png", "webp"], key="custom_bg")
    st.divider()
    st.subheader("تكوين الصورة")
    scale = st.slider("حجم الشخص", 0.45, 1.45, 1.0, 0.05)
    x_percent = st.slider("الموضع الأفقي", 0, 100, 50)
    y_percent = st.slider("الموضع العمودي", 0, 100, 13)
    rotation = st.slider("تدوير الصورة (درجة)", -180, 180, 0)
    st.caption("هذه أدوات التكوين الأولية؛ سحب مقابض الزوايا مباشرة بالفأرة غير مفعّل بعد.")
    st.divider()
    st.subheader("تحسين خفيف")
    brightness = st.slider("السطوع", 0.5, 1.5, 1.0, 0.05)
    contrast = st.slider("التباين", 0.5, 1.5, 1.0, 0.05)
    saturation = st.slider("تشبع الألوان", 0.0, 1.8, 1.0, 0.05)
    sharpness = st.slider("الحدة", 0.5, 2.0, 1.0, 0.1)

if not st.session_state.original_bytes:
    st.info("ابدأ برفع صورة من القائمة الجانبية. يمكنك تجربة الخلفيات المحايدة بعد رفع الصورة.")
    st.markdown("### مميزات النسخة الحالية")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🎨 Fond neutre**\n\nألوان بيضاء ورمادية وبيج وعاجية مع تدرج ناعم.")
    c2.markdown("**✂️ إزالة الخلفية**\n\nقص الشخص آليًا باستخدام نموذج rembg عند توفره.")
    c3.markdown("**⚡ تنزيل سريع**\n\nتخزين الصورة النهائية الجاهزة لتجنب إعادة المعالجة عند التنزيل.")
    st.stop()

try:
    original = open_image(st.session_state.original_bytes)
except Exception as exc:
    st.error(f"تعذّر فتح الصورة. جرّب ملفًا آخر. التفاصيل: {exc}")
    st.stop()

col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    st.subheader("الصورة الأصلية")
    st.image(original, use_container_width=True)

with col_b:
    st.subheader("النتيجة النهائية")
    remove_clicked = st.button("✂️ إزالة الخلفية بالذكاء الاصطناعي", type="primary", use_container_width=True)
    if remove_clicked:
        try:
            with st.spinner("يجري قصّ الشخص... قد تستغرق أول مرة وقتًا أطول لتحميل النموذج."):
                st.session_state.cutout_bytes = remove_background_cached(st.session_state.original_bytes, model_name)
                st.session_state.last_result = None
                st.session_state.last_result_key = None
            st.success("تم تجهيز قصّ الشخص. اختر الخلفية ثم نزّل النتيجة.")
        except Exception as exc:
            st.error("لم تنجح إزالة الخلفية. تأكد من تثبيت rembg/onnxruntime أو جرّب مجددًا.")
            st.exception(exc)

    if st.session_state.cutout_bytes:
        try:
            subject = open_image(st.session_state.cutout_bytes).convert("RGBA")
            if custom_bg_file is not None:
                bg = open_image(custom_bg_file.getvalue()).convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
            else:
                bg = make_background(original.size, selected_color, gradient)
            result = composite_subject(subject, bg, scale, x_percent, y_percent, rotation)
            result = enhance_image(result, brightness, contrast, saturation, sharpness)
            key_material = "|".join([hashlib.sha256(st.session_state.cutout_bytes).hexdigest(), selected_color, str(gradient), str(scale), str(x_percent), str(y_percent), str(rotation), str(brightness), str(contrast), str(saturation), str(sharpness), hashlib.sha256(custom_bg_file.getvalue()).hexdigest() if custom_bg_file else "default"])
            result_key = hashlib.sha256(key_material.encode()).hexdigest()
            # Prepare final bytes once per unique set of settings; download does not rerun AI.
            if st.session_state.last_result_key != result_key:
                buf = io.BytesIO()
                result.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
                st.session_state.last_result = buf.getvalue()
                st.session_state.last_result_key = result_key
            st.image(result, use_container_width=True)

            # Keep a short, deduplicated history of finished previews.
            history_key = hashlib.sha256(buf.getvalue()).hexdigest() if st.session_state.last_result else result_key
            if not st.session_state.history or st.session_state.history[-1]["key"] != result_key:
                st.session_state.history.append({
                    "key": result_key,
                    "bytes": st.session_state.last_result,
                    "label": f"تعديل {len(st.session_state.history) + 1}"
                })
                st.session_state.history = st.session_state.history[-20:]

            import base64
            preview64 = base64.b64encode(save_preview(result)).decode("ascii") if False else base64.b64encode(st.session_state.last_result).decode("ascii")
            st.components.v1.html(f"""
            <div style="font:14px Arial,sans-serif;color:#777;margin-bottom:6px">🔎 تكبير/تصغير بعجلة الفأرة، واسحب الصورة لتحريك المعاينة</div>
            <div id="zoom-stage" style="height:500px;overflow:hidden;border:1px solid #aaa;border-radius:12px;background:#222;display:flex;align-items:center;justify-content:center;cursor:grab;touch-action:none">
              <img id="zoom-image" src="data:image/jpeg;base64,{preview64}" style="max-width:90%;max-height:90%;object-fit:contain;transform:translate(0px,0px) scale(1);transform-origin:center;pointer-events:none;user-select:none">
            </div>
            <div style="margin-top:8px"><button onclick="window.resetZoomView()">إعادة العرض</button> <span id="zoom-value">100%</span></div>
            <script>(function(){{
              const s=document.getElementById('zoom-stage'),i=document.getElementById('zoom-image'),v=document.getElementById('zoom-value');
              let z=1,x=0,y=0,drag=false,sx=0,sy=0,ox=0,oy=0;
              function draw(){{i.style.transform='translate('+x+'px,'+y+'px) scale('+z+')';v.textContent=Math.round(z*100)+'%';}}
              s.addEventListener('wheel',e=>{{e.preventDefault();let r=s.getBoundingClientRect(),px=e.clientX-r.left-r.width/2,py=e.clientY-r.top-r.height/2,old=z;z=Math.max(.5,Math.min(5,z*(e.deltaY<0?1.12:.89)));x=px-(px-x)*z/old;y=py-(py-y)*z/old;draw();}},{{passive:false}});
              s.addEventListener('pointerdown',e=>{{drag=true;sx=e.clientX;sy=e.clientY;ox=x;oy=y;s.setPointerCapture(e.pointerId);s.style.cursor='grabbing';}});
              s.addEventListener('pointermove',e=>{{if(drag){{x=ox+e.clientX-sx;y=oy+e.clientY-sy;draw();}}}});
              s.addEventListener('pointerup',()=>{{drag=false;s.style.cursor='grab';}});
              window.resetZoomView=()=>{{z=1;x=0;y=0;draw();}};
            }})();</script>
            """, height=570, scrolling=False)

            if st.session_state.history:
                with st.expander("🕘 سجل آخر 20 نتيجة"):
                    selected_history = st.selectbox("اختر نتيجة محفوظة", range(len(st.session_state.history)),
                        format_func=lambda i: st.session_state.history[i]["label"], key="history_choice")
                    st.image(st.session_state.history[selected_history]["bytes"], caption="معاينة من السجل", use_container_width=True)
                    st.download_button("تنزيل النتيجة المختارة من السجل",
                        data=st.session_state.history[selected_history]["bytes"],
                        file_name="seif_photo_history.png", mime="image/png")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("⬇️ تنزيل JPEG عالي الجودة", data=st.session_state.last_result, file_name="seif_photo_studio_result.jpg", mime="image/jpeg", use_container_width=True)
            with d2:
                png_buf = io.BytesIO()
                result.save(png_buf, format="PNG", optimize=True)
                st.download_button("⬇️ تنزيل PNG", data=png_buf.getvalue(), file_name="seif_photo_studio_result.png", mime="image/png", use_container_width=True)
            st.caption("التكبير بالفأرة مخصص للمعاينة. التنزيل يحافظ على دقة الصورة الأصلية ولا يعيد تشغيل نموذج القص.")
        except Exception as exc:
            st.error("تعذّر تركيب الصورة. جرّب إعدادات أبسط أو صورة أخرى.")
            st.exception(exc)
    else:
        st.info("اضغط «إزالة الخلفية بالذكاء الاصطناعي» لبدء التحرير.")

st.divider()
st.subheader("ملاحظات")
st.write("- خلفيات Fond neutre في هذه النسخة تُنشأ محليًا من ألوان استوديو محايدة، ويمكنك رفع صورة خلفية حقيقية من جهازك.")
st.write("- محرك القص أصبح قابلًا للاختيار بين BirefNet Portrait وU²-Net Human Seg وU²-Net، مع تحسين حواف alpha matting.")
st.write("- التكبير بعجلة الفأرة والسحب متاحان في نافذة المعاينة؛ هذه الحركة لا تغيّر ملف التصدير نفسه.")
st.write("- أدوات التدوير والحجم والموضع تعمل حاليًا من عناصر التحكم؛ التدوير والسحب المباشر بمقابض الزوايا بالفأرة يحتاجان إلى مكوّن محرر تفاعلي مستقل.")
