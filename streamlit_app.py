
from __future__ import annotations

import io
import streamlit as st
from PIL import Image, ImageOps

from image_processing import (
    load_image,
    remove_background,
    composite_background,
    enhance_image,
    fix_lighting,
    official_crop,
    build_official_photo,
    create_solid_background,
    create_gradient_background,
    image_to_bytes,
    OFFICIAL_SIZES,
)

st.set_page_config(
    page_title="SEIF PHOTO STUDIO PRO MAX",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101114, #191b20);
    color: #f7f7f7;
}
h1, h2, h3 { color: #d8b56a !important; }
section[data-testid="stSidebar"] {
    background: #111216;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}
div.stButton > button[kind="primary"] {
    background: #c6a45d;
    color: #111111;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("📸 SEIF PHOTO STUDIO")
st.caption("PRO MAX • Professional Photo Background Editor")

with st.sidebar:
    st.markdown("## ⚙️ لوحة التحكم")
    page = st.radio(
        "القسم",
        ["استوديو الصور", "صانع الصور الرسمية", "مكتبتي", "حول الاستوديو"],
    )
    st.divider()
    st.caption("SEIF PHOTO STUDIO PRO MAX")

if "photo_library" not in st.session_state:
    st.session_state["photo_library"] = []

def add_to_library(image, filename, fmt="PNG"):
    data = image_to_bytes(image, fmt)
    st.session_state["photo_library"].append({
        "filename": filename,
        "format": fmt,
        "data": data,
    })

if page == "مكتبتي":
    st.header("🗂️ مكتبتي")
    st.write("احفظ الصور الأصلية والنتائج هنا للوصول إليها خلال جلسة العمل الحالية.")
    library = st.session_state["photo_library"]
    if not library:
        st.info("مكتبتك فارغة الآن. افتح استوديو الصور واحفظ صورة أو نتيجة فيها.")
    else:
        st.caption(f"عدد الصور المحفوظة: {len(library)}")
        for i, item in enumerate(library):
            with st.container(border=True):
                st.write(f"**{item['filename']}**")
                st.download_button(
                    "⬇️ تنزيل",
                    data=item["data"],
                    file_name=item["filename"],
                    mime="image/png" if item["format"] == "PNG" else "image/jpeg",
                    key=f"library_download_{i}",
                )
                if st.button("🗑️ حذف من المكتبة", key=f"library_delete_{i}"):
                    st.session_state["photo_library"].pop(i)
                    st.rerun()
    st.warning("مهم: هذه نسخة أولية من المكتبة تحفظ الصور داخل جلسة Streamlit فقط. لا تعتبر تخزينًا دائمًا؛ التخزين الدائم لكل حساب يحتاج قاعدة بيانات وتخزينًا سحابيًا آمنًا.")
    st.stop()

if page == "حول الاستوديو":
    st.header("عن الاستوديو")
    st.write(
        "استوديو لتعديل الصور، تغيير الخلفيات، "
        "تحسين الإضاءة وإنشاء الصور الرسمية."
    )
    st.info(
        "إزالة الخلفية تعتمد على نماذج rembg وقد تحتاج "
        "إلى تنزيل النموذج عند أول استعمال."
    )
    st.stop()

uploaded = st.file_uploader(
    "📂 اختر صورة من جهازك",
    type=["png", "jpg", "jpeg", "webp"],
    key="main_photo_upload",
)

if uploaded is None:
    st.markdown("### مرحبًا بك في استوديو الصور")
    st.write(
        "ارفع صورة للبدء. حافظ على الصورة الأصلية "
        "إذا كنت تريد الرجوع إليها لاحقًا."
    )
    st.stop()

try:
    original = load_image(io.BytesIO(uploaded.getvalue()))
except Exception as exc:
    st.error(f"تعذّر فتح الصورة: {exc}")
    st.stop()

st.success(f"تم تحميل الصورة: {uploaded.name}")

if st.button("📥 حفظ الصورة الأصلية في مكتبتي", key="save_original_to_library"):
    add_to_library(original, f"original_{uploaded.name.rsplit('.', 1)[0]}.png", "PNG")
    st.success("تم حفظ الصورة الأصلية في مكتبتك لهذه الجلسة.")

if page == "صانع الصور الرسمية":
    st.header("🪪 صانع الصور الرسمية")

    size_name = st.selectbox(
        "مقاس الصورة",
        list(OFFICIAL_SIZES.keys()),
    )
    bg_color = st.selectbox(
        "لون الخلفية",
        ["أبيض", "رمادي فاتح", "رمادي داكن", "أزرق فاتح"],
    )
    colors = {
        "أبيض": "#FFFFFF",
        "رمادي فاتح": "#D9D9D9",
        "رمادي داكن": "#777777",
        "أزرق فاتح": "#DCEAF7",
    }

    brightness = st.slider(
        "تصحيح الإضاءة",
        min_value=0.70,
        max_value=1.30,
        value=1.0,
        step=0.05,
    )

    st.warning(
        "للحصول على خلفية موحّدة، يجب أولًا قصّ الشخص "
        "من الخلفية الأصلية."
    )

    if st.button("✂️ إزالة الخلفية وإنشاء الصورة الرسمية",
                 type="primary"):
        with st.spinner("جارٍ قص الشخص ومعالجة الصورة..."):
            try:
                cutout = remove_background(original)
                result = build_official_photo(
                    cutout,
                    background_color=colors[bg_color],
                    size=OFFICIAL_SIZES[size_name],
                    brightness=brightness,
                )
                st.session_state["official_result"] = result
            except Exception as exc:
                st.error(
                    "تعذّرت معالجة الصورة. تحقق من تثبيت "
                    f"المكتبات وتنزيل النموذج. التفاصيل: {exc}"
                )

    result = st.session_state.get("official_result")
    if result is not None:
        st.subheader("النتيجة")
        st.image(result, caption="الصورة الرسمية", width="stretch")
        if st.button("🗂️ حفظ الصورة الرسمية في مكتبتي", key="save_official_to_library"):
            add_to_library(result, "seif_official_photo.png", "PNG")
            st.success("تم حفظ الصورة الرسمية في مكتبتك لهذه الجلسة.")

        st.download_button(
            "⬇️ تنزيل الصورة الرسمية",
            data=image_to_bytes(result, "JPEG"),
            file_name="seif_official_photo.jpg",
            mime="image/jpeg",
            key="download_official",
        )

else:
    st.header("🎨 استوديو تغيير الخلفية")

    left, right = st.columns(2)
    with left:
        st.subheader("الصورة الأصلية")
        st.image(original, width="stretch")

    with st.expander("✂️ إعدادات القصّ الرسمي"):
        use_crop = st.checkbox("تطبيق قصّ بنسبة صورة رسمية")
        crop_size = st.selectbox(
            "المقاس",
            list(OFFICIAL_SIZES.keys()),
            key="studio_crop_size",
        )

    with st.expander("💡 تحسين الإضاءة والألوان"):
        brightness = st.slider(
            "الإضاءة", 0.5, 1.5, 1.0, 0.05
        )
        contrast = st.slider(
            "التباين", 0.5, 1.5, 1.0, 0.05
        )
        saturation = st.slider(
            "تشبع الألوان", 0.0, 1.5, 1.0, 0.05
        )
        sharpness = st.slider(
            "حدة الصورة", 0.0, 2.0, 1.0, 0.1
        )

    st.subheader("🖼️ الخلفية")
    background_choice = st.radio(
        "اختر نوع الخلفية",
        ["أبيض", "رمادي فاتح", "رمادي داكن",
         "خلفية متدرجة", "صورة من جهازي"],
        horizontal=True,
    )

    background_colors = {
        "أبيض": "#FFFFFF",
        "رمادي فاتح": "#D9D9D9",
        "رمادي داكن": "#666666",
    }

    custom_background = None
    if background_choice == "خلفية متدرجة":
        col1, col2 = st.columns(2)
        with col1:
            color1 = st.color_picker(
                "اللون الأول", "#FFFFFF"
            )
        with col2:
            color2 = st.color_picker(
                "اللون الثاني", "#D9DDE3"
            )
    elif background_choice == "صورة من جهازي":
        bg_upload = st.file_uploader(
            "اختر صورة الخلفية",
            type=["png", "jpg", "jpeg", "webp"],
            key="background_upload",
        )
        if bg_upload:
            try:
                custom_background = load_image(
                    io.BytesIO(bg_upload.getvalue())
                )
                st.image(
                    custom_background,
                    caption="الخلفية المختارة",
                    width="stretch",
                )
            except Exception as exc:
                st.error(f"تعذّر فتح الخلفية: {exc}")

    if st.button("✨ معالجة الصورة", type="primary"):
        with st.spinner(
            "جارٍ قصّ الشخص وتجهيز الصورة..."
        ):
            try:
                # إزالة الخلفية الأصلية من الشخص
                cutout = remove_background(original)

                # تحسين الصورة مع المحافظة على الملامح
                cutout = enhance_image(
                    cutout,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    sharpness=sharpness,
                )

                if use_crop:
                    cutout = official_crop(
                        cutout,
                        size=OFFICIAL_SIZES[crop_size],
                    )

                # تجهيز الخلفية المختارة
                if background_choice in background_colors:
                    bg = create_solid_background(
                        cutout.size,
                        background_colors[background_choice],
                    )
                elif background_choice == "خلفية متدرجة":
                    bg = create_gradient_background(
                        cutout.size, color1, color2
                    )
                else:
                    if custom_background is None:
                        st.error("اختر صورة للخلفية أولًا.")
                        st.stop()
                    bg = custom_background.resize(
                        cutout.size, Image.Resampling.LANCZOS
                    )

                result = composite_background(cutout, bg)
                st.session_state["studio_result"] = result

            except Exception as exc:
                st.error(
                    "حدث خطأ أثناء المعالجة. قد يحتاج نموذج "
                    "إزالة الخلفية إلى التنزيل. "
                    f"التفاصيل: {exc}"
                )

    result = st.session_state.get("studio_result")

    if result is not None:
        st.divider()
        st.subheader("✅ مقارنة النتيجة")
        before, after = st.columns(2)

        with before:
            st.caption("قبل التعديل")
            st.image(original, width="stretch")

        with after:
            st.caption("بعد التعديل")
            st.image(result, width="stretch")

        fmt = st.radio(
            "صيغة التنزيل",
            ["PNG", "JPEG"],
            horizontal=True,
        )
        file_bytes = image_to_bytes(result, fmt)

        if st.button("🗂️ حفظ النتيجة في مكتبتي", key="save_studio_to_library"):
            add_to_library(result, "seif_photo_final.png", "PNG")
            st.success("تم حفظ النتيجة في مكتبتك لهذه الجلسة.")

        st.download_button(
            "⬇️ تنزيل الصورة النهائية",
            data=file_bytes,
            file_name=(
                "seif_photo_final.png"
                if fmt == "PNG"
                else "seif_photo_final.jpg"
            ),
            mime=(
                "image/png"
                if fmt == "PNG"
                else "image/jpeg"
            ),
            key="download_studio_result",
        )

st.divider()
st.caption("SEIF PHOTO STUDIO PRO MAX ©")
