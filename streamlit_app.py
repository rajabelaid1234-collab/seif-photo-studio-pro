
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageOps
from streamlit_drawable_canvas import st_canvas

def search_commons_backgrounds(query, limit=18):
    """Search Wikimedia Commons for reusable background photos and return license metadata."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'filetype:bitmap {query}',
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 500,
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "SeifPhotoStudio/1.0 (background gallery)"})
    with urllib.request.urlopen(request, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    results = []
    for page in sorted(pages.values(), key=lambda p: p.get("title", "")):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_name = meta.get("LicenseShortName", {}).get("value", "License not listed")
        artist = meta.get("Artist", {}).get("value", "")
        results.append({
            "title": page.get("title", "Background").replace("File:", ""),
            "preview": info.get("thumburl") or info.get("url"),
            "url": info.get("url"),
            "page": info.get("descriptionurl"),
            "license": license_name,
            "artist": artist,
        })
    return [item for item in results if item.get("url") and item.get("preview")]

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
    brush_edit,
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
        ["استوديو الصور", "صانع الصور الرسمية", "مكتبة الخلفيات", "مكتبتي", "حول الاستوديو"],
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

if page == "مكتبة الخلفيات":
    st.header("🖼️ مكتبة الخلفيات")
    st.write("تصفّح الخلفيات أولًا، حتى لو لم ترفع صورة بعد. اختر صورة لاحقًا من استوديو الصور لتطبيقها على الشخص.")
    category_labels = {
        "أعراس وقاعات زفاف": "wedding venue wedding hall interior",
        "بحر وشواطئ": "seascape beach coastline",
        "طبيعة وغابات": "forest landscape nature",
        "حدائق وأزهار": "flower garden botanical",
        "جبال وشلالات": "mountain waterfall landscape",
        "مدن ومعالم": "city architecture landscape",
        "خلفيات استوديو": "photography studio backdrop",
        "طراز تقليدي تونسي": "Tunisia traditional architecture courtyard",
        "غروب وشروق": "sunset sunrise landscape",
        "سماء وسحب": "cloudscape sky landscape",
    }
    chosen_category = st.selectbox(
        "اختر تصنيف الخلفيات",
        list(category_labels.keys()),
        key="standalone_commons_bg_category",
    )
    if st.button("🔎 عرض الصور", type="primary", key="standalone_commons_search"):
        st.session_state["standalone_background_search_category"] = chosen_category
        st.session_state.pop("standalone_background_results", None)
    active_category = st.session_state.get("standalone_background_search_category", chosen_category)
    try:
        with st.spinner("جارٍ تحميل صور المكتبة من الإنترنت..."):
            if "standalone_background_results" not in st.session_state:
                st.session_state["standalone_background_results"] = search_commons_backgrounds(
                    category_labels[active_category], limit=18
                )
        results = st.session_state["standalone_background_results"]
        if not results:
            st.info("لم تظهر صور لهذه الفئة. اختر تصنيفًا آخر واضغط «عرض الصور».")
        else:
            st.caption(f"تم العثور على {len(results)} صورة. اضغط على اسم الصورة لعرضها وتفاصيل ترخيصها.")
            chosen_image = st.selectbox(
                "اختر صورة لمعاينتها",
                results,
                format_func=lambda item: item["title"],
                key="standalone_commons_selected",
            )
            image_cols = st.columns(3)
            for idx, item in enumerate(results):
                with image_cols[idx % 3]:
                    st.image(item["preview"], caption=item["title"], use_container_width=True)
                    st.caption("الترخيص: " + item["license"])
                    if item.get("page"):
                        st.markdown(f"[المصدر والتفاصيل]({item['page']})")
            st.divider()
            st.subheader("معاينة الصورة المختارة")
            st.image(chosen_image["preview"], caption=chosen_image["title"], use_container_width=True)
            st.write("**الترخيص:** " + chosen_image["license"])
            if chosen_image.get("artist"):
                st.write("**المؤلف:** " + chosen_image["artist"])
            if chosen_image.get("page"):
                st.markdown(f"[افتح صفحة الصورة الأصلية]({chosen_image['page']})")
            if st.button("⬇️ تنزيل الخلفية المختارة", key="standalone_download_background"):
                try:
                    req = urllib.request.Request(chosen_image["url"], headers={"User-Agent": "SeifPhotoStudio/1.0"})
                    with urllib.request.urlopen(req, timeout=20) as response:
                        image_bytes = response.read()
                    st.download_button(
                        "حفظ الصورة على الجهاز",
                        data=image_bytes,
                        file_name=chosen_image["title"].replace("/", "_"),
                        mime="image/jpeg",
                        key="standalone_background_file_download",
                    )
                except Exception as exc:
                    st.error(f"تعذّر تنزيل الصورة: {exc}")
    except Exception as exc:
        st.error(f"تعذّر تحميل مكتبة الخلفيات من الإنترنت: {exc}")
        st.info("تحقق من اتصال التطبيق بالإنترنت ثم جرّب مرة أخرى.")
    st.stop()

if page == "مكتبتي":
    st.header("🗂️ مكتبتي")
    st.write("احفظ أكبر عدد ممكن من الصور والنتائج في مكتبتك أثناء جلسة العمل.")
    library = st.session_state["photo_library"]
    if not library:
        st.info("مكتبتك فارغة الآن. افتح استوديو الصور واحفظ صورة أو نتيجة فيها.")
    else:
        st.caption(f"عدد الصور المحفوظة حاليًا: {len(library)} — لا يوجد حدّ عددي مفروض داخل التطبيق؛ الحد العملي هو ذاكرة جلسة الاستضافة.")
        if st.button("🗑️ مسح المكتبة كاملة", key="clear_photo_library"):
            st.session_state["photo_library"] = []
            st.rerun()
        for i, item in enumerate(library):
            with st.container(border=True):
                col_img, col_actions = st.columns([1, 2])
                with col_img:
                    try:
                        st.image(Image.open(io.BytesIO(item["data"])), caption=item["filename"], use_container_width=True)
                    except Exception:
                        st.caption(item["filename"])
                with col_actions:
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
    st.warning("مهم: المكتبة الحالية تحفظ الصور داخل جلسة Streamlit فقط. لا يوجد حدّ ثابت لعدد الصور في الكود، لكن الصور الكبيرة تستهلك ذاكرة الخادم، وقد تختفي عند انتهاء الجلسة أو إعادة تشغيل التطبيق. لجعلها دائمة عبر تسجيل الدخول نحتاج إلى تخزين سحابي آمن.")
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
         "خلفية متدرجة", "صورة من جهازي", "📚 مكتبة خلفيات الإنترنت"],
        horizontal=True,
    )

    background_colors = {
        "أبيض": "#FFFFFF",
        "رمادي فاتح": "#D9D9D9",
        "رمادي داكن": "#666666",
    }

    custom_background = None
    selected_commons_background = None
    if background_choice == "📚 مكتبة خلفيات الإنترنت":
        st.markdown("### 📚 مكتبة خلفيات جاهزة")
        st.caption("اختر فئة ثم خلفية. الصور تأتي من Wikimedia Commons وتظهر معها معلومات الترخيص والمصدر.")
        category_labels = {
            "أعراس وقاعات زفاف": "wedding venue wedding hall interior",
            "بحر وشواطئ": "seascape beach coastline",
            "طبيعة وغابات": "forest landscape nature",
            "حدائق وأزهار": "flower garden botanical",
            "جبال وشلالات": "mountain waterfall landscape",
            "مدن ومعالم": "city architecture landscape",
            "خلفيات استوديو": "photography studio backdrop",
            "طراز تقليدي تونسي": "Tunisia traditional architecture courtyard",
            "غروب وشروق": "sunset sunrise landscape",
            "سماء وسحب": "cloudscape sky landscape",
        }
        chosen_category = st.selectbox("تصنيف الخلفيات", list(category_labels.keys()), key="commons_bg_category")
        try:
            with st.spinner("جارٍ البحث عن خلفيات مناسبة..."):
                background_results = search_commons_backgrounds(category_labels[chosen_category])
            if not background_results:
                st.info("لم تظهر نتائج لهذه الفئة. جرّب فئة أخرى.")
            else:
                selected_commons_background = st.selectbox(
                    "اختر الخلفية بالاسم",
                    background_results,
                    format_func=lambda item: item["title"],
                    key="commons_bg_selected",
                )
                preview_col, details_col = st.columns([1, 1])
                with preview_col:
                    st.image(selected_commons_background["preview"], caption=selected_commons_background["title"], use_container_width=True)
                with details_col:
                    st.write("**الترخيص:** " + selected_commons_background["license"])
                    if selected_commons_background.get("artist"):
                        st.markdown("**المؤلف:** " + selected_commons_background["artist"], unsafe_allow_html=True)
                    if selected_commons_background.get("page"):
                        st.markdown(f'[صفحة الصورة والمصدر]({selected_commons_background["page"]})')
                if st.button("⬇️ تجهيز الخلفية المختارة", key="load_commons_background"):
                    try:
                        request = urllib.request.Request(
                            selected_commons_background["url"],
                            headers={"User-Agent": "SeifPhotoStudio/1.0"},
                        )
                        with urllib.request.urlopen(request, timeout=20) as response:
                            custom_background = load_image(io.BytesIO(response.read())).convert("RGB")
                        st.session_state["selected_online_background"] = custom_background
                        st.session_state["selected_online_background_title"] = selected_commons_background["title"]
                        st.success("تم تجهيز الخلفية. اضغط «معالجة الصورة» لتطبيقها.")
                    except Exception as exc:
                        st.error(f"تعذّر تنزيل الخلفية: {exc}")
                custom_background = st.session_state.get("selected_online_background")
        except Exception as exc:
            st.warning(f"تعذّر الاتصال بمكتبة الخلفيات الآن. تحقق من اتصال الموقع بالإنترنت ثم أعد المحاولة. التفاصيل: {exc}")
            custom_background = st.session_state.get("selected_online_background")
    elif background_choice == "خلفية متدرجة":
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
                        st.error("اختر صورة للخلفية أولًا، ثم اضغط «تجهيز الخلفية المختارة» إذا كانت من المكتبة.")
                        st.stop()
                    bg = custom_background.resize(
                        cutout.size, Image.Resampling.LANCZOS
                    )

                st.session_state["studio_cutout"] = cutout.copy()
                st.session_state["studio_background"] = bg.copy()
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

        # فرشاة يدوية حقيقية: يرسم المستخدم فوق معاينة الشخص، ثم نطبّق القناع على الصورة الأصلية.
        with st.expander("🖌️ مختبر إصلاح القصّ — استرجاع أو محو أجزاء"):
            editable = st.session_state.get("studio_cutout")
            if editable is None:
                st.info("عالج الصورة أولًا حتى تظهر أداة الفرشاة.")
            else:
                brush_mode = st.radio(
                    "وظيفة الفرشاة", ["استرجاع الجزء من الصورة الأصلية", "محو الجزء وجعله شفافًا"],
                    horizontal=True, key="brush_mode"
                )
                brush_size = st.slider("حجم الفرشاة", 4, 80, 24, key="brush_size")
                max_w = min(700, editable.width)
                canvas_h = max(200, int(editable.height * max_w / editable.width))
                preview_bg = composite_background(
                    editable, create_solid_background(editable.size, "#777777")
                ).convert("RGB").resize((max_w, canvas_h), Image.Resampling.LANCZOS)
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 255, 0)", stroke_width=brush_size,
                    stroke_color="#ff00ff", background_image=preview_bg,
                    update_streamlit=True, height=canvas_h, width=max_w,
                    drawing_mode="freedraw", key="mask_repair_canvas"
                )
                if st.button("✅ تطبيق الفرشاة", key="apply_brush_edit"):
                    if canvas_result.image_data is None:
                        st.warning("ارسم فوق المنطقة التي تريد تعديلها أولًا.")
                    else:
                        drawn = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
                        baseline = preview_bg.convert("RGB")
                        difference = ImageChops.difference(drawn, baseline).convert("L")
                        stroke_mask = difference.point(lambda p: 255 if p > 35 else 0)
                        if stroke_mask.getbbox() is None:
                            st.warning("لم يتم اكتشاف أي ضربة فرشاة. ارسم ثم أعد المحاولة.")
                        else:
                            stroke_mask = stroke_mask.resize(editable.size, Image.Resampling.NEAREST)
                            edited = brush_edit(
                                editable, original, stroke_mask,
                                restore=(brush_mode == "استرجاع الجزء من الصورة الأصلية")
                            )
                            st.session_state["studio_cutout"] = edited
                            bg_saved = st.session_state.get("studio_background")
                            if bg_saved is not None:
                                st.session_state["studio_result"] = composite_background(edited, bg_saved)
                            st.rerun()

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
