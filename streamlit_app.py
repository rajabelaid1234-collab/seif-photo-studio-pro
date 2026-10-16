from __future__ import annotations
import io, hashlib, json, base64
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

st.set_page_config(page_title="SEIF PHOTO STUDIO PRO MAX", page_icon="📸", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1600px}
.hero{padding:1.2rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#111827,#41465a);color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.45rem 0 0;color:#e4e7ef}
div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{border-radius:10px;min-height:2.55rem}
section[data-testid="stSidebar"]{border-left:1px solid #ddd;border-right:0;order:2;min-width:285px!important;width:285px!important} div[data-testid="stAppViewContainer"] > .main{order:1}
/* Medium-sized, readable navigation */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{font-size:1.25rem!important;margin-bottom:.8rem!important}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:.45rem!important}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:.55rem .65rem!important;min-height:2.7rem!important;border-radius:9px!important}
section[data-testid="stSidebar"] div[role="radiogroup"] label p{font-size:1.05rem!important;line-height:1.35!important}
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown p{font-size:1rem!important}
</style>
<div class="hero"><h1>📸 SEIF PHOTO STUDIO PRO MAX</h1><p>استوديو الصور • مكتبة الخلفيات • الصور الرسمية • سجل المشاريع</p></div>
""",unsafe_allow_html=True)

ROOT=Path(__file__).parent
BG_ROOT=ROOT/"backgrounds"
PROJECTS=ROOT/"saved_projects"
MODEL_OPTIONS={"BirefNet Portrait — بورتريه":"birefnet-portrait","U²-Net Human Seg — أشخاص":"u2net_human_seg","U²-Net — عام":"u2net"}
NEUTRAL={"أبيض نقي":"#FFFFFF","رمادي فاتح":"#D9D9D9","رمادي متوسط":"#92969C","رمادي داكن":"#50545B","بيج":"#D9C5A3","عاجي":"#F2EBDD","تاوب":"#8B8178","أسود":"#202124","أزرق هادئ":"#AFC5D6","أخضر sage":"#B8C5B1"}
CATEGORIES={"Fond neutre":["neutral","gray","white"],"أعراس":["wedding"],"تقليدي":["traditional"],"فاخر":["luxury"],"طبيعة":["nature"]}
# Direct image URLs are optional online examples. Local files are preferred for reliable production use.
ONLINE_BACKGROUNDS=[
("قاعة أعراس ذهبية","أعراس","https://images.unsplash.com/photo-1519741497674-611481863552?auto=format&fit=crop&w=1800&q=88"),
("ديكور زفاف أنيق","أعراس","https://images.unsplash.com/photo-1519225421980-888cb3e8ba2b?auto=format&fit=crop&w=1800&q=88"),
("حفل زفاف خارجي","أعراس","https://images.unsplash.com/photo-1511285560929-80b456fea0bc?auto=format&fit=crop&w=1800&q=88"),
("ورود وديكور زفاف","أعراس","https://images.unsplash.com/photo-1464366400600-7168b8af9bc3?auto=format&fit=crop&w=1800&q=88"),
("ممر زفاف","أعراس","https://images.unsplash.com/photo-1515934751635-c81c6bc9a2d8?auto=format&fit=crop&w=1800&q=88"),
("طبيعة خضراء","طبيعة","https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1800&q=88"),
("حديقة مضيئة","طبيعة","https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?auto=format&fit=crop&w=1800&q=88"),
("غابة هادئة","طبيعة","https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=1800&q=88"),
("ديكور فاخر","فاخر","https://images.unsplash.com/photo-1616486338812-3dadae4b4ace?auto=format&fit=crop&w=1800&q=88"),
("صالون أنيق","فاخر","https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?auto=format&fit=crop&w=1800&q=88"),
("جدار محايد","Fond neutre","https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?auto=format&fit=crop&w=1800&q=88"),
("ديكور بيج","Fond neutre","https://images.unsplash.com/photo-1616486029423-aaa4789e8c9a?auto=format&fit=crop&w=1800&q=88"),
]
# Add the remaining library slots as clear prompts for the user to add licensed photos locally;
# never pretend placeholder slots are real photos.
LIBRARY_TARGETS={"Fond neutre":10,"أعراس":20,"تقليدي":15,"فاخر":15,"طبيعة":10}

@st.cache_resource(show_spinner=False)
def get_session(model):
    from rembg import new_session
    return new_session(model)

@st.cache_data(show_spinner=False,max_entries=10)
def cutout_cached(raw,model):
    from rembg import remove
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGBA")
    # Alpha matting improves edge detail but can be costly on low-memory hosts.
    result=remove(im,session=get_session(model),alpha_matting=True,alpha_matting_foreground_threshold=240,alpha_matting_background_threshold=10,alpha_matting_erode_size=10)
    out=io.BytesIO(); result.save(out,format="PNG",optimize=True); return out.getvalue()

@st.cache_data(show_spinner=False,max_entries=12)
def download_online_image(url):
    import requests
    response=requests.get(url,timeout=25,headers={"User-Agent":"SEIF-PHOTO-STUDIO/1.0"})
    response.raise_for_status()
    return response.content

def pil_open(data):
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")

def background_color(size,hex_color,gradient=True):
    w,h=size; rgb=tuple(int(hex_color[i:i+2],16) for i in (1,3,5))
    if not gradient:return Image.new("RGB",size,rgb)
    y,x=np.mgrid[0:h,0:w]; nx=(x-w*.5)/max(w*.75,1); ny=(y-h*.35)/max(h*.8,1)
    light=np.clip(1.07-.16*(nx*nx+ny*ny),.82,1.07); a=np.zeros((h,w,3),dtype=np.uint8)
    for i in range(3):a[:,:,i]=np.clip(rgb[i]*light,0,255).astype("uint8")
    return Image.fromarray(a,"RGB")

def composite(subject,bg,scale=1,x=50,y=13,angle=0):
    subject=subject.convert("RGBA"); bg=bg.convert("RGBA")
    if angle:subject=subject.rotate(angle,expand=True,resample=Image.Resampling.BICUBIC)
    th=max(1,int(bg.height*.78*scale)); tw=max(1,int(subject.width*th/max(subject.height,1)))
    subject=subject.resize((tw,th),Image.Resampling.LANCZOS)
    layer=Image.new("RGBA",bg.size,(0,0,0,0))
    # Position uses center point and keeps subject from being lost entirely.
    px=int((bg.width-tw)*x/100); py=int((bg.height-th)*y/100)
    layer.alpha_composite(subject,(px,py))
    return Image.alpha_composite(bg,layer).convert("RGB")

def enhance(im,b,c,s,sh):
    im=ImageEnhance.Brightness(im).enhance(b); im=ImageEnhance.Contrast(im).enhance(c)
    im=ImageEnhance.Color(im).enhance(s); return ImageEnhance.Sharpness(im).enhance(sh)

def bytes_png(im):
    b=io.BytesIO();im.save(b,format="PNG",optimize=True);return b.getvalue()
def bytes_jpg(im):
    b=io.BytesIO();im.convert("RGB").save(b,format="JPEG",quality=96,optimize=True,subsampling=0);return b.getvalue()

def local_backgrounds(category):
    results=[]
    for folder in CATEGORIES[category]:
        p=BG_ROOT/folder
        if p.exists():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in (".jpg",".jpeg",".png",".webp"):results.append((f.stem,f))
    return results

def official_crop(im,preset):
    # Standard output proportions; exact requirements vary by country/institution.
    sizes={"3 × 4 cm":(3,4),"4 × 6 cm":(4,6),"5 × 7 cm":(5,7),"Square 1:1":(1,1),"Passport portrait 35:45":(35,45)}
    rw,rh=sizes[preset]; target_ratio=rw/rh
    w,h=im.size; ratio=w/h
    if ratio>target_ratio:
        nw=int(h*target_ratio); left=(w-nw)//2; im=im.crop((left,0,left+nw,h))
    else:
        nh=int(w/target_ratio); top=max(0,int((h-nh)*.25)); im=im.crop((0,top,w,min(h,top+nh)))
    return im.resize((int(rw*300/2.54),int(rh*300/2.54)),Image.Resampling.LANCZOS) if "cm" in preset else im

for k,v in [("original",None),("cutout",None),("history",[]),("projects",{}),("lastkey",None),("lastpng",None)]:
    if k not in st.session_state:st.session_state[k]=v

with st.sidebar:
    st.markdown("### 🧭 التنقّل")
    page=st.radio("القائمة الرئيسية",["Home","استوديو الصور","المكتبة","مشاريعي / السجل","من نحن"],label_visibility="collapsed")
    st.divider()
    if page=="استوديو الصور":
        uploaded=st.file_uploader("ارفع صورة",type=["jpg","jpeg","png","webp"])
        if uploaded:
            raw=uploaded.getvalue(); digest=hashlib.sha256(raw).hexdigest()
            if st.session_state.get("digest")!=digest:
                st.session_state.original=raw;st.session_state.digest=digest;st.session_state.cutout=None;st.session_state.lastkey=None
        model_label=st.selectbox("محرك القص",list(MODEL_OPTIONS.keys()),index=1)
        model=MODEL_OPTIONS[model_label]
        st.caption("الاختيار الافتراضي U²-Net Human Seg أخف على الخادم. محرك BirefNet Portrait يحتاج تنزيل نموذج ضخم يقارب 1 GB وقد يتسبب في نفاد الذاكرة أو بطء بدء التطبيق.")
        st.divider();st.subheader("الخلفية")
        category=st.selectbox("قسم المكتبة",list(CATEGORIES.keys()))
        local=local_backgrounds(category)
        online=[x for x in ONLINE_BACKGROUNDS if x[1]==category]
        selected_bg=None
        names=["لون استوديو محايد"]+[f"{x[0]} (صورة عبر الإنترنت)" for x in online]+[f"{name} (محلي)" for name,_ in local]
        selected_name=st.selectbox("اختر خلفية",names)
        if selected_name.startswith("لون استوديو"):
            selected_bg=None
        elif "(صورة عبر الإنترنت)" in selected_name:
            selected_bg=next(x for x in online if x[0] in selected_name)
        else:
            selected_bg=next((("local",str(p)) for name,p in local if f"{name} (محلي)"==selected_name),None)
        color_name=st.selectbox("لون Fond neutre",list(NEUTRAL.keys()))
        color=NEUTRAL[color_name];gradient=st.toggle("تدرّج ناعم",True)
        custom=st.file_uploader("أو ارفع خلفية خاصة",type=["jpg","jpeg","png","webp"],key="custom")
        st.divider();st.subheader("التحرير")
        zoom=st.slider("تكبير/تصغير الشخص",.3,1.7,1.0,.05)
        rotate=st.slider("زاوية التدوير", -180,180,0)
        st.caption("يمكنك أيضًا استعمال عجلة الفأرة والسحب داخل مساحة المعاينة، ثم تثبيت التعديل بزر تطبيق التحويل.")
        st.divider();st.subheader("صور رسمية")
        preset=st.selectbox("مقاس الصورة",["بدون قص خاص","3 × 4 cm","4 × 6 cm","5 × 7 cm","Square 1:1","Passport portrait 35:45"])
        st.divider();st.subheader("تحسين خفيف")
        bright=st.slider("السطوع",.5,1.5,1.0,.05);contrast=st.slider("التباين",.5,1.5,1.0,.05)
        sat=st.slider("تشبع الألوان",0.0,1.8,1.0,.05);sharp=st.slider("الحدة",.5,2.0,1.0,.1)
    else:
        uploaded=None

if page=="Home":
    st.title("مرحبًا بك في SEIF PHOTO STUDIO PRO MAX")
    st.write("استوديو صور لتغيير الخلفيات، إعداد الصور الرسمية، تحسين الحواف، وتنظيم المشاريع.")
    a,b,c=st.columns(3)
    a.markdown("### ✂️ قص ذكي\nإزالة الخلفية واختيار محرك قص.")
    b.markdown("### 🖼️ مكتبة\nخلفيات محايدة وأعراس وفخامة وطبيعة.")
    c.markdown("### 🕘 مشاريع\nسجل نتائج يمكن معاينتها وتنزيلها.")
    st.info("اختر «استوديو الصور» من القائمة اليمنى لرفع صورة وبدء التعديل.")
    st.stop()
if page=="المكتبة":
    st.title("🖼️ مكتبة الخلفيات")
    st.write("يمكنك الاختيار من الصور المتاحة عبر الإنترنت أو إضافة صورك المرخّصة إلى مجلدات المكتبة داخل GitHub.")
    for cat in CATEGORIES:
        st.subheader(f"{cat} — الهدف {LIBRARY_TARGETS.get(cat,0)} خلفية")
        items=[x for x in ONLINE_BACKGROUNDS if x[1]==cat]
        if items:
            cols=st.columns(min(3,len(items)))
            for i,item in enumerate(items):
                with cols[i%len(cols)]:
                    try:
                        data=download_online_image(item[2]); st.image(data,caption=item[0],width="stretch")
                        st.download_button("تنزيل الخلفية",data=data,file_name=f"background_{i+1}.jpg",mime="image/jpeg",key=f"libdl_{cat}_{i}")
                    except Exception:st.warning(f"تعذر تحميل الصورة الآن: {item[0]}")
        else:st.caption("لا توجد صور حقيقية مضافة لهذا القسم بعد.")
        loc=local_backgrounds(cat)
        if loc:st.caption(f"صورك المحلية: {len(loc)}")
    st.warning("هذه نسخة أولية للمكتبة: تحتوي على صور خارجية مختارة للمعاينة، وليست مكتبة مكتملة من 80 صورة محلية. يجب استكمال الصور الناقصة قبل اعتبار المكتبة مكتملة.")
    st.stop()
if page=="مشاريعي / السجل":
    st.title("🕘 مشاريعي وسجل النتائج")
    if not st.session_state.history:st.info("ستظهر هنا النتائج التي تحفظها من الاستوديو في الجلسة الحالية.")
    else:
        for i,item in enumerate(reversed(st.session_state.history)):
            with st.expander(item["label"],expanded=(i==0)):
                st.image(item["bytes"],width="stretch")
                st.download_button("تنزيل النتيجة",data=item["bytes"],file_name=f"seif_history_{len(st.session_state.history)-i}.png",mime="image/png",key=f"hist_{i}")
    st.caption("حاليًا السجل داخل جلسة المتصفح. حفظ دائم بين الزيارات يحتاج قاعدة بيانات أو ملفات مشاريع على مساحة تخزين دائمة.")
    st.stop()
if page=="من نحن":
    st.title("من نحن")
    st.write("SEIF PHOTO STUDIO PRO MAX — أدوات عملية للمصورين لتجهيز الصور الرسمية وتغيير الخلفيات.")
    st.markdown("**الخدمات:** صور رسمية • أعراس • بورتريه • تعديل خلفيات • تجهيز الصور للطباعة")
    st.markdown("**الموقع:** VG9M+PVX, Rue Taher Sfar, Kalâa Kebira, Tunisia")
    st.markdown("**الهاتف:** 97262953")
    st.markdown("**البريد:** seifjaballah77@gmail.com")
    st.stop()

st.title("🎛️ استوديو الصور")
if not st.session_state.original:
    st.info("ارفع صورة من القائمة الجانبية لبدء العمل.")
    st.stop()
original=pil_open(st.session_state.original)
left,right=st.columns(2,gap="large")
with left:
    st.subheader("قبل")
    st.image(original,width="stretch")
with right:
    st.subheader("بعد")
    if st.button("✂️ إزالة الخلفية",type="primary",width="stretch"):
        try:
            with st.spinner("يجري قصّ الشخص وتحسين الحواف..."):
                st.session_state.cutout=cutout_cached(st.session_state.original,model)
            st.success("اكتمل القص. راجع الحواف، ثم اختر الخلفية.")
        except Exception as e:st.error(f"فشل القص: {e}")
    if st.session_state.cutout:
        subject=pil_open(st.session_state.cutout)
        if selected_bg and selected_bg[0]!="local":
            try:bg=pil_open(download_online_image(selected_bg[2])).convert("RGB").resize(original.size,Image.Resampling.LANCZOS)
            except Exception:bg=background_color(original.size,color,gradient)
        elif selected_bg and selected_bg[0]=="local":
            try:bg=pil_open(Path(selected_bg[1]).read_bytes()).convert("RGB").resize(original.size,Image.Resampling.LANCZOS)
            except Exception:bg=background_color(original.size,color,gradient)
        elif custom:bg=pil_open(custom.getvalue()).convert("RGB").resize(original.size,Image.Resampling.LANCZOS)
        else:bg=background_color(original.size,color,gradient)
        result=composite(subject,bg,zoom,50,13,rotate)
        result=enhance(result,bright,contrast,sat,sharp)
        if preset!="بدون قص خاص":result=official_crop(result,preset)
        result_png=bytes_png(result)
        result_key=hashlib.sha256(result_png).hexdigest()
        st.image(result,width="stretch")
        # Keep unique result history within this session.
        if not st.session_state.history or st.session_state.history[-1]["key"]!=result_key:
            st.session_state.history.append({"key":result_key,"bytes":result_png,"label":f"نتيجة {len(st.session_state.history)+1}"})
            st.session_state.history=st.session_state.history[-20:]
        # Before/after comparison with a draggable range control.
        st.subheader("مقارنة قبل / بعد")
        compare=st.slider("حرّك المؤشر لمقارنة الصورة الأصلية والنتيجة",0,100,50,key="compare")
        ca,cb=st.columns(2)
        with ca:st.image(original.resize(result.size,Image.Resampling.LANCZOS),caption=f"الأصل — {100-compare}%",width="stretch")
        with cb:st.image(result,caption=f"النتيجة — {compare}%",width="stretch")
        # Brush correction: paint regions to mark alpha erase/restore. This is a dedicated mask canvas.
        st.subheader("🖌️ فرشاة تصحيح القص")
        st.caption("ارسم بالأحمر على أجزاء الخلفية المتبقية لإخفائها، وبالأخضر على أجزاء الشخص التي تريد استعادتها. ثم طبّق التصحيح.")
        try:
            from streamlit_drawable_canvas import st_canvas
            c1,c2=st.columns(2)
            with c1:
                brush_mode=st.radio("نوع الفرشاة",["محو الخلفية","استعادة الشخص"],horizontal=True,key="brushmode")
                brush_color="#FF0000" if brush_mode=="محو الخلفية" else "#00FF00"
                brush_size=st.slider("حجم الفرشاة",5,80,24,key="brushsize")
            with c2:
                st.write("ارسم فوق معاينة القص، ثم اضغط تطبيق.")
            cutout_for_canvas=subject.copy()
            cutout_for_canvas.thumbnail((850,850))
            canvas_result=st_canvas(fill_color="rgba(0,0,0,0)",stroke_width=brush_size,stroke_color=brush_color,background_image=cutout_for_canvas,drawing_mode="freedraw",height=cutout_for_canvas.height,width=cutout_for_canvas.width,key="brush_canvas")
            if st.button("تطبيق تصحيح الفرشاة"):
                if canvas_result.image_data is None:st.warning("ارسم بعض الخطوط أولًا.")
                else:
                    rgba=np.array(subject.convert("RGBA"))
                    canvas_arr=np.array(canvas_result.image_data)
                    # Map canvas stroke marks back to the full-size alpha mask.
                    mark_red=(canvas_arr[:,:,0]>180)&(canvas_arr[:,:,1]<100)&(canvas_arr[:,:,2]<100)
                    mark_green=(canvas_arr[:,:,1]>150)&(canvas_arr[:,:,0]<100)&(canvas_arr[:,:,2]<100)
                    mask_img=Image.fromarray((mark_red.astype("uint8")*255),"L").resize(subject.size,Image.Resampling.NEAREST)
                    restore_img=Image.fromarray((mark_green.astype("uint8")*255),"L").resize(subject.size,Image.Resampling.NEAREST)
                    alpha=rgba[:,:,3].copy()
                    erase=np.array(mask_img)>0; restore=np.array(restore_img)>0
                    alpha[erase]=0; alpha[restore]=255; rgba[:,:,3]=alpha
                    st.session_state.cutout=bytes_png(Image.fromarray(rgba,"RGBA"))
                    st.success("طُبّق تعديل الفرشاة. اضغط إزالة الخلفية فقط إذا أردت إعادة القص من الأصل.")
                    st.rerun()
        except ImportError:
            st.warning("فرشاة التصحيح تحتاج تثبيت streamlit-drawable-canvas؛ أضفها من requirements.txt ثم أعد تشغيل التطبيق.")
        d1,d2=st.columns(2)
        with d1:st.download_button("تنزيل JPG عالي الجودة",data=bytes_jpg(result),file_name="seif_photo_studio.jpg",mime="image/jpeg",width="stretch")
        with d2:st.download_button("تنزيل PNG",data=result_png,file_name="seif_photo_studio.png",mime="image/png",width="stretch")
        st.subheader("حفظ المشروع")
        project_name=st.text_input("اسم المشروع",value="مشروع جديد")
        if st.button("حفظ في سجل الجلسة"):
            st.session_state.projects[project_name]={"original":st.session_state.original,"cutout":st.session_state.cutout,"result":result_png,"settings":{"zoom":zoom,"rotation":rotate,"background":color_name,"preset":preset}}
            st.success("تم حفظ المشروع في جلسة المتصفح الحالية.")
    else:st.info("اضغط «إزالة الخلفية» لبدء التحرير.")

st.divider()
st.caption("ملاحظة: التكبير والتدوير في هذه النسخة يتمان من عناصر التحكم. محرر التحويل الحر بمقابض الماوس وربط سحب/تدوير الصورة بملف التصدير ما زال يحتاج مكوّنًا تفاعليًا مخصصًا.")
