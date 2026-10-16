# SEIF PHOTO STUDIO PRO MAX

Streamlit photo studio for background replacement, portrait cutout, neutral backgrounds, official-photo crops, brush corrections, and a session history.

## Deploy on Streamlit Community Cloud

- Repository: `rajabelaid1234-collab/seif-photo-studio-pro`
- Branch: `main`
- Main file: `streamlit_app.py`

## Current features

- Home, Studio, Background Library, Projects/History, and About navigation.
- Background removal using selectable rembg models: BirefNet Portrait, U²-Net Human Seg, and U²-Net.
- Alpha matting for potentially cleaner hair and clothing edges.
- Neutral studio color backgrounds and custom background upload.
- Online example gallery plus local background folders.
- Brush correction UI for marking areas to erase or restore.
- Official-photo crop presets (3×4 cm, 4×6 cm, 5×7 cm, 1:1, 35:45).
- Brightness, contrast, saturation, sharpness.
- Before/after comparison controls.
- PNG/JPEG downloads and session-based result history.

## Background library status

The requested library target is 80 photographs (10 neutral, 20 wedding, 15 traditional, 15 luxury, 10 nature, plus 10 additional neutral/gray/white variants). This initial code contains a small set of online example links and local folder support; it does **not** yet contain 80 bundled photo files. Add appropriately licensed images to:

- `backgrounds/neutral/`
- `backgrounds/gray/`
- `backgrounds/white/`
- `backgrounds/wedding/`
- `backgrounds/traditional/`
- `backgrounds/luxury/`
- `backgrounds/nature/`

## Important limitations

- Mouse-wheel zoom/drag preview is not yet wired as a persistent transform of the exported image; current person size and rotation are controlled in the Studio controls.
- The brush UI is an initial implementation and should be tested on Streamlit Cloud before production use.
- History and saved project data are currently session-based, not persistent between visits.
- The first AI cutout can be slow while the model downloads; deployment memory limits may affect large images.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```
