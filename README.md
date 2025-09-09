# DOZE — Drone Operation Zone Editor

DOZE is a Streamlit app to sketch **Flight Geography (FG)** and automatically derive **Contingency Volume (CV)**, **Operational Volume (OV)** and **Ground Risk Buffer (GRB)**. These can then be exported to .kmz to use with Google Earth.

> This tool does **not** check airspace/NO-fly rules. It’s a planning helper based on https://www.lba.de/SharedDocs/Downloads/DE/B/B5_UAS/Leitfaden_FG_CV_GRB_eng.pdf which is in turn based on Regulation (EU) 2019/947

---
- Draw a polygon on a map and choose whether it represents **FG**, **CV**, or **GRB**.
- Automatically derive the other boundaries based on your input.
- Switch between **drone profiles** to prefill performance and accuracy parameters.
- View live **calculated safety margins** (CV lateral margin, vertical buffer, FG apex, CV apex, GRB margin).
- Export all zones as a **KMZ file** (with 3D volumes for FG/CV/OV, and the GRB).
---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>

# Recommended: create a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

To run:
```bash
streamlit run app.py
```
The app opens in your browser (default: http://localhost:8501/


## Disclaimer
This software provides planning visualisation only. It does not validate legal/operational constraints or local airspace rules. You are responsible for ensuring compliance with all applicable regulations and site-specific restrictions.
