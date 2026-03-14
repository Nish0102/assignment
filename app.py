import streamlit as st
import fitz  # PyMuPDF
import base64
import json
import io
import os
import re
from PIL import Image
from google import genai
from google.genai import types

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDR Generator | UrbanRoof",
    page_icon="🏠",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 2rem; text-align: center;
    }
    .hero h1 { color: #f5c518; font-size: 2.2rem; margin: 0; }
    .hero p  { color: #ccc; margin: 0.5rem 0 0; font-size: 1rem; }
    .section-card {
        background: white; border-left: 4px solid #f5c518;
        border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1a1a2e;
        text-transform: uppercase; letter-spacing: .5px;
        border-bottom: 2px solid #f5c518; padding-bottom: .5rem; margin-bottom: 1rem;
    }
    .severity-high   { background:#fee2e2; color:#991b1b; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .severity-medium { background:#fef3c7; color:#92400e; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .severity-low    { background:#d1fae5; color:#065f46; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .thermal-badge   { background:#e0f2fe; color:#0369a1; padding:.2rem .6rem; border-radius:12px; font-size:.78rem; font-weight:600; }
    .warn-box        { background:#fffbeb; border-left:4px solid #f59e0b; padding:1rem; border-radius:8px; margin:.5rem 0; }
    .stButton>button { background:linear-gradient(135deg,#f5c518,#e6b800); color:#1a1a2e; font-weight:700; border:none; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ── PDF Helpers ───────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def extract_pdf_images(pdf_bytes, max_images=30, min_size=250):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                w, h = pil_img.size
                if w < min_size or h < min_size:
                    continue
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=72)
                b64 = base64.b64encode(buf.getvalue()).decode()
                images.append({"page": page_num + 1, "b64": b64, "width": w, "height": h})
                if len(images) >= max_images:
                    return images
            except Exception:
                continue
    return images


def b64_to_display(b64):
    return f"data:image/jpeg;base64,{b64}"


def get_severity_badge(sev):
    s = sev.lower()
    if any(x in s for x in ["high", "critical", "immediate", "poor"]):
        return f'<span class="severity-high">🔴 {sev}</span>'
    elif any(x in s for x in ["medium", "moderate"]):
        return f'<span class="severity-medium">🟡 {sev}</span>'
    else:
        return f'<span class="severity-low">🟢 {sev}</span>'


# ── Gemini DDR Generation ─────────────────────────────────────────────────────
DDR_PROMPT = """You are an expert building inspection engineer for UrbanRoof.
Analyse the two documents below (Inspection Report + Thermal Imaging Report)
and generate a complete Detailed Diagnosis Report (DDR).

STRICT RULES:
- Never invent facts not present in the documents
- If information is missing write "Not Available"
- If information conflicts mention the conflict explicitly
- Use simple client-friendly language
- Map impacted areas (-ve side) to their source areas (+ve side)
- Combine thermal temperature readings with visual observations

Return ONLY a valid JSON object — no markdown fences, no explanation, pure JSON.

JSON schema:
{{
  "property_summary": {{
    "property_type": "",
    "inspection_date": "",
    "inspected_by": "",
    "floors": "",
    "previous_audit": "",
    "previous_repairs": "",
    "overall_condition": "",
    "total_issues": 0,
    "brief_overview": ""
  }},
  "area_observations": [
    {{
      "area_name": "",
      "issue_type": "",
      "negative_side": "",
      "positive_side": "",
      "thermal_reading": "",
      "visual_description": ""
    }}
  ],
  "root_causes": [
    {{
      "cause": "",
      "affected_areas": [],
      "explanation": ""
    }}
  ],
  "severity_assessment": [
    {{
      "area": "",
      "severity": "",
      "reasoning": ""
    }}
  ],
  "recommended_actions": [
    {{
      "action": "",
      "areas": [],
      "priority": "",
      "description": ""
    }}
  ],
  "additional_notes": [],
  "missing_information": []
}}

=== INSPECTION REPORT ===
{inspection_text}

=== THERMAL IMAGING REPORT ===
{thermal_text}

Generate the DDR JSON now:"""


def generate_ddr(inspection_text, thermal_text, api_key):
    client = genai.Client(api_key=api_key)
    prompt = DDR_PROMPT.format(
        inspection_text=inspection_text[:7000],
        thermal_text=thermal_text[:3000],
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=8192),
    )
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Image assignment ──────────────────────────────────────────────────────────
def assign_images(inspection_imgs, thermal_imgs, n_areas):
    assignment = {}
    ip = max(1, len(inspection_imgs) // max(n_areas, 1))
    tp = max(1, len(thermal_imgs)    // max(n_areas, 1))
    for i in range(n_areas):
        assignment[i] = {
            "inspection": inspection_imgs[i*ip : i*ip+2],
            "thermal":    thermal_imgs[i*tp   : i*tp+1],
        }
    return assignment


# ── Render DDR ────────────────────────────────────────────────────────────────
def render_ddr(ddr, inspection_imgs, thermal_imgs):
    ps = ddr.get("property_summary", {})

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);
                padding:2rem;border-radius:16px;color:white;margin-bottom:2rem'>
        <h1 style='color:#f5c518;margin:0'>📋 Detailed Diagnosis Report</h1>
        <p style='color:#aaa;margin:.5rem 0 0'>UrbanRoof Private Limited | www.urbanroof.in</p>
    </div>""", unsafe_allow_html=True)

    # Section 1
    st.markdown('<div class="section-card"><div class="section-title">Section 1 — Property Summary</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Property Type",    ps.get("property_type","N/A"))
    c2.metric("Inspection Date",  ps.get("inspection_date","N/A"))
    c3.metric("Inspected By",     ps.get("inspected_by","N/A"))
    c4,c5,c6 = st.columns(3)
    c4.metric("Floors",           ps.get("floors","N/A"))
    c5.metric("Previous Audit",   ps.get("previous_audit","N/A"))
    c6.metric("Previous Repairs", ps.get("previous_repairs","N/A"))
    st.markdown(f"**Overall Condition:** {ps.get('overall_condition','N/A')}")
    st.info(ps.get("brief_overview",""))
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2
    st.markdown('<div class="section-card"><div class="section-title">Section 2 — Area-wise Observations</div>', unsafe_allow_html=True)
    areas   = ddr.get("area_observations", [])
    img_map = assign_images(inspection_imgs, thermal_imgs, len(areas))
    for i, obs in enumerate(areas):
        with st.expander(f"📍 {obs.get('area_name','Area')}  —  {obs.get('issue_type','')}", expanded=(i==0)):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**🔻 Impacted Side:** {obs.get('negative_side','N/A')}")
                st.markdown(f"**🔺 Source Side:**   {obs.get('positive_side','N/A')}")
                st.markdown(f"**👁 Description:**   {obs.get('visual_description','N/A')}")
                tr = obs.get("thermal_reading","")
                if tr and tr not in ("N/A","Not Available",""):
                    st.markdown(f'<span class="thermal-badge">🌡 Thermal: {tr}</span>', unsafe_allow_html=True)
            with right:
                for img in img_map.get(i,{}).get("inspection",[]):
                    st.image(b64_to_display(img["b64"]), caption=f"Inspection Photo — Page {img['page']}", use_container_width=True)
                for img in img_map.get(i,{}).get("thermal",[]):
                    st.image(b64_to_display(img["b64"]), caption=f"Thermal Scan — Page {img['page']}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3
    st.markdown('<div class="section-card"><div class="section-title">Section 3 — Probable Root Causes</div>', unsafe_allow_html=True)
    for rc in ddr.get("root_causes",[]):
        st.markdown(f"**🔍 {rc.get('cause','')}**")
        st.markdown(f"*Affected areas:* {', '.join(rc.get('affected_areas',[]))}")
        st.markdown(rc.get("explanation",""))
        st.divider()
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 4
    st.markdown('<div class="section-card"><div class="section-title">Section 4 — Severity Assessment</div>', unsafe_allow_html=True)
    for sa in ddr.get("severity_assessment",[]):
        ca, cb = st.columns([2,3])
        with ca:
            st.markdown(f"**{sa.get('area','')}**")
            st.markdown(get_severity_badge(sa.get("severity","")), unsafe_allow_html=True)
        with cb:
            st.markdown(f"_{sa.get('reasoning','')}_")
        st.divider()
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 5
    st.markdown('<div class="section-card"><div class="section-title">Section 5 — Recommended Actions</div>', unsafe_allow_html=True)
    pord = {"Immediate":0,"High":1,"Medium":2,"Low":3}
    for action in sorted(ddr.get("recommended_actions",[]), key=lambda x: pord.get(x.get("priority","Low"),3)):
        pri = action.get("priority","")
        icon = "🔴" if pri=="Immediate" else "🟡" if pri in ("High","Medium") else "🟢"
        with st.expander(f"{icon} {action.get('action','')}  [{pri}]"):
            st.markdown(f"**Areas:** {', '.join(action.get('areas',[]))}")
            st.markdown(action.get("description",""))
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 6
    notes = ddr.get("additional_notes",[])
    if notes:
        st.markdown('<div class="section-card"><div class="section-title">Section 6 — Additional Notes</div>', unsafe_allow_html=True)
        for n in notes:
            st.markdown(f"• {n}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 7
    st.markdown('<div class="section-card"><div class="section-title">Section 7 — Missing or Unclear Information</div>', unsafe_allow_html=True)
    for m in ddr.get("missing_information",[]):
        st.warning(f"⚠️ {m}")
    if not ddr.get("missing_information"):
        st.success("All key information was present in the source documents.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary Table
    import pandas as pd
    st.markdown('<div class="section-card"><div class="section-title">📊 Issue Summary Table</div>', unsafe_allow_html=True)
    rows = [{"Area":o.get("area_name",""), "Issue":o.get("issue_type",""),
             "Negative Side":o.get("negative_side",""), "Source":o.get("positive_side",""),
             "Thermal":o.get("thermal_reading","N/A")} for o in areas]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Thermal gallery
    if thermal_imgs:
        st.markdown('<div class="section-card"><div class="section-title">🌡 Thermal Image Gallery</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, img in enumerate(thermal_imgs[:15]):
            cols[idx%3].image(b64_to_display(img["b64"]), caption=f"Page {img['page']}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer + download
    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#666;border-top:1px solid #eee;margin-top:2rem'>
        <strong style='color:#f5c518'>UrbanRoof Private Limited</strong><br>
        www.urbanroof.in | info@urbanroof.in | +91-8925-805-805
    </div>""", unsafe_allow_html=True)

    st.download_button("📥 Download DDR (JSON)", data=json.dumps(ddr, indent=2),
                       file_name="DDR_Report.json", mime="application/json")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero">
        <h1>🏠 DDR Report Generator</h1>
        <p>AI-powered Detailed Diagnosis Report — Inspection + Thermal Documents</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password",
                                placeholder="AIza...", help="Get free key at aistudio.google.com")
        st.markdown("[🔑 Get free Gemini API key](https://aistudio.google.com/apikey)")
        st.divider()
        st.markdown("### How it works")
        st.markdown("1. 📄 Upload **Inspection Report** PDF\n2. 🌡 Upload **Thermal Images** PDF\n3. 🚀 Click **Generate DDR**\n4. 📥 View & download")
        st.divider()
        st.caption("Stack: Streamlit · PyMuPDF · Gemini 2.0 Flash")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 📄 Inspection Report")
        insp_file = st.file_uploader("Inspection PDF", type=["pdf"], key="insp", label_visibility="collapsed")
        if insp_file:
            st.success(f"✅ {insp_file.name}  ({insp_file.size//1024} KB)")
    with col_r:
        st.markdown("#### 🌡 Thermal Images Report")
        therm_file = st.file_uploader("Thermal PDF", type=["pdf"], key="therm", label_visibility="collapsed")
        if therm_file:
            st.success(f"✅ {therm_file.name}  ({therm_file.size//1024} KB)")

    st.divider()
    gcol, _ = st.columns([2,5])
    with gcol:
        go = st.button("🚀 Generate DDR Report", use_container_width=True)

    if go:
        if not api_key:
            st.error("⚠️ Enter your Gemini API key in the sidebar.")
            return
        if not insp_file or not therm_file:
            st.error("⚠️ Upload both PDF files.")
            return

        prog   = st.progress(0)
        status = st.empty()

        status.info("📖 Reading Inspection Report...")
        ib = insp_file.read();  it = extract_pdf_text(ib);  prog.progress(15)

        status.info("📖 Reading Thermal Report...")
        tb = therm_file.read(); tt = extract_pdf_text(tb);  prog.progress(30)

        status.info("🖼 Extracting inspection photos...")
        ii = extract_pdf_images(ib, 20, 200);               prog.progress(50)

        status.info("🌡 Extracting thermal images...")
        ti = extract_pdf_images(tb, 30, 300);               prog.progress(65)

        status.info("🤖 Generating DDR with Gemini AI...")
        try:
            ddr = generate_ddr(it, tt, api_key)
        except json.JSONDecodeError as e:
            st.error(f"AI returned invalid JSON — try again. Detail: {e}")
            return
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return

        prog.progress(100)
        status.success("✅ DDR Generated Successfully!")
        st.divider()
        render_ddr(ddr, ii, ti)


if __name__ == "__main__":
    main()
