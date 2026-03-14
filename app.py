import streamlit as st
import fitz
import base64
import json
import io
import os
import re
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="DDR Generator | UrbanRoof", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center; }
    .hero h1 { color: #f5c518; font-size: 2.2rem; margin: 0; }
    .hero p  { color: #ccc; margin: 0.5rem 0 0; font-size: 1rem; }
    .section-card { background: white; border-left: 4px solid #f5c518; border-radius: 8px;
                    padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; text-transform: uppercase;
                     letter-spacing: .5px; border-bottom: 2px solid #f5c518;
                     padding-bottom: .5rem; margin-bottom: 1rem; }
    .severity-high   { background:#fee2e2; color:#991b1b; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .severity-medium { background:#fef3c7; color:#92400e; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .severity-low    { background:#d1fae5; color:#065f46; padding:.25rem .75rem; border-radius:20px; font-weight:600; font-size:.85rem; }
    .thermal-badge   { background:#e0f2fe; color:#0369a1; padding:.2rem .6rem; border-radius:12px; font-size:.78rem; font-weight:600; }
    .stButton>button { background:linear-gradient(135deg,#f5c518,#e6b800); color:#1a1a2e; font-weight:700; border:none; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


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


DDR_PROMPT = """You are an expert building inspection engineer for UrbanRoof.
Analyse the two documents below and generate a Detailed Diagnosis Report (DDR).

STRICT RULES:
- Never invent facts not in the documents
- Missing info = write "Not Available"
- Conflicts = mention them explicitly
- Simple client-friendly language
- Map impacted areas (-ve side) to source areas (+ve side)
- Combine thermal readings with visual observations

Return ONLY valid JSON, no markdown fences, no explanation.

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

Return only the JSON object:"""


def generate_ddr(inspection_text, thermal_text, api_key):
    client = genai.Client(api_key=api_key)
    prompt = DDR_PROMPT.format(
        inspection_text=inspection_text[:5000],
        thermal_text=thermal_text[:2000],
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=8192),
    )
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    # Try normal parse first, fall back to json_repair
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from json_repair import repair_json
        repaired = repair_json(raw)
        return json.loads(repaired)


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


def generate_pdf(ddr, thermal_imgs, inspection_imgs):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable,
                                    Image as RLImage, PageBreak, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=15*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    YELLOW = colors.HexColor("#f5c518")
    DARK   = colors.HexColor("#1a1a2e")
    GRAY   = colors.HexColor("#f8f9fa")
    RED    = colors.HexColor("#991b1b")
    AMBER  = colors.HexColor("#92400e")
    GREEN  = colors.HexColor("#065f46")

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 textColor=colors.white, fontSize=22,
                                 backColor=DARK, spaceAfter=4, spaceBefore=4,
                                 leftIndent=8, rightIndent=8, leading=28)
    h1   = ParagraphStyle("h1", parent=styles["Heading1"], textColor=DARK,
                           fontSize=12, spaceBefore=10, spaceAfter=4,
                           backColor=colors.HexColor("#fefce8"))
    h2   = ParagraphStyle("h2", parent=styles["Heading2"], textColor=DARK,
                           fontSize=10, spaceBefore=6, spaceAfter=2)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9,
                           leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    warn = ParagraphStyle("warn", parent=styles["Normal"], fontSize=9,
                           textColor=AMBER, leading=13,
                           backColor=colors.HexColor("#fffbeb"),
                           leftIndent=8, rightIndent=8, spaceAfter=4)

    story = []

    # Cover
    story.append(Paragraph("Detailed Diagnosis Report", title_style))
    story.append(Paragraph("UrbanRoof Private Limited | www.urbanroof.in",
                            ParagraphStyle("sub", parent=styles["Normal"],
                                           textColor=YELLOW, fontSize=9,
                                           backColor=DARK, spaceAfter=6, leftIndent=8)))
    story.append(HRFlowable(width="100%", thickness=3, color=YELLOW, spaceAfter=12))

    # Section 1
    ps = ddr.get("property_summary", {})
    story.append(Paragraph("SECTION 1 — PROPERTY SUMMARY", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    props = [
        ["Property Type",    ps.get("property_type","N/A"),  "Inspection Date",  ps.get("inspection_date","N/A")],
        ["Inspected By",     ps.get("inspected_by","N/A"),   "Floors",           ps.get("floors","N/A")],
        ["Previous Audit",   ps.get("previous_audit","N/A"), "Previous Repairs", ps.get("previous_repairs","N/A")],
        ["Overall Condition",ps.get("overall_condition","N/A"), "", ""],
    ]
    pt = Table(props, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1), DARK), ("BACKGROUND",(2,0),(2,-1), DARK),
        ("TEXTCOLOR", (0,0),(0,-1), colors.white), ("TEXTCOLOR",(2,0),(2,-1), colors.white),
        ("BACKGROUND",(1,0),(1,-1), GRAY), ("BACKGROUND",(3,0),(3,-1), GRAY),
        ("FONTSIZE",  (0,0),(-1,-1), 8), ("FONTNAME",(0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0),(2,-1), "Helvetica-Bold"),
        ("GRID",      (0,0),(-1,-1), 0.5, colors.white),
        ("PADDING",   (0,0),(-1,-1), 5), ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(pt)
    story.append(Spacer(1,4))
    story.append(Paragraph(ps.get("brief_overview",""), body))
    story.append(Spacer(1,8))

    # Section 2
    areas   = ddr.get("area_observations", [])
    img_map = assign_images(inspection_imgs, thermal_imgs, len(areas))
    story.append(Paragraph("SECTION 2 — AREA-WISE OBSERVATIONS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    for i, obs in enumerate(areas):
        ac = []
        ac.append(Paragraph(f"Area {i+1}: {obs.get('area_name','')} — {obs.get('issue_type','')}", h2))
        rows = [
            ["Impacted Side", obs.get("negative_side","N/A")],
            ["Source Side",   obs.get("positive_side","N/A")],
            ["Description",   obs.get("visual_description","N/A")],
        ]
        tr = obs.get("thermal_reading","")
        if tr and tr not in ("N/A","Not Available",""):
            rows.append(["Thermal Reading", tr])
        ot = Table(rows, colWidths=[40*mm, 135*mm])
        ot.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,-1), DARK), ("TEXTCOLOR",(0,0),(0,-1), colors.white),
            ("BACKGROUND",(1,0),(1,-1), GRAY), ("FONTSIZE",(0,0),(-1,-1), 8),
            ("FONTNAME",(0,0),(0,-1), "Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1), 0.5, colors.white),
            ("PADDING",(0,0),(-1,-1), 5), ("VALIGN",(0,0),(-1,-1), "TOP"),
        ]))
        ac.append(ot)
        imgs_row = []
        for img in img_map.get(i,{}).get("inspection",[])[:1]:
            try:
                ib = io.BytesIO(base64.b64decode(img["b64"]))
                imgs_row.append(RLImage(ib, width=55*mm, height=40*mm))
            except: pass
        for img in img_map.get(i,{}).get("thermal",[])[:1]:
            try:
                ib = io.BytesIO(base64.b64decode(img["b64"]))
                imgs_row.append(RLImage(ib, width=55*mm, height=40*mm))
            except: pass
        if imgs_row:
            while len(imgs_row) < 2:
                imgs_row.append(Paragraph("", body))
            it = Table([imgs_row], colWidths=[90*mm, 90*mm])
            it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("PADDING",(0,0),(-1,-1),4)]))
            ac.append(it)
        ac.append(Spacer(1,6))
        story.append(KeepTogether(ac))

    # Section 3
    story.append(Paragraph("SECTION 3 — PROBABLE ROOT CAUSES", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    for rc in ddr.get("root_causes",[]):
        story.append(Paragraph(f"{rc.get('cause','')}", h2))
        story.append(Paragraph(f"<b>Affected:</b> {', '.join(rc.get('affected_areas',[]))}", body))
        story.append(Paragraph(rc.get("explanation",""), body))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=4))

    # Section 4
    story.append(Paragraph("SECTION 4 — SEVERITY ASSESSMENT", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    sev_data = [["Area","Severity","Reasoning"]]
    for sa in ddr.get("severity_assessment",[]):
        sev_data.append([sa.get("area",""), sa.get("severity",""), sa.get("reasoning","")])
    st_table = Table(sev_data, colWidths=[40*mm, 30*mm, 110*mm])
    sev_style = [
        ("BACKGROUND",(0,0),(-1,0), DARK), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1), 8),
        ("GRID",(0,0),(-1,-1), 0.5, colors.lightgrey), ("PADDING",(0,0),(-1,-1), 5),
        ("VALIGN",(0,0),(-1,-1), "TOP"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
    ]
    for idx, sa in enumerate(ddr.get("severity_assessment",[]), start=1):
        s = sa.get("severity","").lower()
        if any(x in s for x in ["high","critical","immediate","poor"]):
            sev_style += [("TEXTCOLOR",(1,idx),(1,idx),RED),("FONTNAME",(1,idx),(1,idx),"Helvetica-Bold")]
        elif any(x in s for x in ["medium","moderate"]):
            sev_style += [("TEXTCOLOR",(1,idx),(1,idx),AMBER),("FONTNAME",(1,idx),(1,idx),"Helvetica-Bold")]
        else:
            sev_style += [("TEXTCOLOR",(1,idx),(1,idx),GREEN)]
    st_table.setStyle(TableStyle(sev_style))
    story.append(st_table)
    story.append(Spacer(1,8))

    # Section 5
    story.append(Paragraph("SECTION 5 — RECOMMENDED ACTIONS", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    pord = {"Immediate":0,"High":1,"Medium":2,"Low":3}
    for action in sorted(ddr.get("recommended_actions",[]), key=lambda x: pord.get(x.get("priority","Low"),3)):
        pri  = action.get("priority","")
        icon = "HIGH PRIORITY" if pri=="Immediate" else "MEDIUM" if pri in ("High","Medium") else "LOW"
        story.append(Paragraph(f"[{icon}] {action.get('action','')}", h2))
        story.append(Paragraph(f"<b>Areas:</b> {', '.join(action.get('areas',[]))}", body))
        story.append(Paragraph(action.get("description",""), body))
        story.append(Spacer(1,4))

    # Section 6
    notes = ddr.get("additional_notes",[])
    if notes:
        story.append(Paragraph("SECTION 6 — ADDITIONAL NOTES", h1))
        story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
        for n in notes:
            story.append(Paragraph(f"• {n}", body))
        story.append(Spacer(1,8))

    # Section 7
    story.append(Paragraph("SECTION 7 — MISSING OR UNCLEAR INFORMATION", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    for m in ddr.get("missing_information",[]):
        story.append(Paragraph(f"WARNING: {m}", warn))
    if not ddr.get("missing_information"):
        story.append(Paragraph("All key information was present in the source documents.", body))

    # Summary Table
    story.append(PageBreak())
    story.append(Paragraph("ISSUE SUMMARY TABLE", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=YELLOW, spaceAfter=6))
    sum_data = [["Area","Issue","Negative Side","Source (Positive)","Thermal"]]
    for obs in areas:
        sum_data.append([obs.get("area_name",""), obs.get("issue_type",""),
                         obs.get("negative_side",""), obs.get("positive_side",""),
                         obs.get("thermal_reading","N/A")])
    sum_t = Table(sum_data, colWidths=[28*mm, 25*mm, 45*mm, 45*mm, 37*mm])
    sum_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), DARK), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1), 7),
        ("GRID",(0,0),(-1,-1), 0.5, colors.lightgrey), ("PADDING",(0,0),(-1,-1), 4),
        ("VALIGN",(0,0),(-1,-1), "TOP"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
    ]))
    story.append(sum_t)

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK)
        canvas.rect(0, 0, A4[0], 18*mm, fill=True, stroke=False)
        canvas.setFillColor(YELLOW)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawCentredString(A4[0]/2, 10*mm,
            "UrbanRoof Private Limited | www.urbanroof.in | +91-8925-805-805")
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(A4[0]/2, 5*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return buf.getvalue()


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
                    st.image(b64_to_display(img["b64"]), caption=f"Inspection — Page {img['page']}", use_container_width=True)
                for img in img_map.get(i,{}).get("thermal",[]):
                    st.image(b64_to_display(img["b64"]), caption=f"Thermal — Page {img['page']}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3
    st.markdown('<div class="section-card"><div class="section-title">Section 3 — Probable Root Causes</div>', unsafe_allow_html=True)
    for rc in ddr.get("root_causes",[]):
        st.markdown(f"**🔍 {rc.get('cause','')}**")
        st.markdown(f"*Affected:* {', '.join(rc.get('affected_areas',[]))}")
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
        pri  = action.get("priority","")
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

    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#666;border-top:1px solid #eee;margin-top:2rem'>
        <strong style='color:#f5c518'>UrbanRoof Private Limited</strong><br>
        www.urbanroof.in | info@urbanroof.in | +91-8925-805-805
    </div>""", unsafe_allow_html=True)

    # Download buttons
    st.markdown("### 📥 Download Report")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📄 Download DDR (PDF)",
            data=generate_pdf(ddr, thermal_imgs, inspection_imgs),
            file_name="DDR_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "🗂 Download DDR (JSON)",
            data=json.dumps(ddr, indent=2),
            file_name="DDR_Report.json",
            mime="application/json",
            use_container_width=True,
        )


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
        st.markdown("1. 📄 Upload **Inspection Report** PDF\n2. 🌡 Upload **Thermal Images** PDF\n3. 🚀 Click **Generate DDR**\n4. 📥 Download as PDF")
        st.divider()
        st.caption("Stack: Streamlit · PyMuPDF · Gemini 2.5 Flash · ReportLab")

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
