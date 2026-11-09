"""Streamlit front end for the Cancer Care AI Companion prototype.
Run from the repository root:  streamlit run app/companion_app.py
Needs models/companion_models.joblib (python src/11_final_models.py)."""
import importlib, pathlib, sys
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
companion = importlib.import_module("14_companion")

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
LOGO, ICON = ASSETS / "cac_logo.png", ASSETS / "cac_icon_64.png"

st.set_page_config(page_title="CAC (Cancer Care AI Companion)", page_icon=str(ICON) if ICON.exists() else "🩺", layout="centered")
if LOGO.exists():
    if hasattr(st, "logo"):                       # header logo needs Streamlit >= 1.35
        st.logo(str(LOGO), icon_image=str(ICON))
    st.image(str(LOGO), width=420)
    st.markdown("<h1 style='position:absolute;left:-9999px'>Cancer Care AI Companion</h1>", unsafe_allow_html=True)  # page title for screen readers
else:
    st.title("Cancer Care AI Companion")
st.caption("Early cost range and financial-risk flag for a cancer hospitalization in India (prototype).")


@st.cache_resource
def bundle():
    return companion.load_bundle()


try:
    B = bundle()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Patient age (years)", 0, 100, 54, help="Whole years, 0 to 100.")
    sex = c1.selectbox("Sex", ["Female", "Male"])
    sector = c1.selectbox("Residence", ["Rural", "Urban"])
    hh = c1.number_input("Household size", 1, 20, 5, help="Number of household members, 1 to 20.")
    quint = c1.selectbox("Household consumption quintile (1 = poorest)", [1, 2, 3, 4, 5], index=1)
    social = c1.selectbox("Social group", ["Scheduled Tribe", "Scheduled Caste", "Other Backward Class", "Other"], index=2)
    hospital = c2.selectbox("Hospital type", ["Public", "Private or charitable"], index=1)
    ward = c2.selectbox("Ward", ["Free", "Paying general", "Paying special"], index=1)
    stay = c2.number_input("Expected length of stay (days)", 1, 120, 10, help="1 to 120 days.")
    surgery = c2.checkbox("Surgery planned", True)
    district = c2.checkbox("Treatment outside home district", True)
    state = c2.checkbox("Treatment outside home state", False)
    insured = c2.checkbox("Covered by a health scheme (for example PM-JAY)", False)
    consumption = st.number_input("Household annual consumption in rupees (optional, 0 to skip)", 0, 5_000_000, 180_000, step=10_000,
                                  help="0 to 50,00,000 rupees a year; enter 0 to skip.")
    st.caption("If a box turns red, the value is outside its allowed range; hover over the ⓘ next to the label to see the range.")
    go = st.form_submit_button("Estimate")

if go:
    p = {"age": age, "female": int(sex == "Female"), "sector_urban": int(sector == "Urban"), "facility_public": int(hospital == "Public"),
         "ward_type": ["Free", "Paying general", "Paying special"].index(ward) + 1, "stay_days": stay, "surgery_received": int(surgery),
         "outside_district": int(district or state), "outside_state": int(state), "insured": int(insured), "mpce_quintile": quint,
         "social_group": {"Scheduled Tribe": 1, "Scheduled Caste": 2, "Other Backward Class": 3, "Other": 9}[social], "hh_size": hh}
    if consumption:
        p["annual_consumption"] = consumption
    r = companion.estimate(p, B)
    st.subheader("Estimated out-of-pocket cost")
    m1, m2, m3 = st.columns(3)
    m1.metric("Low end", f"₹{r['cost_low_rupees']:,.0f}")
    m2.metric("Typical", f"₹{r['cost_typical_rupees']:,.0f}")
    m3.metric("High end", f"₹{r['cost_high_rupees']:,.0f}")
    st.caption("Low end to high end is the 80% range: about 8 in 10 similar hospitalizations fall inside it.")
    st.metric("Catastrophic-cost risk", r["che_risk"], f"{100 * r['che_probability']:.0f}% chance", delta_color="off")
    if r["che_flag"]:
        st.warning("Flag: this hospitalization is likely to take more than 10% of the household's annual consumption.")
    if "share_of_annual_consumption" in r:
        s = r["share_of_annual_consumption"]
        pct = lambda v: "under 1%" if v < .005 else f"about {100 * v:.0f}%"
        st.info(f"Medical cost alone: {pct(s['typical'])} of annual consumption (typical) and {pct(s['high'])} (high end).")
    st.subheader("What drives this estimate")
    for k, title in (("raises_cost", "Raises the estimate"), ("lowers_cost", "Lowers the estimate")):
        st.write(f"**{title}:** " + (", ".join(n for n, _ in r["drivers"][k]) or "no single dominant factor"))
    st.subheader("Financial-aid checklist")
    st.caption("Questions to ask and documents to gather. Items are suggestions, not eligibility decisions.")
    st.markdown("\n".join(f"{i}. {item}" for i, item in enumerate(r["checklist"], 1)))
    st.caption(r["disclaimer"])
