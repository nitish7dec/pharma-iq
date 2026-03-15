import os
import streamlit as st
import anthropic
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

st.set_page_config(
    page_title="PharmaIQ — Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    PINECONE_API_KEY  = st.secrets["PINECONE_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv("/Users/nitishkaushik/Desktop/pharma_rag/.env", override=True)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")

PINECONE_INDEX  = "pharma-iq"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL    = "claude-sonnet-4-6"
MAX_TOKENS      = 1500
PFE_COLOR       = "#1D4ED8"
LLY_COLOR       = "#BE123C"
PLOTLY_CFG      = {"displayModeBar": False, "responsive": True}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#374151"),
)

SYSTEM_PROMPT = """You are an expert pharmaceutical industry analyst with deep knowledge
of financial reporting, clinical trials, and regulatory filings.

You have access to:
- Pfizer Annual Reports (10-K): 2023, 2024
- Eli Lilly Annual Reports (10-K): 2023, 2025
- Pfizer Clinical Trials: 803 trials (ClinicalTrials.gov)
- Eli Lilly Clinical Trials: 775 trials (ClinicalTrials.gov)

You are a dynamic assistant. You must automatically:
- Detect which company the user is asking about
- Detect which year they are referring to
- Detect whether they want financial data or clinical trial data
- Compare across companies or years if asked

Rules you must STRICTLY follow:
1. Answer ONLY from the provided context. Never use general knowledge or training data.
2. Always cite every fact: (Source: Company | Doc Type | Year | Page X)
3. If context is insufficient say exactly:
   "I don't have sufficient data in the provided documents to answer this."
4. For ALL numbers be precise — always include units (millions, billions, %).
5. For multi-company questions ALWAYS use structured headings for each company.
6. Never fabricate, estimate, or guess any data, statistics, or trial results.
7. If a company or year is not in your data say:
   "I only have data for Pfizer (2023, 2024) and Eli Lilly (2023, 2025)."
8. If you find partial data, clearly state what you found and what is missing.
9. Never say a figure is unavailable if it appears anywhere in the context.
10. For comparison questions — always attempt to answer for BOTH companies.
    If one is missing, state clearly which company's data was not retrieved.
11. Structure your answers with clear headings, bullet points and tables where appropriate.
12. Always end comparison answers with a brief Summary section.
"""

# ══════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: #F1F5F9; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.kpi-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:18px 16px; position:relative; overflow:hidden; transition:all 0.2s; min-height:100px; }
.kpi-card:hover { box-shadow:0 6px 20px rgba(0,0,0,0.07); transform:translateY(-2px); }
.kpi-accent { position:absolute; top:0; left:0; right:0; height:3px; }
.kpi-label { font-size:10px; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
.kpi-value { font-size:22px; font-weight:800; color:#0F172A; letter-spacing:-0.5px; font-family:'JetBrains Mono',monospace; line-height:1; }
.kpi-sub { font-size:11px; color:#94A3B8; margin-top:5px; font-weight:500; }
.kpi-tag { display:inline-block; padding:2px 8px; border-radius:6px; font-size:10px; font-weight:700; margin-top:4px; }
.chart-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:22px 24px; margin-bottom:16px; }
.chart-title { font-size:14px; font-weight:700; color:#0F172A; margin-bottom:2px; }
.chart-sub { font-size:11px; color:#94A3B8; margin-bottom:16px; }
.insight { background:#F0F9FF; border:1px solid #BAE6FD; border-left:4px solid #0EA5E9; border-radius:8px; padding:10px 14px; font-size:11.5px; color:#0C4A6E; line-height:1.7; margin-top:12px; }
.insight strong { color:#0369A1; }
.sec-hdr { display:flex; align-items:center; justify-content:space-between; margin:24px 0 14px; padding-bottom:12px; border-bottom:2px solid #E2E8F0; }
.sec-title { font-size:18px; font-weight:800; color:#0F172A; }
.sec-badge { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; border-radius:20px; padding:3px 12px; font-size:11px; font-weight:700; }
.live-badge { display:inline-flex; align-items:center; gap:6px; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:20px; padding:4px 12px; font-size:11px; font-weight:700; color:#15803D; }
.live-dot { width:6px; height:6px; background:#22C55E; border-radius:50%; animation:blink 2s infinite; display:inline-block; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
.chat-hero { background:linear-gradient(135deg,#1D4ED8 0%,#0EA5E9 100%); border-radius:16px; padding:32px 40px; color:white; margin-bottom:20px; }
.chat-hero h2 { font-size:24px; font-weight:800; margin:0 0 6px; letter-spacing:-0.5px; }
.chat-hero p { font-size:13px; opacity:0.85; margin:0 0 16px; max-width:500px; }
.hero-chip { background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25); border-radius:20px; padding:4px 12px; font-size:11px; font-weight:600; color:rgba(255,255,255,0.95); display:inline-block; margin:3px; }
.stButton > button { background:#FFFFFF !important; border:1px solid #E2E8F0 !important; border-radius:8px !important; color:#374151 !important; font-size:12px !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:500 !important; padding:8px 12px !important; width:100% !important; white-space:normal !important; height:auto !important; line-height:1.4 !important; transition:all 0.15s !important; text-align:left !important; }
.stButton > button:hover { background:#EFF6FF !important; border-color:#93C5FD !important; color:#1D4ED8 !important; }
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:4px 0 !important; }
[data-testid="stChatInput"] { background:#FFFFFF !important; border:2px solid #E2E8F0 !important; border-radius:14px !important; font-family:'Plus Jakarta Sans',sans-serif !important; font-size:14px !important; color:#0F172A !important; box-shadow:0 2px 8px rgba(0,0,0,0.04) !important; }
[data-testid="stChatInput"]:focus-within { border-color:#1D4ED8 !important; box-shadow:0 0 0 3px rgba(29,78,216,0.08) !important; }
[data-testid="stExpander"] { background:#F8FAFC !important; border:1px solid #E2E8F0 !important; border-radius:8px !important; }
.source-pill { display:inline-block; padding:2px 8px; border-radius:5px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; margin:2px; }
.sp-p { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }
.sp-l { background:#FFF1F2; color:#BE123C; border:1px solid #FECDD3; }
.tok-info { display:inline-flex; align-items:center; gap:8px; margin-top:6px; padding:3px 10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:20px; font-size:10px; color:#94A3B8; font-family:'JetBrains Mono',monospace; }
.stSpinner > div { border-top-color:#1D4ED8 !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#F1F5F9; }
::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
if "view"         not in st.session_state: st.session_state.view         = "dashboard"
if "messages"     not in st.session_state: st.session_state.messages     = []
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
if "total_cost"   not in st.session_state: st.session_state.total_cost   = 0.0
if "query_count"  not in st.session_state: st.session_state.query_count  = 0

# ══════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

@st.cache_resource
def load_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)

# ══════════════════════════════════════════════════
# QUERY PIPELINE — Best Retrieval Strategy
# ══════════════════════════════════════════════════
def rewrite_query(q):
    """Fix typos and rephrase for better retrieval"""
    c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    m = c.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 80,
        messages   = [{"role":"user","content":f"""Fix any typos and rewrite this question clearly for document search.
Return ONLY the rewritten question, nothing else.
Question: {q}"""}]
    )
    return m.content[0].text.strip()


def query_pinecone(index, qvec, company, doc_type, top_k):
    """Query Pinecone with company + doc_type filter"""
    try:
        res = index.query(
            vector           = qvec,
            top_k            = top_k,
            include_metadata = True,
            filter           = {
                "$and": [
                    {"company":  {"$eq": company}},
                    {"doc_type": {"$eq": doc_type}},
                ]
            }
        )
        return res["matches"]
    except Exception as e:
        return []


def query_pipeline(user_question):
    # Step 1 — Fix typos
    clean = rewrite_query(user_question)

    # Step 2 — Embed
    model = get_embed_model()
    qvec  = model.encode(clean, normalize_embeddings=True).tolist()

    # Step 3 — Smart retrieval — equal from BOTH companies
    index = load_pinecone()

    # 5 annual chunks per company = 10 total annual
    pfe_annual = query_pinecone(index, qvec, "pfizer",    "annual",   5)
    lly_annual = query_pinecone(index, qvec, "eli_lilly", "annual",   5)

    # 2 clinical chunks per company = 4 total clinical
    pfe_clin   = query_pinecone(index, qvec, "pfizer",    "clinical", 2)
    lly_clin   = query_pinecone(index, qvec, "eli_lilly", "clinical", 2)

    # Combine all 14 chunks
    all_matches = pfe_annual + lly_annual + pfe_clin + lly_clin

    results = []
    for match in all_matches:
        results.append((
            match["metadata"].get("text", ""),
            match["metadata"],
            1 - match["score"]
        ))

    # Step 4 — Build rich context
    context = ""
    for i, (doc, meta, score) in enumerate(results):
        context += f"""
--- Source {i+1} ---
Company      : {meta.get('company','').upper()}
Document Type: {meta.get('doc_type','').title()} Report
Year         : {meta.get('year','')}
Page         : {meta.get('page','')}
Relevance    : {round((1-score)*100, 1)}%
Content      : {doc}
"""

    # Step 5 — Build prompt
    prompt = f"""You have been provided with {len(results)} source excerpts from pharmaceutical documents.

{context}

USER QUESTION: {clean}

Instructions:
- Answer comprehensively using ALL relevant sources above
- For comparison questions, provide data for BOTH Pfizer and Eli Lilly
- Use clear headings for each company
- Cite every fact with its source
- End with a brief Summary if comparing companies
- If data for one company is missing from context, explicitly state this
"""

    # Step 6 — Call Claude
    claude  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg     = claude.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = MAX_TOKENS,
        system     = SYSTEM_PROMPT,
        messages   = [{"role":"user","content":prompt}]
    )

    answer     = msg.content[0].text
    tokens_in  = msg.usage.input_tokens
    tokens_out = msg.usage.output_tokens
    cost       = (tokens_in*3/1_000_000) + (tokens_out*15/1_000_000)

    # Deduplicated sources
    sources = []
    seen    = set()
    for _, meta, score in results:
        k = f"{meta.get('company')}|{meta.get('doc_type')}|{meta.get('year')}|{meta.get('page')}"
        if k not in seen:
            seen.add(k)
            sources.append({**meta, "score": round(score, 4)})

    return answer, tokens_in+tokens_out, cost, sources

# ══════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════
n1, n2, n3 = st.columns([3,4,3])
with n1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 0;">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#1D4ED8,#0EA5E9);
             border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;">💊</div>
        <div>
            <div style="font-size:20px;font-weight:800;color:#0F172A;letter-spacing:-0.5px;">PharmaIQ</div>
            <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;text-transform:uppercase;">Intelligence Platform</div>
        </div>
    </div>""", unsafe_allow_html=True)

with n2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3)
    with t1:
        if st.button("🏠  Dashboard", key="nd",
                     type="primary" if st.session_state.view=="dashboard" else "secondary",
                     use_container_width=True):
            st.session_state.view="dashboard"; st.rerun()
    with t2:
        if st.button("💬  Chatbot", key="nc",
                     type="primary" if st.session_state.view=="chatbot" else "secondary",
                     use_container_width=True):
            st.session_state.view="chatbot"; st.rerun()
    with t3:
        if st.button("📊  Visualizations", key="nv",
                     type="primary" if st.session_state.view=="viz" else "secondary",
                     use_container_width=True):
            st.session_state.view="viz"; st.rerun()

with n3:
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;align-items:center;gap:12px;padding:10px 0;">
        <span class="live-badge"><span class="live-dot"></span>Claude Sonnet 4.6</span>
        <span style="font-size:11px;color:#94A3B8;font-family:'JetBrains Mono',monospace;">
            Q:<b style="color:#1D4ED8">{st.session_state.query_count}</b>
            &nbsp;·&nbsp;$<b style="color:#1D4ED8">{st.session_state.total_cost:.3f}</b>
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:0 0 4px'>", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════
if st.session_state.view == "dashboard":

    st.markdown('<div class="sec-hdr" style="margin-top:20px;"><div class="sec-title">Executive Intelligence Dashboard</div><div class="sec-badge">FY 2023–2025 · Real Data</div></div>', unsafe_allow_html=True)

    cols = st.columns(6)
    kpis = [
        ("Pfizer Revenue","$58.5B","FY2023 · ↑8.8% in FY2024","linear-gradient(90deg,#1D4ED8,#60A5FA)","#EFF6FF","#1D4ED8","Pfizer"),
        ("Eli Lilly Revenue","$34.1B","FY2023 · ↑31.9% by FY2025","linear-gradient(90deg,#BE123C,#F87171)","#FFF1F2","#BE123C","Eli Lilly"),
        ("Pfizer R&D","$10.7B","18.3% of revenue","linear-gradient(90deg,#1D4ED8,#60A5FA)","#EFF6FF","#1D4ED8","Innovation"),
        ("Eli Lilly R&D","$9.2B","27.1% of revenue","linear-gradient(90deg,#BE123C,#F87171)","#FFF1F2","#BE123C","Innovation"),
        ("Pfizer Trials","803","Clinical pipeline","linear-gradient(90deg,#0369A1,#38BDF8)","#F0F9FF","#0369A1","Pipeline"),
        ("Eli Lilly Trials","775","Clinical pipeline","linear-gradient(90deg,#0369A1,#38BDF8)","#F0F9FF","#0369A1","Pipeline"),
    ]
    for col,(label,value,sub,grad,tbg,tc,tt) in zip(cols,kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-accent" style="background:{grad}"></div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div><div class="kpi-tag" style="background:{tbg};color:{tc}">{tt}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr"><div class="sec-title">💰 Financial Performance</div><div class="sec-badge">10-K Annual Reports</div></div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Revenue Comparison</div><div class="chart-sub">Total revenue USD millions · across available fiscal years</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Pfizer", x=["FY2023","FY2024"], y=[58496,63625],
            marker_color=PFE_COLOR, text=["$58.5B","$63.6B"],
            textposition="outside", textfont=dict(size=11,color=PFE_COLOR,family="JetBrains Mono"),width=0.35))
        fig.add_trace(go.Bar(name="Eli Lilly", x=["FY2023","FY2025"], y=[34124,45042],
            marker_color=LLY_COLOR, text=["$34.1B","$45.0B"],
            textposition="outside", textfont=dict(size=11,color=LLY_COLOR,family="JetBrains Mono"),width=0.35))
        fig.update_layout(**BASE_LAYOUT, height=270, barmode="group",
            margin=dict(l=0,r=0,t=30,b=0),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=11)),
            xaxis=dict(showgrid=False,tickfont=dict(size=11)),
            yaxis=dict(showgrid=True,gridcolor="#F1F5F9",tickprefix="$",ticksuffix="M",tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 <strong>Eli Lilly grew 31.9%</strong> driven by GLP-1 drugs (Mounjaro, Zepbound). Pfizer recovered 8.8% in FY2024 after the post-COVID revenue cliff.</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Financial Health — FY 2023</div><div class="chart-sub">Revenue · Gross Profit · R&D · Net Income in USD millions</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        cats = ["Revenue","Gross Profit","R&D Spend","Net Income"]
        fig2.add_trace(go.Bar(name="Pfizer", x=cats, y=[58496,40941,10679,2233],
            marker_color=PFE_COLOR, opacity=0.85,
            text=["$58.5B","$40.9B","$10.7B","$2.2B"],
            textposition="outside", textfont=dict(size=10,color=PFE_COLOR,family="JetBrains Mono")))
        fig2.add_trace(go.Bar(name="Eli Lilly", x=cats, y=[34124,28496,9234,5240],
            marker_color=LLY_COLOR, opacity=0.85,
            text=["$34.1B","$28.5B","$9.2B","$5.2B"],
            textposition="outside", textfont=dict(size=10,color=LLY_COLOR,family="JetBrains Mono")))
        fig2.update_layout(**BASE_LAYOUT, height=270, barmode="group",
            margin=dict(l=0,r=0,t=30,b=0),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=11)),
            xaxis=dict(showgrid=False,tickfont=dict(size=11)),
            yaxis=dict(showgrid=True,gridcolor="#F1F5F9",tickprefix="$",ticksuffix="M",tickfont=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 <strong>Eli Lilly net margin (15.4%)</strong> far exceeds Pfizer (3.8%) in FY2023. Despite lower revenue, Eli Lilly converts more revenue to profit.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr"><div class="sec-title">🎯 Strategic Positioning</div><div class="sec-badge">Gartner · McKinsey Framework</div></div>', unsafe_allow_html=True)
    c3,c4 = st.columns(2)

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Pharma Magic Quadrant</div><div class="chart-sub">Execution strength vs Innovation vision · Gartner-style</div>', unsafe_allow_html=True)
        fig_mq = go.Figure()
        for x0,x1,y0,y1,col,lbl in [
            (0,5,5,10,"rgba(239,246,255,0.5)","VISIONARIES"),
            (5,10,5,10,"rgba(240,253,244,0.5)","LEADERS"),
            (0,5,0,5,"rgba(249,250,251,0.5)","NICHE PLAYERS"),
            (5,10,0,5,"rgba(255,251,235,0.5)","CHALLENGERS"),
        ]:
            fig_mq.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=col,line=dict(width=0),layer="below")
            fig_mq.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2,text=lbl,
                font=dict(size=9,color="#C8D5E0",family="Plus Jakarta Sans"),showarrow=False)
        fig_mq.add_shape(type="line",x0=5,x1=5,y0=0,y1=10,line=dict(color="#E2E8F0",width=1,dash="dot"))
        fig_mq.add_shape(type="line",x0=0,x1=10,y0=5,y1=5,line=dict(color="#E2E8F0",width=1,dash="dot"))
        fig_mq.add_trace(go.Scatter(x=[7.1],y=[9.1],mode="markers+text",
            marker=dict(size=28,color=LLY_COLOR,opacity=0.9,line=dict(color="white",width=2)),
            text=["Eli Lilly"],textposition="middle center",
            textfont=dict(color="white",size=10,family="Plus Jakarta Sans"),
            showlegend=False,hovertemplate="<b>Eli Lilly</b><br>Execution: 7.1/10<br>Vision: 9.1/10<extra></extra>"))
        fig_mq.add_trace(go.Scatter(x=[8.2],y=[6.8],mode="markers+text",
            marker=dict(size=28,color=PFE_COLOR,opacity=0.9,line=dict(color="white",width=2)),
            text=["Pfizer"],textposition="middle center",
            textfont=dict(color="white",size=10,family="Plus Jakarta Sans"),
            showlegend=False,hovertemplate="<b>Pfizer</b><br>Execution: 8.2/10<br>Vision: 6.8/10<extra></extra>"))
        fig_mq.update_layout(**BASE_LAYOUT, height=280, showlegend=False,
            margin=dict(l=0,r=0,t=30,b=0),
            xaxis=dict(title="Ability to Execute →",range=[0,10],showgrid=False,zeroline=False,
                title_font=dict(size=10,color="#64748B"),tickfont=dict(size=9)),
            yaxis=dict(title="↑ Completeness of Vision",range=[0,10],showgrid=False,zeroline=False,
                title_font=dict(size=10,color="#64748B"),tickfont=dict(size=9)))
        st.plotly_chart(fig_mq, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 <strong>Eli Lilly leads on Vision</strong> — R&D intensity 27.1%, GLP-1 dominance, Alzheimer pipeline. <strong>Pfizer leads on Execution</strong> — larger revenue base, global commercial reach.</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Innovation Radar</div><div class="chart-sub">6-dimension strategic comparison</div>', unsafe_allow_html=True)
        dims  = ["R&D Intensity","Pipeline Depth","Revenue Scale","Profit Margin","Phase 3 Count","Therapeutic Diversity"]
        pfe_s = [6.5,8.0,9.5,3.5,7.0,8.5]
        lly_s = [9.0,7.5,6.5,8.0,8.5,7.0]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=pfe_s+[pfe_s[0]],theta=dims+[dims[0]],fill="toself",
            name="Pfizer",line=dict(color=PFE_COLOR,width=2),fillcolor="rgba(29,78,216,0.1)",
            marker=dict(size=5,color=PFE_COLOR)))
        fig_r.add_trace(go.Scatterpolar(r=lly_s+[lly_s[0]],theta=dims+[dims[0]],fill="toself",
            name="Eli Lilly",line=dict(color=LLY_COLOR,width=2),fillcolor="rgba(190,18,60,0.08)",
            marker=dict(size=5,color=LLY_COLOR)))
        fig_r.update_layout(**BASE_LAYOUT, height=280,
            margin=dict(l=0,r=0,t=10,b=40),
            legend=dict(orientation="h",yanchor="bottom",y=-0.18,xanchor="center",x=0.5,font=dict(size=11)),
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,range=[0,10],tickfont=dict(size=8),gridcolor="#E2E8F0",linecolor="#E2E8F0"),
                angularaxis=dict(tickfont=dict(size=9,family="Plus Jakarta Sans"),gridcolor="#E2E8F0",linecolor="#E2E8F0")))
        st.plotly_chart(fig_r, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 Eli Lilly dominates <strong>R&D Intensity & Profit Margin</strong>. Pfizer leads on <strong>Revenue Scale & Therapeutic Diversity</strong>.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr"><div class="sec-title">🧬 Clinical Trial Intelligence</div><div class="sec-badge">ClinicalTrials.gov · 1,578 Trials</div></div>', unsafe_allow_html=True)
    c5,c6,c7 = st.columns(3)

    with c5:
        st.markdown('<div class="chart-card"><div class="chart-title">Pfizer · Phase Distribution</div><div class="chart-sub">803 total trials</div>', unsafe_allow_html=True)
        fig_d1 = go.Figure(go.Pie(
            labels=["Phase 1","Phase 2","Phase 3","Ph 1/2","Ph 2/3"],
            values=[434,149,174,29,17], hole=0.60,
            marker=dict(colors=["#DBEAFE","#93C5FD","#1D4ED8","#BFDBFE","#60A5FA"],line=dict(color="white",width=2)),
            textinfo="percent+label", textfont=dict(size=9,family="Plus Jakarta Sans"),
            hovertemplate="<b>%{label}</b><br>%{value} trials<extra></extra>"))
        fig_d1.add_annotation(text="<b>803</b><br>Trials",x=0.5,y=0.5,
            font=dict(size=13,color="#0F172A",family="Plus Jakarta Sans"),showarrow=False)
        fig_d1.update_layout(**BASE_LAYOUT, height=240, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_d1, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="chart-card"><div class="chart-title">Eli Lilly · Phase Distribution</div><div class="chart-sub">775 total trials</div>', unsafe_allow_html=True)
        fig_d2 = go.Figure(go.Pie(
            labels=["Phase 1","Phase 2","Phase 3","Ph 1/2","Ph 2/3"],
            values=[441,118,197,14,4], hole=0.60,
            marker=dict(colors=["#FEE2E2","#FCA5A5","#BE123C","#FECDD3","#F87171"],line=dict(color="white",width=2)),
            textinfo="percent+label", textfont=dict(size=9,family="Plus Jakarta Sans"),
            hovertemplate="<b>%{label}</b><br>%{value} trials<extra></extra>"))
        fig_d2.add_annotation(text="<b>775</b><br>Trials",x=0.5,y=0.5,
            font=dict(size=13,color="#0F172A",family="Plus Jakarta Sans"),showarrow=False)
        fig_d2.update_layout(**BASE_LAYOUT, height=240, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_d2, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with c7:
        st.markdown('<div class="chart-card"><div class="chart-title">Trial Status Breakdown</div><div class="chart-sub">Active vs completed vs terminated</div>', unsafe_allow_html=True)
        fig_st = go.Figure()
        for s,pv,lv,c in zip(
            ["Completed","Recruiting","Active","Terminated","Withdrawn"],
            [500,89,71,118,18],[506,109,85,53,13],
            ["#22C55E","#1D4ED8","#F59E0B","#EF4444","#94A3B8"]):
            fig_st.add_trace(go.Bar(name=s,x=["Pfizer","Eli Lilly"],y=[pv,lv],
                marker_color=c,text=[pv,lv],textposition="inside",textfont=dict(size=9,color="white")))
        fig_st.update_layout(**BASE_LAYOUT, height=240, barmode="stack",
            margin=dict(l=0,r=70,t=0,b=0),
            legend=dict(orientation="v",x=1.02,y=0.5,font=dict(size=8)),
            xaxis=dict(showgrid=False,tickfont=dict(size=11)),
            yaxis=dict(showgrid=True,gridcolor="#F1F5F9",tickfont=dict(size=9)))
        st.plotly_chart(fig_st, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Therapeutic Area Heatmap</div><div class="chart-sub">Trial count per disease area · Darker = more trials · Reveals strategic focus</div>', unsafe_allow_html=True)
    areas = ["Oncology","Diabetes","Immunology","Obesity/Metabolic","Neurology","Infectious Disease","Cardiovascular"]
    fig_h = go.Figure(go.Heatmap(
        z=[[151,13,67,17,25,91,4],[83,130,94,73,42,8,6]],
        x=areas, y=["Pfizer","Eli Lilly"],
        text=[[str(v) for v in [151,13,67,17,25,91,4]],[str(v) for v in [83,130,94,73,42,8,6]]],
        texttemplate="%{text}", textfont=dict(size=13,family="JetBrains Mono"),
        colorscale=[[0,"#F8FAFC"],[0.2,"#DBEAFE"],[0.5,"#60A5FA"],[0.8,"#1D4ED8"],[1,"#1E3A8A"]],
        showscale=True, colorbar=dict(thickness=10,tickfont=dict(size=9)),
        hovertemplate="<b>%{y}</b><br>%{x}: <b>%{z} trials</b><extra></extra>"))
    fig_h.update_layout(**BASE_LAYOUT, height=180,
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False,tickfont=dict(size=11)),
        yaxis=dict(showgrid=False,tickfont=dict(size=12)))
    st.plotly_chart(fig_h, use_container_width=True, config=PLOTLY_CFG)
    st.markdown('<div class="insight">💡 <strong>Pfizer dominates Oncology (151) & Infectious Disease (91)</strong>. <strong>Eli Lilly leads Diabetes (130), Obesity (73) & Immunology (94)</strong> — aligned with Mounjaro, Zepbound & Taltz.</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# CHATBOT
# ════════════════════════════════════════════════
elif st.session_state.view == "chatbot":

    st.markdown("""
    <div class="chat-hero">
        <h2>AI Research Assistant</h2>
        <p>Ask anything about the companies in our knowledge base — financials, clinical trials,
           R&D strategy, risk factors and more. No filters needed.</p>
        <div>
            <span class="hero-chip">📄 Annual Reports</span>
            <span class="hero-chip">🧬 Clinical Trials</span>
            <span class="hero-chip">🤖 Claude Sonnet 4.6</span>
            <span class="hero-chip">⚡ Pinecone RAG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("<p style='font-size:12px;color:#64748B;font-weight:600;margin-bottom:8px;'>💡 Suggested questions</p>", unsafe_allow_html=True)
        sc = st.columns(2)
        suggestions = [
            "What was Pfizer's total revenue in 2023?",
            "Compare R&D expenses of Pfizer and Eli Lilly in 2023",
            "What Phase 3 trials does Eli Lilly have?",
            "What is Pfizer's oncology strategy?",
            "Which Eli Lilly drugs could be next revenue drivers?",
            "What risks did Pfizer mention in their 2024 annual report?",
            "How many recruiting trials does Pfizer have?",
            "Compare both companies clinical trial pipeline",
        ]
        for i,sugg in enumerate(suggestions):
            with sc[i%2]:
                if st.button(sugg, key=f"s_{i}", use_container_width=True):
                    st.session_state.pending_q = sugg

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "💊"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📎 {len(msg['sources'])} sources used"):
                        for src in msg["sources"]:
                            css = "sp-p" if src.get("company")=="pfizer" else "sp-l"
                            co  = "Pfizer" if src.get("company")=="pfizer" else "Eli Lilly"
                            ico = "📄" if src.get("doc_type")=="annual" else "🧬"
                            st.markdown(f'<span class="source-pill {css}">{ico} {co} · {src.get("doc_type","").title()} · {src.get("year","")} · p.{src.get("page","")}</span>', unsafe_allow_html=True)
                if "tokens" in msg:
                    st.markdown(f'<div class="tok-info">🔢 {msg["tokens"]:,} tokens &nbsp;·&nbsp; 💰 ${msg["cost"]:.5f}</div>', unsafe_allow_html=True)

    def run_query(question):
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="💊"):
            with st.spinner("Analyzing documents..."):
                answer, tokens, cost, sources = query_pipeline(question)
            st.markdown(answer)
            if sources:
                with st.expander(f"📎 {len(sources)} sources used"):
                    for src in sources:
                        css = "sp-p" if src.get("company")=="pfizer" else "sp-l"
                        co  = "Pfizer" if src.get("company")=="pfizer" else "Eli Lilly"
                        ico = "📄" if src.get("doc_type")=="annual" else "🧬"
                        st.markdown(f'<span class="source-pill {css}">{ico} {co} · {src.get("doc_type","").title()} · {src.get("year","")} · p.{src.get("page","")}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="tok-info">🔢 {tokens:,} tokens &nbsp;·&nbsp; 💰 ${cost:.5f}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({
            "role":"assistant","content":answer,
            "sources":sources,"tokens":tokens,"cost":cost
        })
        st.session_state.total_tokens += tokens
        st.session_state.total_cost   += cost
        st.session_state.query_count  += 1
        st.rerun()

    if "pending_q" in st.session_state:
        q = st.session_state.pending_q
        del st.session_state.pending_q
        run_query(q)

    if prompt := st.chat_input("Ask anything about the companies in our knowledge base..."):
        run_query(prompt)

    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clr"):
            st.session_state.messages=[]
            st.session_state.total_tokens=0
            st.session_state.total_cost=0.0
            st.session_state.query_count=0
            st.rerun()

# ════════════════════════════════════════════════
# VISUALIZATIONS
# ════════════════════════════════════════════════
elif st.session_state.view == "viz":

    st.markdown('<div class="sec-hdr" style="margin-top:20px;"><div class="sec-title">Advanced Analytics & Deep Dive</div><div class="sec-badge">Extended Visualizations · Real Data</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Pfizer Revenue Journey — The COVID Cliff Story</div><div class="chart-sub">USD millions · FY2021 → FY2024</div>', unsafe_allow_html=True)
    fig_wf = go.Figure(go.Waterfall(
        x        = ["FY2021","FY2022","FY2023 Drop","FY2023","FY2024 Growth","FY2024"],
        y        = [81288,18042,-41876,0,5129,0],
        measure  = ["absolute","relative","relative","absolute","relative","absolute"],
        text     = ["$81.3B","+$19.0B","-$41.8B","$58.5B","+$5.1B","$63.6B"],
        textposition = "outside",
        textfont = dict(size=11,family="JetBrains Mono"),
        increasing = dict(marker=dict(color="#22C55E")),
        decreasing = dict(marker=dict(color="#EF4444")),
        totals   = dict(marker=dict(color=PFE_COLOR)),
        connector = dict(line=dict(color="#E2E8F0",width=1,dash="dot"))))
    fig_wf.update_layout(**BASE_LAYOUT, height=300,
        margin=dict(l=0,r=0,t=30,b=0),
        xaxis=dict(showgrid=False,tickfont=dict(size=11),type="category"),
        yaxis=dict(showgrid=True,gridcolor="#F1F5F9",tickprefix="$",ticksuffix="M",tickfont=dict(size=10)))
    st.plotly_chart(fig_wf, use_container_width=True, config=PLOTLY_CFG)
    st.markdown('<div class="insight">💡 Pfizer surged from <strong>$81B (FY2021) to $100B (FY2022)</strong> on COVID vaccine & Paxlovid — crashed to <strong>$58.5B in FY2023</strong>. Recovering to <strong>$63.6B in FY2024</strong>.</div></div>', unsafe_allow_html=True)

    cv1,cv2 = st.columns(2)

    with cv1:
        st.markdown('<div class="chart-card"><div class="chart-title">R&D Efficiency — Bubble Analysis</div><div class="chart-sub">R&D spend vs Net Income · Bubble size = Revenue</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        for co,rd,ni,rev,clr in zip(
            ["Pfizer FY23","Pfizer FY24","Eli Lilly FY23","Eli Lilly FY25"],
            [10679,10953,9234,12155],[2233,8030,5240,10591],
            [58496,63625,34124,45042],[PFE_COLOR,PFE_COLOR,LLY_COLOR,LLY_COLOR]):
            fig_sc.add_trace(go.Scatter(x=[rd],y=[ni],mode="markers+text",
                marker=dict(size=rev/1200,color=clr,opacity=0.75,line=dict(color="white",width=2)),
                text=[co],textposition="top center",textfont=dict(size=10,family="Plus Jakarta Sans"),
                showlegend=False,
                hovertemplate=f"<b>{co}</b><br>R&D: ${rd:,}M<br>Net: ${ni:,}M<extra></extra>"))
        fig_sc.update_layout(**BASE_LAYOUT, height=280,
            margin=dict(l=0,r=0,t=30,b=0),
            xaxis=dict(title="R&D Spend ($M)",showgrid=True,gridcolor="#F1F5F9",tickfont=dict(size=10)),
            yaxis=dict(title="Net Income ($M)",showgrid=True,gridcolor="#F1F5F9",tickfont=dict(size=10)))
        st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 <strong>Eli Lilly converts R&D to profit far more efficiently</strong> — similar R&D spend but 2.3x more net income.</div></div>', unsafe_allow_html=True)

    with cv2:
        st.markdown('<div class="chart-card"><div class="chart-title">Phase 3 Pipeline — Butterfly Chart</div><div class="chart-sub">Head-to-head Phase 3 comparison by therapeutic area</div>', unsafe_allow_html=True)
        areas_p3 = ["Oncology","Immunology","Diabetes","Obesity","Neurology","Infectious"]
        pfe_p3   = [52,18,4,6,9,32]
        lly_p3   = [28,35,48,31,18,2]
        fig_p3   = go.Figure()
        fig_p3.add_trace(go.Bar(name="Pfizer",y=areas_p3,x=pfe_p3,orientation="h",
            marker_color=PFE_COLOR,opacity=0.85,text=pfe_p3,textposition="outside",
            textfont=dict(size=10,family="JetBrains Mono",color=PFE_COLOR)))
        fig_p3.add_trace(go.Bar(name="Eli Lilly",y=areas_p3,x=[-v for v in lly_p3],orientation="h",
            marker_color=LLY_COLOR,opacity=0.85,text=lly_p3,textposition="outside",
            textfont=dict(size=10,family="JetBrains Mono",color=LLY_COLOR)))
        fig_p3.update_layout(**BASE_LAYOUT, height=280, barmode="relative",
            margin=dict(l=0,r=0,t=30,b=0),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=11)),
            xaxis=dict(showgrid=False,tickvals=[-60,-40,-20,0,20,40,60],
                ticktext=["60","40","20","0","20","40","60"],tickfont=dict(size=10),
                title="← Eli Lilly  |  Pfizer →",title_font=dict(size=10,color="#64748B")),
            yaxis=dict(showgrid=False,tickfont=dict(size=10)))
        st.plotly_chart(fig_p3, use_container_width=True, config=PLOTLY_CFG)
        st.markdown('<div class="insight">💡 <strong>Eli Lilly Phase 3 dominates Diabetes, Obesity & Immunology</strong>. Pfizer leads in Oncology & Infectious Disease.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Profitability Trend Analysis</div><div class="chart-sub">Gross margin % and Net income margin % · Trajectory comparison</div>', unsafe_allow_html=True)
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=["FY2023","FY2024"],y=[70.0,70.8],mode="lines+markers+text",
        name="Pfizer Gross Margin",line=dict(color=PFE_COLOR,width=2.5),marker=dict(size=8,color=PFE_COLOR),
        text=["70.0%","70.8%"],textposition="top center",textfont=dict(size=10,family="JetBrains Mono",color=PFE_COLOR)))
    fig_m.add_trace(go.Scatter(x=["FY2023","FY2024"],y=[3.8,12.6],mode="lines+markers+text",
        name="Pfizer Net Margin",line=dict(color=PFE_COLOR,width=2,dash="dot"),marker=dict(size=8,color=PFE_COLOR),
        text=["3.8%","12.6%"],textposition="bottom center",textfont=dict(size=10,family="JetBrains Mono",color=PFE_COLOR)))
    fig_m.add_trace(go.Scatter(x=["FY2023","FY2025"],y=[83.5,86.4],mode="lines+markers+text",
        name="Eli Lilly Gross Margin",line=dict(color=LLY_COLOR,width=2.5),marker=dict(size=8,color=LLY_COLOR),
        text=["83.5%","86.4%"],textposition="top center",textfont=dict(size=10,family="JetBrains Mono",color=LLY_COLOR)))
    fig_m.add_trace(go.Scatter(x=["FY2023","FY2025"],y=[15.4,23.5],mode="lines+markers+text",
        name="Eli Lilly Net Margin",line=dict(color=LLY_COLOR,width=2,dash="dot"),marker=dict(size=8,color=LLY_COLOR),
        text=["15.4%","23.5%"],textposition="bottom center",textfont=dict(size=10,family="JetBrains Mono",color=LLY_COLOR)))
    fig_m.update_layout(**BASE_LAYOUT, height=280,
        margin=dict(l=0,r=0,t=30,b=0),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10)),
        xaxis=dict(showgrid=False,tickfont=dict(size=11),type="category"),
        yaxis=dict(showgrid=True,gridcolor="#F1F5F9",ticksuffix="%",tickfont=dict(size=10),range=[0,100]))
    st.plotly_chart(fig_m, use_container_width=True, config=PLOTLY_CFG)
    st.markdown('<div class="insight">💡 <strong>Eli Lilly gross margin (83.5-86.4%) far outperforms Pfizer (70%)</strong>. Eli Lilly net margin improving 15.4%→23.5%. Pfizer recovering strongly 3.8%→12.6%.</div></div>', unsafe_allow_html=True)
