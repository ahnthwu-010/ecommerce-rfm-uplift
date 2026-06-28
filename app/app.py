import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, json, duckdb

st.set_page_config(
    page_title="E-commerce RFM & Uplift",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL CSS

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #12151f; border-right: 1px solid #2e3350; }
[data-testid="stSidebar"] *        { color: #c9d1d9 !important; }
h1, h2, h3                         { color: #e6edf3 !important; font-weight: 500 !important; }
p, li, label, caption              { color: #8b949e !important; }
[data-testid="stMarkdownContainer"] p { color: #8b949e; }

/* ── KPI grid ── */
.kpi-grid {
    display: grid;
    gap: 14px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #1e2235;
    border-radius: 10px;
    padding: 18px 22px 14px;
    border-left: 4px solid #30363d;
    white-space: nowrap;
}
.kpi-card.blue   { border-left-color: #378ADD; }
.kpi-card.teal   { border-left-color: #1D9E75; }
.kpi-card.amber  { border-left-color: #EF9F27; }
.kpi-card.red    { border-left-color: #D85A30; }
.kpi-card.purple { border-left-color: #7F77DD; }
.kpi-card.gray   { border-left-color: #888780; }
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #8b949e;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1.15;
    white-space: nowrap;
}
.kpi-sub {
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
}

/* ── Section header ── */
.section-header {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #8b949e;
    border-bottom: 1px solid #2e3350;
    padding-bottom: 6px;
    margin: 24px 0 16px;
}

/* ── Info / finding box ── */
.finding-box {
    background: #161b22;
    border: 1px solid #2e3350;
    border-left: 4px solid #1D9E75;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 16px 0;
    color: #c9d1d9;
    font-size: 14px;
    line-height: 1.6;
}
.finding-box.warn  { border-left-color: #EF9F27; }
.finding-box.alert { border-left-color: #D85A30; }
.finding-box.info  { border-left-color: #378ADD; }

/* ── SQL code block ── */
.sql-block {
    background: #161b22;
    border: 1px solid #2e3350;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: #79c0ff;
    overflow-x: auto;
    white-space: pre;
    margin: 10px 0 16px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Sidebar nav label ── */
.nav-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #484f58;
    padding: 18px 0 6px;
}

/* ── Matplotlib chart bg ── */
.stImage img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# LOAD ARTIFACTS

@st.cache_data
def load_rfm():
    return pd.read_csv('app/rfm_segments.csv')

@st.cache_data
def load_uplift():
    return pd.read_csv('app/uplift_scores.csv')

@st.cache_resource
def load_models():
    mt = pickle.load(open('app/xgb_treatment_model.pkl', 'rb'))
    mc = pickle.load(open('app/xgb_control_model.pkl',   'rb'))
    fc = json.load(open('app/feature_cols.json', 'r'))
    return mt, mc, fc

rfm_df             = load_rfm()
uplift_df          = load_uplift()
model_t, model_c, feature_cols = load_models()

# CONSTANTS

SEG_ORDER = ['Champions','Loyal','At Risk',
             'Promising','New','Needs Attention','Lost']
SEG_COLORS = {
    'Champions':       '#1D9E75',
    'Loyal':           '#0F6E56',
    'At Risk':         '#D85A30',
    'Promising':       '#378ADD',
    'New':             '#85B7EB',
    'Needs Attention': '#EF9F27',
    'Lost':            '#888780',
}
UPLIFT_COLORS = {
    'Persuadables':    '#1D9E75',
    'Sure Things':     '#378ADD',
    'Needs Attention': '#EF9F27',
    'Lost Causes':     '#D85A30',
}

def mpl_dark(fig, *axes):
    """Apply dark background to matplotlib figures."""
    fig.patch.set_facecolor('#1e2235')
    for ax in axes:
        ax.set_facecolor('#1e2235')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#2e3350')

# SIDEBAR

with st.sidebar:
    st.markdown("## 🛒 RFM & Uplift")
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊  RFM Overview",
        "🔍  SQL Explorer",
        "🎯  Uplift Results",
        "💰  Campaign Simulator",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="nav-label">Datasets</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:12px; line-height:1.8;">
    📦 <b style="color:#c9d1d9">Module A</b><br>
    Olist Brazilian E-commerce<br>
    93,358 customers · DuckDB SQL<br><br>
    🎯 <b style="color:#c9d1d9">Module B</b><br>
    Hillstrom MineThatData<br>
    42,693 customers · RCT email
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="font-size:11px; color:#484f58; line-height:1.7;">
    Stack: DuckDB · XGBoost<br>
    scikit-learn · Streamlit<br>
    <i>Two datasets — technique demo</i>
    </p>
    """, unsafe_allow_html=True)


# PAGE 1 — RFM OVERVIEW

if page == "📊  RFM Overview":
    st.markdown("# RFM Segmentation")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-12px; margin-bottom:24px;'>"
        "Olist Brazilian E-commerce · 93,358 customers · DuckDB SQL · Sep 2016–Oct 2018</p>",
        unsafe_allow_html=True
    )

    total_rev      = rfm_df['monetary'].sum()
    repeat_n       = (rfm_df['frequency'] >= 2).sum()
    repeat_rev     = rfm_df[rfm_df['frequency'] >= 2]['monetary'].sum()
    champions_pct  = (rfm_df['segment'] == 'Champions').mean() * 100
    promising_pct  = (rfm_df['segment'] == 'Promising').mean() * 100

    # KPI row 1
    st.markdown("""
    <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
      <div class="kpi-card blue">
        <div class="kpi-label">Total customers</div>
        <div class="kpi-value">93,358</div>
        <div class="kpi-sub">Delivered orders · unique IDs</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-label">Portfolio revenue</div>
        <div class="kpi-value">$15.4M</div>
        <div class="kpi-sub">Sep 2016 – Oct 2018</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Repeat buyers</div>
        <div class="kpi-value">2,801</div>
        <div class="kpi-sub">3.0% of base · $864k revenue</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">One-time buyers</div>
        <div class="kpi-value">97.0%</div>
        <div class="kpi-sub">Acquisition-heavy market</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Segment breakdown</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = (rfm_df['segment'].value_counts()
                      .reindex(SEG_ORDER).fillna(0))
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_dark(fig, ax)
        segs = SEG_ORDER[::-1]
        vals = seg_counts.reindex(segs).values
        bars = ax.barh(segs, vals / vals.sum() * 100,
                       color=[SEG_COLORS[s] for s in segs],
                       height=0.55, zorder=2)
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{w:.1f}%  ({int(v):,})',
                    va='center', fontsize=8.5, color='#8b949e')
        ax.set_xlabel('% of customers', fontsize=9, color='#8b949e')
        ax.set_xlim(0, 52)
        ax.set_title('Customer distribution by segment',
                     fontsize=11, pad=10, color='#e6edf3')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#2e3350')
        ax.xaxis.grid(True, color='#2e3350', lw=0.5, zorder=1)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        seg_rev = (rfm_df.groupby('segment')['monetary']
                   .sum().reindex(SEG_ORDER).fillna(0))
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_dark(fig, ax)
        segs2 = SEG_ORDER[::-1]
        revs  = seg_rev.reindex(segs2).values
        ax.barh(segs2, revs / 1e6,
                color=[SEG_COLORS[s] for s in segs2],
                height=0.55, zorder=2)
        for i, (s, v) in enumerate(zip(segs2, revs)):
            ax.text(v/1e6 + 0.02, i,
                    f'${v/1e6:.2f}M  ({v/total_rev*100:.1f}%)',
                    va='center', fontsize=8.5, color='#8b949e')
        ax.set_xlabel('Revenue (USD millions)', fontsize=9, color='#8b949e')
        ax.set_xlim(0, 8)
        ax.set_title('Revenue contribution by segment',
                     fontsize=11, pad=10, color='#e6edf3')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#2e3350')
        ax.xaxis.grid(True, color='#2e3350', lw=0.5, zorder=1)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Key findings
    st.markdown("""
    <div class="finding-box info">
    <b style="color:#e6edf3">📌 Pareto insight:</b>
    Customer share và revenue share gần như bằng nhau ở mọi segment — Olist không có "whale customers".
    Revenue dominated bởi volume (Promising 39% customers = 37% revenue), không phải individual value.
    Đây là đặc trưng acquisition-heavy market, không phải loyalty-driven platform.
    </div>
    <div class="finding-box warn">
    <b style="color:#e6edf3">⚠️ Retention opportunity:</b>
    At Risk (621 khách, avg $324/head) là nhóm có value cao nhất đang rời đi.
    Retention campaign nhắm 621 người này có ROI cao nhất — chi phí thấp, value at stake cao.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Recency × Monetary space</div>',
                unsafe_allow_html=True)

    sample = rfm_df.sample(5000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 4))
    mpl_dark(fig, ax)
    for seg in SEG_ORDER:
        d = sample[sample['segment'] == seg]
        ax.scatter(d['recency'], d['monetary'].clip(upper=2000),
                   c=SEG_COLORS[seg], label=seg,
                   alpha=0.45, s=15, linewidths=0, zorder=2)
    ax.set_xlabel('Recency (days since last purchase)', fontsize=9)
    ax.set_ylabel('Monetary USD (capped $2,000)', fontsize=9)
    ax.set_title('5,000 sample customers — Recency vs Monetary',
                 fontsize=11, pad=10)
    ax.legend(fontsize=8, frameon=False, ncol=4,
              loc='upper right', labelcolor='#8b949e')
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.grid(True, color='#2e3350', lw=0.5, zorder=1)
    ax.yaxis.grid(True, color='#2e3350', lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.text(660, 1900,
            'Capped at $2,000 (max $13,664 excl.)',
            ha='right', fontsize=8, color='#484f58', style='italic')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Segment stats table
    st.markdown('<div class="section-header">Segment statistics</div>',
                unsafe_allow_html=True)
    seg_table = (rfm_df.groupby('segment')
                 .agg(customers=('customer_unique_id','count'),
                      avg_recency=('recency','mean'),
                      avg_frequency=('frequency','mean'),
                      avg_monetary=('monetary','mean'),
                      total_revenue=('monetary','sum'))
                 .reindex(SEG_ORDER)
                 .round(1))
    seg_table['revenue_pct'] = (
        seg_table['total_revenue'] / seg_table['total_revenue'].sum() * 100
    ).round(1)
    seg_table.columns = ['Customers','Avg recency (days)',
                         'Avg frequency','Avg monetary ($)',
                         'Total revenue ($)','Revenue %']
    st.dataframe(seg_table.style
                 .format({'Total revenue ($)': '{:,.0f}',
                          'Customers': '{:,}',
                          'Revenue %': '{:.1f}%'})
                 .background_gradient(subset=['Revenue %'],
                                      cmap='Blues', vmin=0, vmax=40),
                 use_container_width=True)


# PAGE 2 — SQL EXPLORER

elif page == "🔍  SQL Explorer":
    st.markdown("# SQL Explorer")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-12px; margin-bottom:24px;'>"
        "DuckDB live queries trên rfm_segments.csv · Thể hiện analytical SQL skill</p>",
        unsafe_allow_html=True
    )

    con_app = duckdb.connect()
    con_app.register('rfm', rfm_df)

    PRESETS = {
        "Revenue concentration by segment": {
            "sql": """SELECT
    segment,
    COUNT(*)                                    AS customers,
    ROUND(SUM(monetary), 0)                     AS total_revenue,
    ROUND(100.0 * SUM(monetary)
          / SUM(SUM(monetary)) OVER (), 2)      AS revenue_pct,
    ROUND(AVG(monetary), 2)                     AS avg_order_value,
    ROUND(AVG(recency), 1)                      AS avg_recency_days
FROM rfm
GROUP BY segment
ORDER BY total_revenue DESC""",
            "insight": "Window function `SUM() OVER()` tính % revenue mà không cần subquery — đây là SQL style thể hiện analytical skill."
        },
        "Top 15 Champions by monetary": {
            "sql": """SELECT
    customer_unique_id,
    recency,
    frequency,
    ROUND(monetary, 2)   AS monetary_usd,
    r_score,
    f_score,
    m_score
FROM rfm
WHERE segment = 'Champions'
ORDER BY monetary DESC
LIMIT 15""",
            "insight": "Champions chỉ 1.1% base nhưng avg $295/head — 1.8x so với one-time buyers."
        },
        "High-value Promising customers": {
            "sql": """SELECT
    customer_unique_id,
    recency,
    ROUND(monetary, 2)   AS monetary_usd,
    r_score,
    m_score
FROM rfm
WHERE segment  = 'Promising'
  AND monetary > 500
ORDER BY monetary DESC
LIMIT 20""",
            "insight": "Promising với monetary > $500: one-time buyers nhưng high-spend — convert được một số sang Loyal sẽ tăng repeat buyer revenue đáng kể."
        },
        "At Risk cohort analysis": {
            "sql": """SELECT
    segment,
    COUNT(*)                      AS customers,
    ROUND(AVG(recency),   1)      AS avg_recency_days,
    ROUND(AVG(frequency), 2)      AS avg_frequency,
    ROUND(AVG(monetary),  2)      AS avg_monetary_usd,
    ROUND(SUM(monetary),  0)      AS total_revenue,
    ROUND(MAX(monetary),  2)      AS max_monetary_usd
FROM rfm
WHERE segment IN ('At Risk', 'Cannot Lose', 'Loyal', 'Champions')
GROUP BY segment
ORDER BY avg_monetary_usd DESC""",
            "insight": "At Risk avg $324 — highest per-customer value trong toàn bộ dataset. Đây là nhóm cần retention campaign gấp nhất."
        },
        "Recency distribution per segment (NTILE)": {
            "sql": """SELECT
    segment,
    MIN(recency)    AS min_days,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY recency), 0) AS p25_days,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY recency), 0) AS median_days,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY recency), 0) AS p75_days,
    MAX(recency)    AS max_days,
    COUNT(*)        AS customers
FROM rfm
GROUP BY segment
ORDER BY median_days""",
            "insight": "Champions median recency thấp nhất — mua gần nhất. Lost median recency cao nhất — đã lâu không mua."
        },
    }

    selected = st.selectbox(
        "Preset queries",
        list(PRESETS.keys()),
        label_visibility="visible"
    )
    preset = PRESETS[selected]

    sql_text = st.text_area(
        "SQL — editable (DuckDB syntax)",
        preset["sql"],
        height=200
    )

    st.markdown(
        f'<div class="sql-block">{preset["sql"]}</div>',
        unsafe_allow_html=True
    )

    col_run, col_info = st.columns([1, 4])
    with col_run:
        run = st.button("▶  Run query", use_container_width=True)

    if run:
        try:
            result = con_app.sql(sql_text).df()
            st.success(f"{len(result):,} rows returned")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query error: {e}")

    st.markdown(
        f'<div class="finding-box info">'
        f'<b style="color:#e6edf3">💡 SQL note:</b> {preset["insight"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Schema reference</div>',
                unsafe_allow_html=True)
    schema_df = pd.DataFrame({
        'Column':      ['customer_unique_id','recency','frequency','monetary',
                        'r_score','f_score','m_score','segment'],
        'Type':        ['string','int','int','float','int(1-5)','int(1-5)','int(1-5)','string'],
        'Description': [
            'Unique customer identifier (customer_unique_id, not customer_id)',
            'Days since last purchase (lower = better)',
            'Number of distinct orders',
            'Total spend USD',
            'Recency score: 5=most recent, 1=oldest',
            'Frequency score: derived from raw value (97% have freq=1)',
            'Monetary score: NTILE(5) on spend',
            'Segment label: Champions / Loyal / At Risk / Promising / New / Needs Attention / Lost'
        ]
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="finding-box">'
        '<b style="color:#e6edf3">⚙️ DuckDB note:</b> '
        'Queries chạy in-memory trên DataFrame — full analytical SQL '
        '(window functions, PERCENTILE_CONT, QUALIFY) không cần server. '
        'Table name: <code style="color:#79c0ff">rfm</code>'
        '</div>',
        unsafe_allow_html=True
    )

# PAGE 3 — UPLIFT RESULTS

elif page == "🎯  Uplift Results":
    st.markdown("# Uplift Modeling — T-Learner")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-12px; margin-bottom:24px;'>"
        "Hillstrom MineThatData · Womens email vs No email · N=42,693 · XGBoost</p>",
        unsafe_allow_html=True
    )

    # KPI row
    st.markdown("""
    <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
      <div class="kpi-card teal">
        <div class="kpi-label">Persuadables</div>
        <div class="kpi-value">10,674</div>
        <div class="kpi-sub">Lift +15.3pp · Send voucher ✅</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Sure Things</div>
        <div class="kpi-value">10,674</div>
        <div class="kpi-sub">Lift +6.7pp · Save budget</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Needs Attention</div>
        <div class="kpi-value">14,051</div>
        <div class="kpi-sub">Lift +0.7pp · Low ROI</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Lost Causes</div>
        <div class="kpi-value">7,294</div>
        <div class="kpi-sub">Lift −7.5pp · Never target ⚠️</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Core finding
    st.markdown("""
    <div class="finding-box">
    <b style="color:#e6edf3">📌 Core finding:</b>
    Naive ATE = +4.5pp visit rate lift — nhưng T-Learner reveals strong heterogeneity:
    Persuadables có lift <b style="color:#1D9E75">+15.3pp</b> (3.4× naive),
    trong khi Lost Causes có lift <b style="color:#D85A30">−7.5pp</b> — email actively harms their conversion.
    </div>
    <div class="finding-box alert">
    <b style="color:#e6edf3">⚠️ Sleeping Dogs (Lost Causes):</b>
    7,294 customers (17.1%) — visit rate <i>giảm</i> khi nhận email (T=10.6% vs C=18.0%).
    Possible explanation: those with high organic intent perceive promotional emails as low-quality signal → backfire effect.
    Targeting this group wastes budget <i>and</i> actively suppresses conversion.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Qini curve & uplift distribution</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Recompute Qini inline for fresh plot
        def compute_qini(df, n_bins=40):
            df_s = df.sort_values('uplift_score', ascending=False).reset_index(drop=True)
            n    = len(df_s)
            n_t  = df_s['treatment'].sum()
            n_c  = n - n_t
            qx, qy, ry = [0.], [0.], [0.]
            total_t_conv = df_s[df_s['treatment']==1]['visit'].sum()
            total_c_conv = df_s[df_s['treatment']==0]['visit'].sum()
            for i in range(1, n_bins+1):
                idx    = int(i/n_bins * n)
                sub    = df_s.iloc[:idx]
                nt_sub = sub['treatment'].sum()
                nc_sub = len(sub) - nt_sub
                yt_sub = sub[sub['treatment']==1]['visit'].sum()
                yc_sub = sub[sub['treatment']==0]['visit'].sum() if nc_sub>0 else 0
                qy.append(yt_sub - yc_sub*(nt_sub/n_c) if n_c>0 else 0)
                ry.append(i/n_bins*(total_t_conv - total_c_conv*n_t/n_c))
                qx.append(i/n_bins)
            return np.array(qx), np.array(qy), np.array(ry)

        qx, qy, ry = compute_qini(uplift_df)
        auuc = np.trapz(qy, qx) - np.trapz(ry, qx)

        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_dark(fig, ax)
        ax.plot(qx, qy, color='#1D9E75', lw=2,
                label=f'T-Learner  AUUC={auuc:.0f}')
        ax.plot(qx, ry, color='#484f58',  lw=1.2,
                linestyle='--', label='Random targeting')
        ax.fill_between(qx, qy, ry, alpha=0.15, color='#1D9E75')
        ax.set_xlabel('Fraction of population targeted', fontsize=9)
        ax.set_ylabel('Incremental conversions', fontsize=9)
        ax.set_title('Qini curve — uplift model vs random', fontsize=11, pad=10)
        ax.legend(fontsize=9, frameon=False, labelcolor='#8b949e')
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.grid(True, color='#2e3350', lw=0.5)
        ax.yaxis.grid(True, color='#2e3350', lw=0.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        mpl_dark(fig, ax)
        bins = np.linspace(uplift_df['uplift_score'].min(),
                           uplift_df['uplift_score'].max(), 40)
        ax.hist(uplift_df[uplift_df['treatment']==1]['uplift_score'],
                bins=bins, alpha=0.6, color='#1D9E75',
                label='Treatment group', density=True)
        ax.hist(uplift_df[uplift_df['treatment']==0]['uplift_score'],
                bins=bins, alpha=0.6, color='#378ADD',
                label='Control group',   density=True)
        ax.axvline(0, color='#D85A30', lw=1.2, linestyle='--',
                   label='Uplift = 0')
        ax.set_xlabel('Uplift score', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title('Uplift score distribution', fontsize=11, pad=10)
        ax.legend(fontsize=8, frameon=False, labelcolor='#8b949e')
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.grid(True, color='#2e3350', lw=0.5)
        ax.yaxis.grid(True, color='#2e3350', lw=0.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Segment lift table</div>',
                unsafe_allow_html=True)

    lift_df = pd.DataFrame({
        'Segment':       ['Persuadables','Sure Things','Needs Attention','Lost Causes'],
        'Customers':     [10674, 10674, 14051, 7294],
        'T visit rate':  ['23.7%', '15.7%', '10.5%', '10.6%'],
        'C visit rate':  ['8.4%',  '9.0%',  '9.8%',  '18.0%'],
        'Lift (pp)':     ['+15.3', '+6.7',  '+0.7',  '−7.5'],
        'Action':        ['✅ Send voucher','💡 Save budget','⏸ Skip','🚫 Never target'],
    })
    st.dataframe(lift_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Feature importance — Model T vs Model C</div>',
                unsafe_allow_html=True)

    feat_names = feature_cols
    imp_t = model_t.feature_importances_
    imp_c = model_c.feature_importances_
    fig, ax = plt.subplots(figsize=(9, 3.5))
    mpl_dark(fig, ax)
    x = np.arange(len(feat_names))
    w = 0.35
    ax.bar(x - w/2, imp_t, width=w, color='#1D9E75', alpha=0.85,
           label='Model T (treatment)', zorder=2)
    ax.bar(x + w/2, imp_c, width=w, color='#378ADD', alpha=0.85,
           label='Model C (control)',   zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Feature importance', fontsize=9)
    ax.set_title('Feature importance — Model T vs Model C', fontsize=11, pad=10)
    ax.legend(fontsize=9, frameon=False, labelcolor='#8b949e')
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.grid(True, color='#2e3350', lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="finding-box info">
    <b style="color:#e6edf3">🔍 Feature divergence insight:</b>
    <code style="color:#79c0ff">womens</code> drives Model T (35%) nhưng thấp trong Model C (10%) —
    email về sản phẩm phụ nữ resonates mạnh với female customers khi có treatment.
    <code style="color:#79c0ff">newbie</code> dominates organic visit (Model C 36%) nhưng bị override bởi email effect trong Model T (19%).
    Divergence này confirm T-Learner đang capture heterogeneous response thật, không phải noise.
    </div>
    """, unsafe_allow_html=True)


# PAGE 4 — CAMPAIGN SIMULATOR

elif page == "💰  Campaign Simulator":
    st.markdown("# Campaign ROI Simulator")
    st.markdown(
        "<p style='color:#8b949e; margin-top:-12px; margin-bottom:24px;'>"
        "Simulate targeting strategy: top N% uplift score vs blast all</p>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Parameters</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        top_pct = st.slider("Target top N% by uplift score",
                            1, 100, 25,
                            help="25% = Persuadables only. 100% = blast all.")
    with col2:
        voucher_cost = st.slider("Voucher cost per email (USD)", 1, 20, 5)
    with col3:
        avg_spend = st.slider("Avg revenue per visit (USD)", 5, 50, 15)

    # Simulation 
    n_total  = len(uplift_df)
    n_target = int(n_total * top_pct / 100)
    targeted = uplift_df.nlargest(n_target, 'uplift_score')

    inc_visits_targeted = float(targeted['uplift_score'].clip(lower=0).sum())
    inc_visits_blast    = float(uplift_df['uplift_score'].clip(lower=0).sum())

    inc_rev_targeted = inc_visits_targeted * avg_spend
    inc_rev_blast    = inc_visits_blast    * avg_spend

    cost_targeted = n_target * voucher_cost
    cost_blast    = n_total  * voucher_cost

    roi_targeted = (inc_rev_targeted - cost_targeted) / cost_targeted * 100 if cost_targeted else 0
    roi_blast    = (inc_rev_blast    - cost_blast)    / cost_blast    * 100 if cost_blast    else 0

    emails_saved  = cost_blast - cost_targeted
    efficiency    = inc_visits_targeted / n_target if n_target else 0
    eff_blast     = inc_visits_blast    / n_total  if n_total  else 0
    eff_ratio     = efficiency / eff_blast if eff_blast else 0

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
      <div class="kpi-card blue">
        <div class="kpi-label">Emails sent</div>
        <div class="kpi-value">{n_target:,}</div>
        <div class="kpi-sub">vs {n_total:,} blast all</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-label">Voucher budget</div>
        <div class="kpi-value">${cost_targeted:,.0f}</div>
        <div class="kpi-sub">Saved ${emails_saved:,.0f} vs blast</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Expected incr. visits</div>
        <div class="kpi-value">{inc_visits_targeted:,.0f}</div>
        <div class="kpi-sub">blast all: {inc_visits_blast:,.0f}</div>
      </div>
      <div class="kpi-card {"teal" if roi_targeted > roi_blast else "red"}">
        <div class="kpi-label">Campaign ROI</div>
        <div class="kpi-value">{roi_targeted:.0f}%</div>
        <div class="kpi-sub">blast all: {roi_blast:.0f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Efficiency callout
    if top_pct <= 30:
        st.markdown(f"""
        <div class="finding-box">
        <b style="color:#e6edf3">✅ Targeting efficiency:</b>
        Top {top_pct}% targeting = <b style="color:#1D9E75">{eff_ratio:.1f}×</b>
        more incremental visits per email vs blast all
        ({efficiency:.3f} vs {eff_blast:.3f} visit/email).
        </div>
        """, unsafe_allow_html=True)
    elif top_pct >= 80:
        st.markdown(f"""
        <div class="finding-box alert">
        <b style="color:#e6edf3">⚠️ Warning:</b>
        Targeting {top_pct}% includes Lost Causes group (bottom 17%).
        Email <i>reduces</i> their visit rate by −7.5pp — blast all strategy wastes budget
        and actively suppresses conversion in this segment.
        </div>
        """, unsafe_allow_html=True)

    # ROI curve
    st.markdown('<div class="section-header">ROI curve across targeting percentiles</div>',
                unsafe_allow_html=True)

    pcts      = list(range(1, 101))
    rois_all  = []
    for p in pcts:
        n  = int(n_total * p / 100)
        tg = uplift_df.nlargest(n, 'uplift_score')
        iv = float(tg['uplift_score'].clip(lower=0).sum())
        ir = iv * avg_spend
        c  = n * voucher_cost
        rois_all.append((ir - c) / c * 100 if c else 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    mpl_dark(fig, ax)
    ax.plot(pcts, rois_all, color='#1D9E75', lw=2.5, zorder=3)
    ax.fill_between(pcts, rois_all, alpha=0.08, color='#1D9E75')
    ax.axvline(top_pct, color='#D85A30', lw=1.5,
               linestyle='--', label=f'Current: top {top_pct}%', zorder=4)
    ax.axvline(25, color='#EF9F27', lw=1, linestyle=':',
               label='Recommended: top 25% (Persuadables)', zorder=4)
    ax.axvline(83, color='#484f58', lw=0.8, linestyle=':',
               label='Lost Causes enter at ~83%', zorder=4)
    ax.set_xlabel('% of population targeted (sorted by uplift score, descending)', fontsize=9)
    ax.set_ylabel('Campaign ROI (%)', fontsize=9)
    ax.set_title('ROI curve — targeted vs blast all strategy', fontsize=11, pad=10)
    ax.legend(fontsize=8.5, frameon=False, labelcolor='#8b949e')
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.grid(True, color='#2e3350', lw=0.5)
    ax.yaxis.grid(True, color='#2e3350', lw=0.5)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Summary table
    st.markdown('<div class="section-header">Strategy comparison</div>',
                unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        'Strategy':          ['Top 25% (Persuadables)','Top 50%','Blast all (100%)'],
        'Emails':            [int(n_total*0.25), int(n_total*0.50), n_total],
        'Voucher cost ($)':  [int(n_total*0.25*voucher_cost),
                              int(n_total*0.50*voucher_cost),
                              int(n_total*voucher_cost)],
        'Incr. visits':      [
            int(uplift_df.nlargest(int(n_total*0.25),'uplift_score')['uplift_score'].clip(lower=0).sum()),
            int(uplift_df.nlargest(int(n_total*0.50),'uplift_score')['uplift_score'].clip(lower=0).sum()),
            int(uplift_df['uplift_score'].clip(lower=0).sum())
        ],
        'Recommendation':    ['✅ Best ROI','⚠️ Diminishing returns','🚫 Includes Sleeping Dogs'],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="finding-box info">
    <b style="color:#e6edf3">📌 Methodology note:</b>
    ROI simulation dùng uplift score trực tiếp làm proxy cho incremental conversion probability.
    Avg revenue per visit ($15 default) calibrated từ Hillstrom spend data.
    Two datasets (Olist + Hillstrom) khác nguồn — đây là technique demonstration,
    không phải end-to-end case study của một công ty cụ thể.
    </div>
    """, unsafe_allow_html=True)