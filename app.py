# ==============================
# STREAMLIT MEPS DASHBOARD BASE
# ==============================

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="MEPS Healthcare Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# LOAD LOGO
# ------------------------------
logo = Image.open("ahrq_logo.jpg")

# ------------------------------
# HEADER WITH TITLE + LOGO
# ------------------------------
col1, col2 = st.columns([7, 2])  # increased space for logo

with col1:
    st.markdown(
        """
        <h1 style='color:#023e8a; margin-bottom:0px;'>
        MEPS Healthcare Expenditure Risk Dashboard
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style='color:#4f6f8f; font-weight:400;'>
        A Risk Adjustment Model Incorporating Demographic, Lifestyle, Economic, Clinical Burden,
        and Healthcare Utilization across MEPS Panels 11–15
        </h4>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.image(logo, width=300)  # increased logo size

# ------------------------------
# SIDEBAR (PROJECT SUMMARY)
# ------------------------------
st.sidebar.title("🧠 Project Summary")

st.sidebar.markdown("""
### 🎯 Objective
Predict **Year-2 healthcare expenditure categories** (Low, Medium, High) using Year-1 data.

---

### 📦 Dataset
- MEPS Panels 11–15 (2006–2011)
- ~50K individuals
- Adult, non-cancer population

---

### ⚙️ Key Steps
- Data Cleaning & Missing Handling  
- Feature Engineering (Chronic Burden, BMI, Age)  
- Inflation Adjustment (CPI)  
- Exploratory Data Analysis  
- Patient Segmentation (K-Means)  
- Predictive Modeling  

---

### 🤖 Models Used
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

---

### 🏆 Best Model
**XGBoost**
- Accuracy: ~72%  
- AUC: ~0.84  

---

### 💡 Insights
- Costs increase sharply with age & chronic burden  
- Socioeconomic gradients strongly impact spending  
- High-cost patients are a small but critical group  

---

🚀 *Goal: Early identification of high-risk individuals for better healthcare planning*
""")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    path = "MEPS_Final.csv"
    df = pd.read_csv(path)
    return df

df = load_data()

# ------------------------------
# MAP REGION LABELS (OPTIONAL BUT CLEAN)
# ------------------------------
region_map = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West"
}

df["Region_Label"] = df["REGIONY1"].map(region_map)

# ------------------------------
# TOP FILTER BAR (RIGHT SIDE)
# ------------------------------
col_left, col_filter1, col_filter2 = st.columns([5, 2.5, 2.5])

with col_filter1:
    selected_years = st.multiselect(
    "📅 Select Year",
    options=sorted(df["BEGRFY1"].dropna().unique().astype(int)),
    default=sorted(df["BEGRFY1"].dropna().unique().astype(int)),
    placeholder="Type or select year..."
    )

with col_filter2:
    selected_regions = st.multiselect(
        "🌍 Select Region",
        options=sorted(df["Region_Label"].dropna().unique()),
        default=sorted(df["Region_Label"].dropna().unique()),
        placeholder="Type or select region..."
    )

st.markdown("""
<style>

/* Bigger dropdown box */
div[data-baseweb="select"] {
    min-height: 45px;
    border-radius: 8px;
}

/* Dropdown options */
div[role="listbox"] {
    font-size: 16px;
}

/* Selected items look like tags */
span[data-baseweb="tag"] {
    font-size: 14px;
    background-color: #3a86ff;
}

/* Label styling */
label {
    font-size: 18px !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Increase label font */
    .stMultiSelect label {
        font-size: 18px !important;
        font-weight: 600;
    }

    /* Increase dropdown text */
    .stMultiSelect div[data-baseweb="select"] {
        font-size: 16px !important;
    }

    /* Increase selected values font */
    .stMultiSelect span {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# APPLY FILTERS
# ------------------------------
df_filtered = df[
    (df["BEGRFY1"].isin(selected_years)) &
    (df["Region_Label"].isin(selected_regions))
]

# ==============================
# KPI CALCULATIONS
# ==============================

total_patients = df_filtered.shape[0]

avg_exp = df_filtered["TOTEXPY1_adj"].mean()

high_cost_pct = (
    (df_filtered["TOTEXPY2_category"] == 2).mean() * 100
)

avg_chronic = df_filtered["total_chronic_conditions"].mean()

# ------------------------------
# KPI DISPLAY
# ------------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div style='background-color:#f0f6ff; border-radius:10px; overflow:hidden; text-align:center;'>
            <div style='background-color:#0b2545; color:white; padding:8px; font-weight:600;'>
                Total Patients
            </div>
            <div style='padding:20px;'>
                <h2>{total_patients:,}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style='background-color:#f0f6ff; border-radius:10px; overflow:hidden; text-align:center;'>
            <div style='background-color:#0b2545; color:white; padding:8px; font-weight:600;'>
                Avg Expenditure
            </div>
            <div style='padding:20px;'>
                <h2>${avg_exp:,.0f}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style='background-color:#f0f6ff; border-radius:10px; overflow:hidden; text-align:center;'>
            <div style='background-color:#0b2545; color:white; padding:8px; font-weight:600;'>
                High Cost Patients (%)
            </div>
            <div style='padding:20px;'>
                <h2>{high_cost_pct:.1f}%</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div style='background-color:#f0f6ff; border-radius:10px; overflow:hidden; text-align:center;'>
            <div style='background-color:#0b2545; color:white; padding:8px; font-weight:600;'>
                Avg Chronic Conditions
            </div>
            <div style='padding:20px;'>
                <h2>{avg_chronic:.2f}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==============================
# SIDE-BY-SIDE PLOTS
# ==============================

with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)

    # -------------------------------
    # CREATE TWO COLUMNS
    # -------------------------------
    col1, col2 = st.columns(2)

    # ==============================
    # LEFT PLOT: AGE vs CHRONIC + EXP
    # ==============================
    with col1:
        st.markdown(
            "<h4 style='text-align: center;'>Chronic Conditions & Expenditure by Age and Sex</h4>",
            unsafe_allow_html=True
        )

        df_plot = df_filtered.copy()

        # Age mapping
        age_map = {
            1: '18–29',
            2: '30–44',
            3: '45–59',
            4: '60–74',
            5: '75+'
        }

        df_plot['Age_Group_Label'] = df_plot['age_group'].map(age_map)

        age_order = ['18–29', '30–44', '45–59', '60–74', '75+']

        df_plot['Age_Group_Label'] = pd.Categorical(
            df_plot['Age_Group_Label'],
            categories=age_order,
            ordered=True
        )

        # Sex mapping
        sex_map = {1: 'Male', 2: 'Female'}
        df_plot['Sex_Label'] = df_plot['SEX'].map(sex_map)

        # Aggregations
        age_sex = (
            df_plot
            .dropna(subset=['Age_Group_Label','Sex_Label','total_chronic_conditions'])
            .groupby(['Age_Group_Label','Sex_Label'])['total_chronic_conditions']
            .mean()
            .reset_index()
        )

        age_exp = (
            df_plot
            .dropna(subset=['Age_Group_Label','Sex_Label','TOTEXPY1_adj'])
            .groupby(['Age_Group_Label','Sex_Label'])['TOTEXPY1_adj']
            .mean()
            .reset_index()
        )

        # Plot
        fig1, ax1 = plt.subplots(figsize=(5.5,4.5))

        sns.barplot(
            data=age_sex,
            x='Age_Group_Label',
            y='total_chronic_conditions',
            ax=ax1
        )

        ax1.set_ylabel('Avg Chronic Conditions')
        ax1.set_xlabel('Age Group')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # Secondary axis
        ax2 = ax1.twinx()

        for sex, color in zip(['Male','Female'], ['orange','red']):
            temp = age_exp[age_exp['Sex_Label'] == sex]

            ax2.plot(
                temp['Age_Group_Label'],
                temp['TOTEXPY1_adj'],
                marker='o',
                linewidth=1.8,
                label=sex,
                color=color
            )

        ax2.set_ylabel('Average Total Healthcare Expenditure ($)')
        ax2.legend(fontsize=8)

        plt.xticks(rotation=20)
        plt.tight_layout()

        st.pyplot(fig1)

    # ==============================
    # RIGHT PLOT: HEALTH FACTORS
    # ==============================
    with col2:
        st.markdown(
            "<h4 style='text-align: center;'>Healthcare Spending by Health Conditions & Behaviors</h4>",
            unsafe_allow_html=True
        )
        df_plot = df_filtered.copy()

        vision_map = {1:0, 2:1, 3:1, 4:1, 5:1}

        df_plot['VISION_BIN'] = df_plot['VISION2'].map(vision_map)
        df_plot['HEARING_BIN'] = df_plot['HEARNG2'].map(vision_map)

        cols = ['ADSMOK2', 'PHYACT3', 'JTPAIN1', 'VISION_BIN', 'HEARING_BIN']

        df_melt = df_plot.melt(
            id_vars='TOTEXPY1_adj',
            value_vars=cols,
            var_name='Health_Factor',
            value_name='Category'
        )

        factor_labels = {
            'ADSMOK2': 'Smoking',
            'PHYACT3': 'Activity',
            'JTPAIN1': 'Joint Pain',
            'VISION_BIN': 'Vision',
            'HEARING_BIN': 'Hearing'
        }

        df_melt['Health_Factor'] = df_melt['Health_Factor'].map(factor_labels)

        df_melt['Category'] = df_melt['Category'].map({0:'No', 1:'Yes'})

        # Plot
        fig2, ax = plt.subplots(figsize=(5.5,4.5))

        sns.barplot(
            data=df_melt,
            x='Health_Factor',
            y='TOTEXPY1_adj',
            hue='Category',
            estimator=np.mean,
            ax=ax
        )

        # Trend line
        line_data = (
            df_melt.groupby('Health_Factor')['TOTEXPY1_adj']
            .mean()
            .reset_index()
        )

        ax.plot(
            line_data['Health_Factor'],
            line_data['TOTEXPY1_adj'],
            linestyle='--',
            marker='o',
            color='black',
            linewidth=1.8
        )

        ax.set_ylabel('Avg Healthcare Expenditure ($)')
        ax.set_xlabel('Health Condition / Behavior')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.legend(fontsize=8)

        plt.xticks(rotation=25)
        plt.tight_layout()

        st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SIDE-BY-SIDE: UTILIZATION ANALYSIS
# ==============================

with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # =========================================================
    # LEFT: UTILIZATION DISTRIBUTION (3D PIE)
    # =========================================================
    with col1:

        st.markdown(
            "<h5 style='text-align:center;'>Utilization Distribution (%)</h5>",
            unsafe_allow_html=True
        )

        df_plot = df_filtered.copy()

        util_cols = [
            'OPTOTVY1', 'IPDISY1', 'RXTOTY1',
            'ERTOTY1', 'OBTOTVY1'
        ]

        util_melted = df_plot.melt(
            value_vars=util_cols,
            var_name='Utilization_Type',
            value_name='Count'
        )

        util_summary = (
            util_melted
            .groupby('Utilization_Type')['Count']
            .sum()
            .reset_index()
        )

        util_summary['Percentage'] = (
            util_summary['Count'] / util_summary['Count'].sum()
        ) * 100

        label_map = {
            'OPTOTVY1': 'Outpatient',
            'IPDISY1': 'Inpatient',
            'RXTOTY1': 'Prescriptions',
            'ERTOTY1': 'ER',
            'OBTOTVY1': 'Office Visits'
        }

        util_summary['Utilization_Type'] = util_summary['Utilization_Type'].map(label_map)

        fig1, ax = plt.subplots(figsize=(5.5,4.5))

        # shadow
        ax.pie(
            util_summary['Percentage'],
            radius=1,
            startangle=140,
            colors=['#bbbbbb'] * len(util_summary),
            wedgeprops=dict(edgecolor='none'),
            center=(0, -0.06)
        )

        wedges, _ = ax.pie(
            util_summary['Percentage'],
            labels=util_summary['Utilization_Type'],
            startangle=140,
            wedgeprops=dict(edgecolor='black', linewidth=1)
        )

        # % labels
        for i, w in enumerate(wedges):
            pct = util_summary['Percentage'].iloc[i]
            if pct < 1:
                continue

            angle = (w.theta2 + w.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(angle))
            y = 0.6 * np.sin(np.deg2rad(angle))

            ax.text(
                x, y,
                f"{pct:.1f}%",
                ha='center', va='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

        ax.set_title('')
        plt.axis('equal')
        st.pyplot(fig1)

    # =========================================================
    # RIGHT: UTILIZATION vs EXPENDITURE BY INSURANCE
    # =========================================================
    with col2:

        st.markdown(
            "<h5 style='text-align:center;'>Utilization vs Expenditure by Insurance</h5>",
            unsafe_allow_html=True
        )

        df_plot = df_filtered.copy()

        ins_map = {
            1: 'Private',
            2: 'Public',
            3: 'Uninsured'
        }

        df_plot['Insurance_Type'] = df_plot['INSCOVY1'].map(ins_map)

        util_cols = ['OPTOTVY1','IPDISY1','RXTOTY1','ERTOTY1','OBTOTVY1']
        util_exp_col = ['OPTEXPY1','IPTEXPY1','RXEXPY1','ERTEXPY1','OBTEXPY1']

        util_labels = {
            'OPTOTVY1': 'Outpatient',
            'IPDISY1': 'Inpatient',
            'RXTOTY1': 'Prescriptions',
            'ERTOTY1': 'ER',
            'OBTOTVY1': 'Office Visits'
        }

        plot_data = []

        for util, exp in zip(util_cols, util_exp_col):

            temp = (
                df_plot
                .groupby('Insurance_Type')
                .agg(avg_util=(util, 'mean'), avg_exp=(exp, 'mean'))
                .reset_index()
            )

            temp['Utilization_Type'] = util_labels[util]
            plot_data.append(temp)

        plot_df = pd.concat(plot_data, ignore_index=True)

        order = [util_labels[u] for u in util_cols]

        plot_df['Utilization_Type'] = pd.Categorical(
            plot_df['Utilization_Type'],
            categories=order,
            ordered=True
        )

        plot_df = plot_df.sort_values('Utilization_Type')

        util_pivot = plot_df.pivot(
            index='Utilization_Type',
            columns='Insurance_Type',
            values='avg_util'
        ).loc[order]

        fig2, ax1 = plt.subplots(figsize=(5.5,4.5))

        # stacked bars
        bottom = np.zeros(len(util_pivot))

        for col in util_pivot.columns:
            ax1.bar(
                util_pivot.index,
                util_pivot[col],
                bottom=bottom,
                label=f'{col}'
            )
            bottom += util_pivot[col].values

        ax2 = ax1.twinx()

        for ins in plot_df['Insurance_Type'].unique():
            subset = plot_df[plot_df['Insurance_Type'] == ins].sort_values('Utilization_Type')

            ax2.plot(
                subset['Utilization_Type'],
                subset['avg_exp'],
                marker='o',
                linewidth=1.8,
                label=f'{ins} Exp'
            )

        ax1.set_ylabel('Utilization')
        ax2.set_ylabel('Expenditure ($)')

        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.xticks(rotation=25)

        # legends
        legend1 = ax1.legend(loc='upper left', fontsize=7)
        legend2 = ax2.legend(loc='upper right', fontsize=7)
        ax1.add_artist(legend1)

        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PLOT: BMI × ACTIVITY vs EXPENDITURE (2x2 GRID)
# ==============================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)

    st.markdown(
        "<h4 style='text-align:center;'>Healthcare Spending by BMI & Physical Activity</h4>",
        unsafe_allow_html=True
    )

    # ---------------------------
    # COPY DATA
    # ---------------------------
    df_plot = df_filtered.copy()

    # ---------------------------
    # Map Activity
    # ---------------------------
    activity_map = {1: 'Yes', 0: 'No'}
    df_plot['Activity_Level'] = df_plot['PHYACT3'].map(activity_map)

    # ---------------------------
    # Map BMI Labels
    # ---------------------------
    bmi_labels = {
        1: 'Underweight',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obese'
    }

    df_plot['BMI_Category_Label'] = df_plot['bmi_category'].map(bmi_labels)

    # ---------------------------
    # Order BMI
    # ---------------------------
    bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']

    df_plot['BMI_Category_Label'] = pd.Categorical(
        df_plot['BMI_Category_Label'],
        categories=bmi_order,
        ordered=True
    )

    # ---------------------------
    # CREATE SUBPLOTS (COMPACT SIZE)
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9,7), sharey=True)
    axes = axes.flatten()

    for i, bmi in enumerate(bmi_order):

        ax = axes[i]

        temp = df_plot[df_plot['BMI_Category_Label'] == bmi]

        sns.scatterplot(
            data=temp,
            x="BMINDX5",
            y="TOTEXPY1_adj",
            hue="Activity_Level",
            alpha=0.6,
            ax=ax
        )

        ax.set_title(bmi, fontsize=10)

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        # Smaller legend per subplot
        ax.legend(title="Activity", loc='upper right', fontsize=7, title_fontsize=8)

    # ---------------------------
    # AXIS LABELS
    # ---------------------------
    for ax in axes:
        ax.set_xlabel("BMI Index", fontsize=8)
        ax.set_ylabel("Exp ($)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ---------------------------
    # SHOW
    # ---------------------------
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SIDE-BY-SIDE: INCOME vs PAYER MIX
# ==============================

with st.container():
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # =========================================================
    # LEFT: INCOME × INSURANCE
    # =========================================================
    with col1:

        st.markdown(
            "<h5 style='text-align:center;'>Spending by Income & Insurance</h5>",
            unsafe_allow_html=True
        )

        df_plot = df_filtered.copy()

        # Drop missing
        df_plot = df_plot.dropna(subset=['TOTEXPY1', 'POVCATY1', 'INSCOVY1'])

        # Maps
        pov_map = {
            1: 'Poor',
            2: 'Near Poor',
            3: 'Low Income',
            4: 'Middle Income',
            5: 'High Income'
        }

        ins_map = {
            1: 'Private',
            2: 'Public',
            3: 'Uninsured'
        }

        df_plot['Income_Group'] = df_plot['POVCATY1'].map(pov_map)
        df_plot['Insurance_Type'] = df_plot['INSCOVY1'].map(ins_map)

        # Aggregation
        median_data = (
            df_plot
            .groupby(['Income_Group','Insurance_Type'])['TOTEXPY1']
            .mean()
            .reset_index()
        )

        # Plot
        fig1, ax1 = plt.subplots(figsize=(5,4.5))

        sns.barplot(
            data=median_data,
            x='Income_Group',
            y='TOTEXPY1',
            hue='Insurance_Type',
            ax=ax1
        )

        ax1.set_ylabel('Mean Exp ($)')
        ax1.set_xlabel('')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        plt.xticks(rotation=25)
        ax1.legend(fontsize=7, title='Insurance')

        plt.tight_layout()
        st.pyplot(fig1)

    # =========================================================
    # RIGHT: PAYER MIX BY AGE
    # =========================================================
    with col2:

        st.markdown(
            "<h5 style='text-align:center;'>Payer Mix by Age</h5>",
            unsafe_allow_html=True
        )

        df_plot = df_filtered.copy()

        # Age mapping
        age_map = {
            1: '18–29',
            2: '30–44',
            3: '45–59',
            4: '60–74',
            5: '75+'
        }

        age_order = ['18–29','30–44','45–59','60–74','75+']

        df_plot['Age_Group_Label'] = df_plot['age_group'].map(age_map)

        df_plot['Age_Group_Label'] = pd.Categorical(
            df_plot['Age_Group_Label'],
            categories=age_order,
            ordered=True
        )

        # Payer columns
        payer_cols = [
            'TOTSLFY1_adj',
            'TOTMCDY1_adj',
            'TOTMCRY1_adj',
            'TOTPRVY1_adj',
            'TOTTRIY1_adj',
            'TOTOPUY1_adj'
        ]

        age_payer = (
            df_plot
            .groupby('Age_Group_Label', observed=False)[payer_cols]
            .sum()
        )

        age_payer["Total"] = age_payer.sum(axis=1)

        age_payer_prop = age_payer[payer_cols].div(age_payer["Total"], axis=0)

        age_payer_prop.columns = [
            'Self-pay',
            'Medicaid',
            'Medicare',
            'Private',
            'Tricare',
            'Other Public'
        ]

        # Plot
        fig2, ax2 = plt.subplots(figsize=(5.5,4.5))

        x = np.arange(len(age_payer_prop.index))
        bottom = np.zeros(len(age_payer_prop.index))

        for col in age_payer_prop.columns:

            values = age_payer_prop[col].to_numpy()

            ax2.bar(
                x,
                values,
                bottom=bottom,
                label=col
            )

            # % labels
            for i, v in enumerate(values):
                if v > 0.10:
                    ax2.text(
                        x[i],
                        bottom[i] + v/2,
                        f"{v*100:.0f}%",
                        ha='center',
                        va='center',
                        color='white',
                        fontsize=8,
                        fontweight='bold'
                    )

            bottom += values

        ax2.set_xticks(x)
        ax2.set_xticklabels(age_payer_prop.index)

        ax2.set_ylabel('Proportion')
        ax2.set_xlabel('')
        ax2.set_ylim(0,1)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        ax2.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=7)

        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Developed by **Kumar Saransh** | Northeastern University")
