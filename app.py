# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
import pytz

df = pd.read_csv("enhanced_internship_jobs_dataset.csv")

st.set_page_config(layout="wide")
st.title("🌐 Job Analytics Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Jobs")

selected_country = st.sidebar.multiselect("Select Country", df["country"].unique(), default=df["country"].unique())
selected_gender = st.sidebar.multiselect("Select Gender", df["gender_preference"].unique(), default=df["gender_preference"].unique())
selected_qualification = st.sidebar.multiselect("Select Qualification", df["qualification"].unique(), default=df["qualification"].unique())
selected_jobtype = st.sidebar.multiselect("Select Employment Type", df["employment_type"].unique(), default=df["employment_type"].unique())
selected_title = st.sidebar.multiselect("Select Job Title", df["job_title"].unique(), default=df["job_title"].unique())
selected_portal = st.sidebar.multiselect("Select Job Portal", df["job_portal"].unique(), default=df["job_portal"].unique())
salary_range = st.sidebar.slider("Minimum Salary (USD)", 0, int(df["salary_usd"].max()), 0)
min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, int(df["experience_years"].max()), 0)
selected_date = st.sidebar.date_input("Posting Date before", pd.to_datetime("2025-01-01"))
# --- Preprocess Date ---
df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
filtered_df = df[
    (df["country"].isin(selected_country)) &
    (df["gender_preference"].isin(selected_gender)) &
    (df["qualification"].isin(selected_qualification)) &
    (df["employment_type"].isin(selected_jobtype)) &
    (df["job_title"].isin(selected_title)) &
    (df["job_portal"].isin(selected_portal)) &
    (df["salary_usd"] >= salary_range) &
    (df["experience_years"] >= min_exp)
]

# --- Main Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Gender Distribution")

    gender_counts = filtered_df["gender_preference"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]

    # Full color scheme
    base_colors = {
        "Female": "#FF69B4",  # Hot Pink
        "Male": "#1E90FF",    # Dodger Blue
        "Other": "#A9A9A9"    # Dark Gray
    }

    # Dimmed versions of the colors
    dimmed_colors = {
        "Female": "#FFC0CB",  # Light Pink
        "Male": "#87CEFA",    # Light Blue
        "Other": "#D3D3D3"    # Light Gray
    }

    # Use selected_gender to decide which color to use
    selected_set = set(selected_gender)
    if len(selected_set) == 1:
        active_gender = list(selected_set)[0]
        gender_colors = {
            gender: base_colors[gender] if gender == active_gender else dimmed_colors[gender]
            for gender in gender_counts["Gender"]
        }
        pulls = [0.1 if gender == active_gender else 0 for gender in gender_counts["Gender"]]
    else:
        gender_colors = base_colors
        pulls = [0 for _ in gender_counts["Gender"]]

    # Plot
    pie = px.pie(
        gender_counts,
        names="Gender",
        values="Count",
        color="Gender",
        color_discrete_map=gender_colors,
        hole=0.4
    )
    pie.update_traces(pull=pulls)

    st.plotly_chart(pie, use_container_width=True)

with col2:
    st.subheader("💰 Average Salary by Country")

    # Calculate average salary
    avg_salary = filtered_df.groupby("country")["salary_usd"].mean().reset_index()

    # Set custom color map
    country_colors = {
        "India": "orange",
        "Germany": "green"
    }

    # Create bar chart with specific country colors
    salary_fig = px.bar(
        avg_salary,
        x="country",
        y="salary_usd",
        color="country",
        color_discrete_map=country_colors
    )

    st.plotly_chart(salary_fig, use_container_width=True)


st.subheader("📈 Salary Distribution (India vs Germany)")
hist_df = filtered_df[filtered_df["country"].isin(["India", "Germany"])]
hist_fig = px.histogram(hist_df, x="salary_usd", color="country",
                        barmode="overlay", nbins=30,
                        color_discrete_map={"India": "orange", "Germany": "green"})
st.plotly_chart(hist_fig, use_container_width=True)

# Job Title vs Posting Frequency - Calendar Heatmap Alternative
st.subheader("Job Titles Frequency by Posting Date")
heat_data = filtered_df.groupby([filtered_df['posting_date'].dt.date, "job_title"]).size().reset_index(name="count")
fig_heat = px.density_heatmap(
    heat_data,
    x="posting_date",
    y="job_title",
    z="count",
    nbinsx=20,
    color_continuous_scale="Blues",
    title="Job Title Frequency Over Time"
)
st.plotly_chart(fig_heat, use_container_width=True)

# --- Special Task Chart ---
now = datetime.now(pytz.timezone('Asia/Kolkata')).time()
start_time = time(15, 0)
end_time = time(17, 0)

special_df = df[
    (df["country"].isin(["India", "Germany"])) &
    (df["qualification"] == "B.tech") &
    (df["employment_type"] == "Full time") &
    (df["experience_years"] > 2) &
    (df["job_title"].isin(["Data Scientist", "Art Teacher", "AeroSpace Engineer"])) &
    (df["salary_usd"] > 10000) &
    (df["gender_preference"] == "Female") &
    (df["job_portal"] == "Indeed") &
    (pd.to_datetime(df["posting_date"], errors="coerce") < pd.to_datetime("2023-08-01", errors="coerce"))
]

if start_time <= now <= end_time and not special_df.empty:
    st.subheader("✅ Special Chart (India vs Germany Jobs Meeting Criteria)")
    task_fig = px.bar(special_df, x="job_title", y="salary_usd", color="country",
                      color_discrete_map={"India": "orange", "Germany": "green"},
                      title="Special Task Salary Chart")
    st.plotly_chart(task_fig, use_container_width=True)

st.caption("© 2025 Job Insights Dashboard")

#old

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from datetime import datetime
# import pytz
# import calplot
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# st.markdown("""
#     <style>
#         .metric-container {
#             color: #111111 !important;
#             font-size: 18px;
#         }
#         .stMetric label, .stMetric div {
#             font-weight: 600;
#             color: #1a1a1a;
#         }
#         h3 {
#             color: #202020 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)


# st.markdown("""
#     <style>
#         body, .stApp {
#             background-color: #FFFFFF;
#             color: #111111;
#             font-family: 'Segoe UI', sans-serif;
#         }
#         h1, h2, h3, h4 {
#             color: #1f1f1f;
#         }
#         .big-font {
#             font-size:28px !important;
#             font-weight: 600;
#             color: #202020;
#         }
#         .subtitle {
#             font-size:18px !important;
#             color: #444444;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("""
#     <style>
#         .stApp {
#             background-color: #f9f9f9; /* Light gray is easier on the eyes */
#         }
#     </style>
# """, unsafe_allow_html=True)



# # Load and preprocess data
# df = pd.read_csv("enhanced_internship_jobs_dataset.csv")

# # Convert posting_date to datetime
# df['posting_date'] = pd.to_datetime(df['posting_date'].astype(str), dayfirst=True, errors='coerce')
# df = df[df['posting_date'].notna()]

# # Normalize text columns to lower for reliable filtering
# df['gender_preference'] = df['gender_preference'].astype(str).str.lower()
# df['employment_type'] = df['employment_type'].astype(str).str.lower()
# df['qualification'] = df['qualification'].astype(str).str.upper()

# # Show raw data if checkbox enabled
# st.markdown(" ### Raw Data (check for first 5 rows) ")
# if st.checkbox("Show raw data"):
#     st.write(df.head())

# # Filters
# st.sidebar.header("📌 Filters")

# countries = st.sidebar.multiselect("Country", df["country"].unique(), default=["India", "Germany"])
# qual = st.sidebar.selectbox("Qualification", df["qualification"].unique(), index=0)
# job_roles = st.sidebar.multiselect("Job Titles", df["job_title"].unique(), default=["Data Scientist", "Art Teacher", "Aerospace Engineer"])
# employment_type = st.sidebar.selectbox("Employment Type", df["employment_type"].unique(), index=0)
# gender = st.sidebar.selectbox("Gender Preference", ["male", "female"], index=1)
# min_salary = st.sidebar.slider("Minimum Salary (USD)", 0, 100000, 10000, 1000)
# min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, 10, 2, 1)
# cutoff_date = st.sidebar.date_input("Posted Before", datetime(2023, 8, 1))

# # Filter dataset
# filtered_df = df[
#     (df["country"].isin(countries)) &
#     (df["qualification"] == qual) &
#     (df["employment_type"] == employment_type.lower()) &
#     (df["experience_years"] > min_exp) &
#     (df["job_title"].isin(job_roles)) &
#     (df["salary_usd"] > min_salary) &
#     (df["gender_preference"] == gender.lower()) &
#     (df["posting_date"] < pd.to_datetime(cutoff_date))
# ]
# date_counts = filtered_df['posting_date'].value_counts().sort_index()
# # Show filtered count
# st.markdown(f"### 🎯 Filtered Results: {filtered_df.shape[0]} job(s)")
# st.markdown("### 📈 Key Insights")
# col1, col2, col3 = st.columns(3)

# col1.metric("Jobs Found", filtered_df.shape[0])
# col2.metric("Avg Salary", f"${filtered_df['salary_usd'].mean():,.0f}" if not filtered_df.empty else "N/A")
# col3.metric("Avg Exp", f"{filtered_df['experience_years'].mean():.1f} years" if not filtered_df.empty else "N/A")

# # Display filtered data
# st.dataframe(filtered_df)

# # Plot salary by country
# if not filtered_df.empty:
#     st.subheader("📊 Salary Distribution by Country")

#     # Split data
#     india_data = filtered_df[filtered_df["country"] == "India"]
#     germany_data = filtered_df[filtered_df["country"] == "Germany"]

#     # Create histogram
#     hist_fig = go.Figure()

#     hist_fig.add_trace(go.Histogram(
#         x=india_data["salary_usd"],
#         name="India",
#         marker_color='orange',
#         opacity=0.7
#     ))

#     hist_fig.add_trace(go.Histogram(
#         x=germany_data["salary_usd"],
#         name="Germany",
#         marker_color='green',
#         opacity=0.7
#     ))

#     # Overlay both histograms
#     hist_fig.update_layout(
#         barmode='overlay',
#         title="Salary Distribution by Country",
#         xaxis_title="Salary (USD)",
#         yaxis_title="Count",
#         legend=dict(x=0.8, y=0.95),
#         template="plotly_white"
#     )

#     st.plotly_chart(hist_fig)


#     total = len(filtered_df)
#     female_count = (filtered_df['gender_preference'].str.lower() == 'female').sum()

#     fig_gauge = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=(female_count / total) * 100 if total > 0 else 0,
#         title={'text': "🎯 % Female-Oriented Jobs"},
#         gauge={
#             'axis': {'range': [0, 100]},
#             'bar': {'color': "green"},
#             'steps': [
#                 {'range': [0, 50], 'color': "lightgray"},
#                 {'range': [50, 100], 'color': "lightgreen"},
#             ],
#         }
#     ))
    
    
#     fig_bubble = px.scatter(
#     filtered_df,
#     x="experience_years",
#     y="salary_usd",
#     color="country",
#     size="salary_usd",
#     hover_name="job_title",
#     title="💼 Salary vs Experience by Job Title")
#     st.plotly_chart(fig_bubble)





#     st.subheader("📅 Job Postings Calendar Heatmap")

#     fig, ax = calplot.calplot(
#         date_counts,
#         cmap='YlOrRd',  # Yellow to Red color scheme
#         colorbar=True,
#         suptitle='Job Postings by Date',
#         edgecolor='white',
#         linewidth=1,
#         how='sum',
#         figsize=(16, 4)
#     )

#     st.pyplot(fig)

#     st.subheader("👥 Gender Preference Counts")
#     gender_counts = df['gender_preference'].value_counts(dropna=True)
#     fig = px.pie(
#     names=gender_counts.index,
#     values=gender_counts.values,
#     title="🧑‍🤝‍🧑 Gender Preference Distribution (All Jobs)",
#     color_discrete_sequence=px.colors.sequential.RdBu)
#     st.plotly_chart(fig, use_container_width=True)


# else:
#     st.warning("⚠️ No jobs found with the selected filters.")
























# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from datetime import datetime
# import pytz
# import matplotlib.pyplot as plt

# # Set wide layout for mobile/tablet support
# st.set_page_config(page_title="Internship Job Insights", layout="wide")

# st.title("📊 Internship Job Analysis Dashboard")

# # Load data
# df = pd.read_csv("enhanced_internship_jobs_dataset.csv")

# # Display raw data (optional)
# with st.expander("🔍 View Raw Data"):
#     st.dataframe(df)

# # Normalize relevant string columns
# df['employment_type'] = df['employment_type'].astype(str).str.strip().str.lower()
# df['qualification'] = df['qualification'].astype(str).str.strip()
# df['country'] = df['country'].astype(str).str.strip()
# df['gender_preference'] = df['gender_preference'].astype(str).str.strip().str.lower()
# df['job_title'] = df['job_title'].astype(str).str.strip()

# # Convert posting_date safely
# df['posting_date'] = pd.to_datetime(df['posting_date'].astype(str), dayfirst=True, errors='coerce')
# df = df[df['posting_date'].notna()]

# # Stepwise filter breakdown
# st.write("Stepwise Filter Breakdown:")
# st.write("Total:", len(df))
# st.write("Country Filter:", len(df[df['country'].isin(["India", "Germany"])]))
# st.write("Qualification Filter:", len(df[df['qualification'] == "B.Tech"]))
# st.write("Employment Filter:", len(df[df['employment_type'] == "full time"]))
# st.write("Experience Filter:", len(df[df['experience_years'] > 2]))
# st.write("Job Title Filter:", len(df[df['job_title'].isin(["Data Scientist", "Art Teacher", "Aerospace Engineer"])]))
# st.write("Salary Filter:", len(df[df['salary_usd'] > 10000]))
# st.write("Gender Filter:", len(df[df['gender_preference'] == "female"]))
# st.write("Posting Date Filter:", len(df[df['posting_date'] < pd.to_datetime("2023-08-01")]))

# # Apply combined filter
# filtered = df[
#     (df['country'].isin(["India", "Germany"])) &
#     (df['qualification'] == "B.Tech") &
#     (df['employment_type'] == "full time") &
#     (df['experience_years'] > 2) &
#     (df['job_title'].isin(["Data Scientist", "Art Teacher", "Aerospace Engineer"])) &
#     (df['salary_usd'] > 10000) &
#     (df['gender_preference'] == "female") &
#     (df['posting_date'] < pd.to_datetime("2023-08-01"))
# ]

# # Gender dropdown filter
# st.write("Available values for Gender:", df['gender_preference'].unique())
# gender = st.selectbox("Select Gender", df['gender_preference'].unique())
# filtered_df = df[df['gender_preference'] == gender]

# # Debug info
# st.write("📅 Sample Posting Dates:", df['posting_date'].dropna().sort_values().head(10))
# st.write("📅 Max Posting Date:", df['posting_date'].max())
# st.write("📦 posting_date dtype:", df['posting_date'].dtype)
# st.write("📅 Filtered Dates Preview:")
# st.dataframe(df[df['posting_date'] < pd.to_datetime("2023-08-01")][['job_title', 'posting_date']].head())

# st.success(f"✅ Total jobs after filter: {len(filtered)}")



# # Load data
# df = pd.read_csv("enhanced_internship_jobs_dataset.csv")

# # Preprocessing
# df['posting_date'] = pd.to_datetime(df['posting_date'], dayfirst=True)

# # Filtering
# filtered_df = df[
#     (df['country'].isin(["India", "Germany"])) &
#     (df['qualification'] == "B.Tech") &
#     (df['employment_type'] == "Full time") &
#     (df['experience_years'] > 2) &
#     (df['job_title'].isin(["Data Scientist", "Art Teacher", "Aerospace Engineer"])) &
#     (df['salary_usd'] > 10000) &
#     (df['gender_preference'].str.lower() == "female") &
#     (df['posting_date'] < pd.to_datetime("2023-08-01"))
# ]

# # View Raw Data (Optional)
# with st.expander("🔍 View Raw Data"):
#     st.dataframe(filtered_df)

# # Title
# st.title("🎯 Internship Job Insights Dashboard")

# # Layout: Two Columns
# col1, col2 = st.columns(2)

# # ========== LEFT COLUMN ==========
# with col1:
#     st.subheader("📌 Job Count by Employment Type")
#     emp_counts = filtered_df['employment_type'].value_counts()
#     st.bar_chart(emp_counts)

#     st.subheader("💰 Average Salary by Country")
#     avg_salary = filtered_df.groupby('country')['salary_usd'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_salary)

#     st.subheader("📉 Experience vs Salary")
#     st.scatter_chart(filtered_df[['experience_years', 'salary_usd']])

# # ========== RIGHT COLUMN ==========
# with col2:
#     st.subheader("📆 Job Postings Over Time")
#     jobs_over_time = filtered_df.groupby(filtered_df['posting_date'].dt.to_period("M")).size()
#     jobs_over_time.index = jobs_over_time.index.astype(str)
#     st.line_chart(jobs_over_time)

#     st.subheader("⚖️ Gender Preference (Pie Chart)")
#     gender_counts = filtered_df['gender_preference'].value_counts()
#     fig, ax = plt.subplots()
#     ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
#     ax.axis('equal')
#     st.pyplot(fig)

# # Footer
# st.markdown("---")
# st.markdown("✅ Developed by Ais for Internship Task | Responsive Design Ready")

# Chart between 3–5 PM IST
# ist = pytz.timezone("Asia/Kolkata")
# now = datetime.now(ist)
# hour = now.hour

# if 15 <= hour < 17:
#     st.subheader("📈 Job Role Count by Country")
#     chart_df = filtered.groupby(['country', 'role']).size().reset_index(name='count')
#     fig = px.bar(chart_df, x="role", y="count", color="country", barmode="group")
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.warning("⏳ The job chart is only visible from 3 PM to 5 PM IST.")
