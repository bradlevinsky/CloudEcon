import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Brad Levinsky EC2 Instance EDA Dashboard")

df = pd.read_csv("ec2dataset.csv")

st.write("Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Instances", len(df))
col2.metric("Number of Columns", len(df.columns))
col3.metric("Missing Values", df.isna().sum().sum())

st.subheader("Dataset Structure")

st.write("Columns:")
st.write(df.columns.tolist())

st.subheader("Data Types")

st.write(df.dtypes.astype(str))

df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)

df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)

def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)

    if "unavailable" in value.lower():
        return None

    return float(
        value.replace("$", "")
        .replace(" hourly", "")
        .strip()
    )

price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)

df["Monthly_On_Demand"] = df["On Demand_USD"] * 730

df["Cost_Per_GiB"] = (
    df["On Demand_USD"] /
    df["Memory_GiB"]
)

df["Memory_per_vCPU"] = (
    df["Memory_GiB"] /
    df["vCPU_Count"]
)

st.subheader("Monthly On-Demand Cost")

st.dataframe(
    df[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
)

st.sidebar.header("Filters")

max_memory = float(df["Memory_GiB"].max())

memory_filter = st.sidebar.slider(
    "Maximum Memory (GiB)",
    min_value=0.5,
    max_value=max_memory,
    value=max_memory
)

cpu_values = sorted(
    df["vCPU_Count"].dropna().unique()
)

selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values
)

network_values = df["Network Performance"].dropna().unique()

selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values
)

storage_types = [
    "EBS",
    "SSD",
    "NVMe",
    "HDD"
]

selected_storage = st.sidebar.multiselect(
    "Storage Type",
    options=storage_types,
    default=storage_types
)

storage_filter = df["Instance Storage"].apply(
    lambda storage: any(
        storage_type.lower() in str(storage).lower()
        for storage_type in selected_storage
    )
)

max_price = float(df["On Demand_USD"].max())

price_filter = st.sidebar.slider(
    "Maximum Hourly Price",
    min_value=0.0,
    max_value=max_price,
    value=max_price
)

monthly_cost_filter = st.sidebar.selectbox(
    "Maximum Monthly Cost",
    options=[10, 25, 50, 100, 500]
)

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter) &
    (df["vCPU_Count"].isin(selected_cpu)) &
    (df["Network Performance"].isin(selected_network)) &
    storage_filter &
    (df["On Demand_USD"] <= price_filter) &
    (df["Monthly_On_Demand"] <= monthly_cost_filter)
]

st.subheader("Filtered EC2 Instances")

st.write(
    f"{len(filtered_df)} instances found"
)

st.dataframe(filtered_df)

st.subheader("EC2 Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Instances",
    len(filtered_df)
)

col2.metric(
    "Avg Memory",
    f"{filtered_df['Memory_GiB'].mean():.2f} GiB"
)

col3.metric(
    "Avg vCPUs",
    f"{filtered_df['vCPU_Count'].mean():.1f}"
)

col4.metric(
    "Avg Hourly Cost",
    f"${filtered_df['On Demand_USD'].mean():.4f}"
)

st.subheader("Memory Distribution")

fig = px.histogram(
    filtered_df,
    x="Memory_GiB",
    nbins=30,
    title="Distribution of EC2 Memory"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("vCPU Distribution")

fig = px.histogram(
    filtered_df,
    x="vCPU_Count",
    title="Distribution of vCPUs"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Memory vs vCPUs")

fig = px.scatter(
    filtered_df,
    x="vCPU_Count",
    y="Memory_GiB",
    hover_name="API Name",
    hover_data=["On Demand_USD"],
    title="EC2 Memory vs vCPU"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Memory vs On-Demand Cost")

fig = px.scatter(
    filtered_df,
    x="Memory_GiB",
    y="On Demand_USD",
    hover_name="API Name",
    size="vCPU_Count",
    title="Memory vs EC2 On-Demand Cost"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Lowest-Cost EC2 Instances")

cheapest = (
    filtered_df
    .sort_values("On Demand_USD")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(cheapest)

st.subheader("Highest-Cost EC2 Instances")

most_expensive = (
    filtered_df
    .sort_values(
        "On Demand_USD",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(most_expensive)

pricing = filtered_df[
    [
        "Name",
        "API Name",
        "On Demand_USD",
        "Linux Reserved cost_USD",
        "Linux Spot Minimum cost_USD",
        "Windows On Demand cost_USD",
        "Windows Reserved cost_USD"
    ]
]

st.subheader("EC2 Pricing Comparison")

st.dataframe(pricing)

if not filtered_df.empty:
    selected_instance = st.selectbox(
        "Select an EC2 Instance",
        filtered_df["API Name"].dropna().unique()
    )

    instance = filtered_df[
        filtered_df["API Name"] == selected_instance
    ].iloc[0]

    pricing_data = pd.DataFrame({
        "Pricing Model": [
            "On Demand",
            "Linux Reserved",
            "Linux Spot"
        ],
        "Hourly Cost": [
            instance["On Demand_USD"],
            instance["Linux Reserved cost_USD"],
            instance["Linux Spot Minimum cost_USD"]
        ]
    })

    pricing_data = pricing_data.dropna()

    fig = px.bar(
        pricing_data,
        x="Pricing Model",
        y="Hourly Cost",
        title=f"Pricing Comparison: {selected_instance}"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    pricing_data["Monthly Cost"] = (
        pricing_data["Hourly Cost"] * 730
    )

    st.dataframe(pricing_data)

st.subheader("Cost per GiB of Memory")

efficiency = (
    filtered_df
    .sort_values("Cost_Per_GiB")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Cost_Per_GiB"
        ]
    ]
    .head(15)
)

st.dataframe(efficiency)

st.subheader("Highest Memory per vCPU")

memory_per_cpu = (
    filtered_df
    .sort_values(
        "Memory_per_vCPU",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "Memory_per_vCPU"
        ]
    ]
    .head(15)
)

st.dataframe(memory_per_cpu)

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv"
)