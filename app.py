import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Brad Levinsky EC2 Instance EDA Dashboard",
    layout="wide"
)

df = pd.read_csv("ec2dataset.csv")

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
        .replace(",", "")
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

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "EC2 Dashboard",
        "Cost Analysis",
        "Regression Analysis"
    ]
)

if page == "EC2 Dashboard":

    st.title("Brad Levinsky EC2 Instance EDA Dashboard")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Number of Instances",
        len(df)
    )

    col2.metric(
        "Number of Columns",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        df.isna().sum().sum()
    )

    st.subheader("Dataset Preview")

    st.dataframe(df)

    st.subheader("Dataset Structure")

    st.write("Columns:")
    st.write(df.columns.tolist())

    st.subheader("Data Types")

    st.write(df.dtypes.astype(str))

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

    max_memory = float(
        df["Memory_GiB"].max()
    )

    memory_filter = st.sidebar.slider(
        "Maximum Memory (GiB)",
        min_value=0.5,
        max_value=max_memory,
        value=max_memory
    )

    cpu_values = sorted(
        df["vCPU_Count"]
        .dropna()
        .unique()
    )

    selected_cpu = st.sidebar.multiselect(
        "vCPU Count",
        options=cpu_values,
        default=cpu_values
    )

    network_values = (
        df["Network Performance"]
        .dropna()
        .unique()
    )

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
            storage_type.lower()
            in str(storage).lower()
            for storage_type
            in selected_storage
        )
    )

    max_price = float(
        df["On Demand_USD"].max()
    )

    price_filter = st.sidebar.slider(
        "Maximum Hourly Price",
        min_value=0.0,
        max_value=max_price,
        value=max_price
    )

    monthly_cost_filter = st.sidebar.selectbox(
        "Maximum Monthly Cost",
        options=[
            10,
            25,
            50,
            100,
            500
        ]
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
            filtered_df[
                "API Name"
            ].dropna().unique()
        )

        instance = filtered_df[
            filtered_df["API Name"]
            == selected_instance
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

    csv = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Filtered Dataset",
        data=csv,
        file_name="filtered_ec2_instances.csv",
        mime="text/csv"
    )

elif page == "Cost Analysis":

    st.title("EC2 Cost Analysis")

    st.subheader("Missing Cost Values")

    missing_values = pd.DataFrame({
        "Pricing Model": price_columns,
        "Missing Values": [
            df[column + "_USD"].isna().sum()
            for column in price_columns
        ]
    })

    st.dataframe(missing_values)

    st.subheader("Cost Summary Statistics")

    cost_summary = df[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].describe()

    cost_summary.columns = price_columns

    st.dataframe(cost_summary)

    st.subheader(
        "Cost Comparison of Amazon EC2 Instances"
    )

    boxplot_data = df[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].copy()

    boxplot_data.columns = price_columns

    boxplot_long = boxplot_data.melt(
        var_name="Pricing Model",
        value_name="Hourly Cost"
    ).dropna()

    fig = px.box(
        boxplot_long,
        x="Pricing Model",
        y="Hourly Cost",
        title="Cost Comparison of Amazon EC2 Instances (Hourly)"
    )

    fig.update_layout(
        xaxis_title="Pricing Model",
        yaxis_title="Cost (USD)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("On-Demand Cost Outliers")

    on_demand = df["On Demand_USD"]

    Q1 = on_demand.quantile(0.25)
    Q3 = on_demand.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_on_demand = df[
        (df["On Demand_USD"] < lower_bound) |
        (df["On Demand_USD"] > upper_bound)
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Q1",
        f"${Q1:.4f}"
    )

    col2.metric(
        "Q3",
        f"${Q3:.4f}"
    )

    col3.metric(
        "Outliers",
        len(outliers_on_demand)
    )

    st.write(
        f"Lower Bound: ${lower_bound:.4f}"
    )

    st.write(
        f"Upper Bound: ${upper_bound:.4f}"
    )

    st.dataframe(
        outliers_on_demand[
            [
                "Name",
                "API Name",
                "Instance Memory",
                "vCPUs",
                "On Demand_USD"
            ]
        ].sort_values(
            "On Demand_USD",
            ascending=False
        )
    )

    st.subheader(
        "On-Demand vs Linux Reserved Cost"
    )

    cost_comparison = df[
        [
            "Name",
            "API Name",
            "On Demand_USD",
            "Linux Reserved cost_USD"
        ]
    ].dropna().sort_values(
        "On Demand_USD"
    )

    st.dataframe(
        cost_comparison.head(10)
    )

    def filter_instance_family(family):
        return df[
            df["Name"].str.startswith(
                family,
                na=False
            )
        ]

    t2_instances = filter_instance_family(
        "T2"
    )

    t3_instances = filter_instance_family(
        "T3"
    )

    st.header("T2 Instance Analysis")

    st.subheader(
        "T2 Instance Costs Summary"
    )

    t2_summary = t2_instances[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].describe()

    t2_summary.columns = price_columns

    st.dataframe(t2_summary)

    st.subheader(
        "T2 Cost Distribution"
    )

    t2_box_data = t2_instances[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].copy()

    t2_box_data.columns = price_columns

    t2_box_long = t2_box_data.melt(
        var_name="Pricing Model",
        value_name="Hourly Cost"
    ).dropna()

    fig = px.box(
        t2_box_long,
        x="Pricing Model",
        y="Hourly Cost",
        points="outliers",
        title="Cost Distribution for T2 Instances"
    )

    fig.update_layout(
        xaxis_title="Pricing Model",
        yaxis_title="Cost (USD)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.header("T3 Instance Analysis")

    st.subheader(
        "T3 Instance Costs Summary"
    )

    t3_summary = t3_instances[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].describe()

    t3_summary.columns = price_columns

    st.dataframe(t3_summary)

    st.subheader(
        "T3 Cost Distribution"
    )

    t3_box_data = t3_instances[
        [
            column + "_USD"
            for column in price_columns
        ]
    ].copy()

    t3_box_data.columns = price_columns

    t3_box_long = t3_box_data.melt(
        var_name="Pricing Model",
        value_name="Hourly Cost"
    ).dropna()

    fig = px.box(
        t3_box_long,
        x="Pricing Model",
        y="Hourly Cost",
        points="outliers",
        title="Cost Distribution for T3 Instances"
    )

    fig.update_layout(
        xaxis_title="Pricing Model",
        yaxis_title="Cost (USD)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.header("T2 and T3 Cost Comparison")

    comparison = pd.concat([
        t2_instances[
            [
                "Name",
                "API Name",
                "On Demand_USD",
                "Linux Reserved cost_USD"
            ]
        ],
        t3_instances[
            [
                "Name",
                "API Name",
                "On Demand_USD",
                "Linux Reserved cost_USD"
            ]
        ]
    ])

    comparison_sorted = (
        comparison
        .dropna()
        .sort_values("On Demand_USD")
    )

    st.dataframe(
        comparison_sorted.head(10)
    )

elif page == "Regression Analysis":

    st.title(
        "EC2 On-Demand Cost Regression Analysis"
    )

    regression_data = df[
        [
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD"
        ]
    ].dropna()

    X = regression_data[
        [
            "Memory_GiB",
            "vCPU_Count"
        ]
    ]

    y = regression_data[
        "On Demand_USD"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression(
        positive=True,
        fit_intercept=False
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    st.subheader(
        "Regression Model Information"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Training Samples",
        len(X_train)
    )

    col2.metric(
        "Testing Samples",
        len(X_test)
    )

    st.subheader(
        "Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{mae:.4f}"
    )

    col2.metric(
        "MSE",
        f"{mse:.4f}"
    )

    col3.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    st.subheader(
        "Regression Coefficients"
    )

    st.write(
        f"Intercept: {model.intercept_:.6f}"
    )

    st.write(
        f"Memory Coefficient: "
        f"{model.coef_[0]:.6f}"
    )

    st.write(
        f"vCPU Coefficient: "
        f"{model.coef_[1]:.6f}"
    )

    prediction_results = pd.DataFrame({
        "Actual Cost": y_test.values,
        "Predicted Cost": y_pred
    })

    st.subheader(
        "Actual vs Predicted On-Demand Costs"
    )

    fig = px.scatter(
        prediction_results,
        x="Actual Cost",
        y="Predicted Cost",
        title="Actual vs Predicted On-Demand Costs"
    )

    max_cost = max(
        prediction_results[
            "Actual Cost"
        ].max(),
        prediction_results[
            "Predicted Cost"
        ].max()
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_cost,
        y1=max_cost
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader(
        "Predict a New EC2 Instance Cost"
    )

    prediction_memory = st.number_input(
        "Memory (GiB)",
        min_value=0.5,
        value=4.0,
        step=0.5
    )

    prediction_vcpus = st.number_input(
        "vCPUs",
        min_value=1,
        value=2,
        step=1
    )

    new_instance = pd.DataFrame({
        "Memory_GiB": [
            prediction_memory
        ],
        "vCPU_Count": [
            prediction_vcpus
        ]
    })

    predicted_cost = model.predict(
        new_instance
    )[0]

    st.metric(
        "Predicted On-Demand Hourly Cost",
        f"${predicted_cost:.4f}"
    )

    st.write(
        f"Estimated Monthly Cost: "
        f"${predicted_cost * 730:.2f}"
    )