from shared_functions import st, np, pd, df, table_data,leaderboard_data, df_mapping, find_metric
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

# page config -> needs to come first for app
# values in HTML tags
st.set_page_config(
    page_title="J3 Dashboard",
    page_icon="✅",
    layout="wide",
)

st.session_state['df'] = df
st.session_state['leaderboard_data'] = leaderboard_data

# Dashboard settings
st.title("What's at Play?")
placeholder = st.empty()

with placeholder.container():

    # create three columns
    kpi1, kpi2, kpi3= st.columns(3)
    #kpi1.subheader("Models :bank:")
    kpi1.metric(
        label="Models :bank:",
        value=len(leaderboard_data)-2,
    )
    #kpi2.subheader("Loan requests :money_mouth_face:")
    kpi2.metric(
        label="Loan requests :money_mouth_face:",
        value=df.shape[0],
    )
    #kpi3.subheader("Requested credit ＄")
    tot_credit = np.sum(df["AMT_CREDIT"])
    kpi3.metric(
        label="Requested credit ＄",
        value= f"$ {tot_credit:,.0f} ",
    )

    st.divider()

    st.subheader("Credit & Cost moving average")

    # window_size = st.slider("Choose moving average window", min_value=1, max_value=100, step=1)

    # df['AMT_CREDIT_MA'] = df['AMT_CREDIT'].rolling(window=window_size, min_periods=1).mean()
    # df['COST_MA'] = -1*df['COST'].rolling(window=window_size, min_periods=1).mean()
    # df['RATIO_CC_MA'] = -1*df['RATIO_CC'].rolling(window=window_size, min_periods=1).mean()

    # fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    # # Add AMT_CREDIT line
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['AMT_CREDIT_MA'], mode='lines',line_shape='spline', name='AMT_CREDIT Moving Average'), secondary_y=False)

    # # Add COST line
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['COST_MA'], mode='lines', name='COST Moving Average'), secondary_y=False)

    # # Add RATIO_CC line on the right-hand axis
    # fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RATIO_CC_MA'], mode='lines', name='RATIO_CC Moving Average'), secondary_y=True)

    # # Set layout options
    # fig.update_layout(
    #     xaxis=dict(
    #         tickformat='%Y-%m-%d %H:%M:%S',
    #         tickangle=45,
    #         showticklabels=True,
    #         title='Timestamp',
    #     ),
    #     legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
    #     height=600,
    #     margin=dict(l=20, r=20, t=1, b=20),
    # )

    # # Set the y-axis titles
    # fig.update_yaxes(title_text='$ Credit or Cost', secondary_y=False)
    # fig.update_yaxes(title_text='Ratio cost/credit', secondary_y=True)

    # st.plotly_chart(fig, use_container_width=True, width='100%', height='1200px')

    # st.header("🏇 The Race To ML Model Utopia 🚀", anchor=None)
    # st.write(show_leaderboard(leaderboard_data))


    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    window_size = st.slider("Choose moving average window", min_value=1, max_value=48, step=1)

    window_size_offset = f'{window_size}H'
    df['AMT_CREDIT_MA'] = df['AMT_CREDIT'].rolling(window=window_size_offset, min_periods=1).mean()
    df['COST_MA'] = -1 * df['COST'].rolling(window=window_size_offset, min_periods=1).mean()
    df['RATIO_CC_MA'] = -1*df['RATIO_CC'].rolling(window=window_size_offset, min_periods=1).mean()


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['AMT_CREDIT_MA'], mode='lines', line_shape='spline', name='AMT_CREDIT Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['COST_MA'], mode='lines',line_shape='spline', name='COST Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RATIO_CC_MA'], mode='lines',line_shape='spline', name='RATIO_CC Moving Average', yaxis='y2'))

    for i, model in enumerate(leaderboard_data):
        model_name = model['model_name']
        if model_name not in ['Random Assignment', 'Utopia']:
            min_ts = find_metric(model_name,"start")
            fig.add_shape(
                type="line",
                x0=min_ts,
                y0=min(df[['AMT_CREDIT_MA', 'COST_MA']].min()),
                x1=min_ts,
                y1=max(df[['AMT_CREDIT_MA', 'COST_MA']].max()),
                line=dict(color="red", width=1),
                xref="x",
                yref="y",
            )
            fig.add_annotation(
                x=min_ts,
                y=max(df[['AMT_CREDIT_MA', 'COST_MA']].max()),
                text=f'{model_name}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                xref="x",
                yref="y",
                font=dict(color="red", size=12),
            )

    fig.update_layout(
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M:%S',
            tickangle=45,
            showticklabels=True,
            title='Timestamp',
        ),
        yaxis=dict(
            title='$ Credit or Cost'
        ),
        yaxis2=dict(
            title='% Ratio cost/credit',
            overlaying='y',
            side='right',
            griddash='dot',
            gridwidth=1,
            tickformat='.0%',
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.75),
        height=600,
        width=500,
        margin=dict(l=20, r=20, t=1, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, width='500', height='600')



st.header("Model details :1234:", anchor=None)
st.dataframe(
    pd.DataFrame(table_data),
    use_container_width=True,
)

st.divider()

color_scale = px.colors.sequential.Viridis


fig_col1, fig_col2 = st.columns([0.5,0.5], gap="small")
with fig_col1:
    st.markdown("#### Receiver Operating Characteristic")
    fig_roc = go.Figure()
    for i, model in enumerate(leaderboard_data):
        model_name = model['model_name']
        fpr, tpr, thresholds_roc = find_metric(model_name,"roc")
        auc = find_metric(model_name,"auc")
        max_rank = max(model['rank'] for model in leaderboard_data)
        rank = model['rank']
        c= len(leaderboard_data)-rank-1
        fig_roc.add_scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f' {model_name} - AUC: {auc:.2f}',
            line=dict(color=color_scale[c])
            )

    fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_roc.update_xaxes(constrain='domain')
    fig_roc.update_layout(
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
    )

    st.plotly_chart(fig_roc, use_container_width=True)

with fig_col2:
    st.markdown("#### Precision Recall Curve")
    fig_pr = go.Figure()

    max_rank = max(model['rank'] for model in leaderboard_data)

    for i, model in enumerate(leaderboard_data):
        model_name = model['model_name']
        precision, recall, thresholds_pr = find_metric(model_name, "precision_recall")
        rank = model['rank']
        c= len(leaderboard_data)-rank-1
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, name=model_name, mode='lines', line=dict(color=color_scale[c])))

    fig_pr.add_shape(
        type='line', line=dict(dash='dash', color='white'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig_pr.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_pr.update_xaxes(constrain='domain',scaleanchor="x",scaleratio=1)
    fig_pr.update_layout(
        xaxis=dict(title='Recall'),
        yaxis=dict(title='Precision'),
        legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1, ),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(fig_pr, use_container_width=True)





