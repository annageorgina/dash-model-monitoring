from shared_functions import st, np, pd, df, leaderboard_data, df_mapping, find_metric
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

# Dashboard settings
st.title("Monitor your model and get closer to Utopia ...")
with st.sidebar:
    st.header('Filters', anchor=None)
    selected_df = st.selectbox('Select your ML Model', list(df_mapping.keys()),key="<uniqueSelect>" )

    # creating single element container so dash updated realtime
placeholder = st.empty()


cost = find_metric(selected_df,"cost")
cf_matrix = find_metric(selected_df,"cf_matrix")
fpr, tpr, thresholds_roc = find_metric(selected_df,"roc")
auc = find_metric(selected_df,"auc")
precision, recall, thresholds_pr = find_metric(selected_df,"precision_recall")
acur_score = find_metric(selected_df,"accuracy_score")
cali_curve_x, cali_curve_y = find_metric(selected_df,"calibration_curve")
rank = find_metric(selected_df,"rank")
f1 = find_metric(selected_df,"f1_score")
cost_credit_ratio = find_metric(selected_df,"cost_credit_ratio")
errors = find_metric(selected_df,"count_errors")
cost_FN = find_metric(selected_df,"cost_FN")
cost_FP = find_metric(selected_df,"cost_FP")


# creating dashboard componenets
with placeholder.container():
    kpi4, kpi5, kpi6, kpi7= st.columns(4)
    kpi4.metric(
        value=f"{rank}",
        label="Model Rank :trophy:",
    )
    kpi5.metric(
        label="Errors :dart:",
        value=f"{errors}",
    )
    kpi6.metric(
        label="Cost/Credit Ratio :scales:",
        value=f"{abs(cost_credit_ratio):,.2f}%",
    )
    kpi7.metric(
        label="Cost :money_with_wings:",
        value=f"$ {abs(cost):,.0f}",
    )

    st.divider()

    kpi7, kpi8, kpi9= st.columns(3)
    kpi7.metric(
        label="AUC",
        value=f"{auc:,.2f}",
    )
    kpi8.metric(
        label="Accuracy Score :dart:",
        value=f"{acur_score:,.2f}",
    )
    kpi9.metric(
        label="F1 score",
        value=f"{f1:,.2f}",
    )
    st.divider()

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("#### Confusion Matrix")
        fig_cm = go.Figure()
        group_names = ['True Negative','False Positive','False Negative','True Positive']
        group_counts = ["{0:,.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.0%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
        group_costs = [0,cost_FP,cost_FN,0]
        labels = [f"{v1}<br>{v2}<br>{v3}<br><br>Cost: $ {abs(v4):,.0f}" for v1, v2, v3, v4 in zip(group_names,group_counts,group_percentages, group_costs)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=labels, fmt='', cmap='Blues')
        fig_cm = ff.create_annotated_heatmap(
            z=cf_matrix / np.sum(cf_matrix),
            annotation_text=labels,
            colorscale='Blues',
        )
        fig_cm.update_layout(
            xaxis=dict(title='Predicted',side='top'),
            yaxis=dict(title='True'),
            margin=dict(l=0, r=0, t=0, b=0),
            width=500,
            height=500
        )
        fig_cm.update_yaxes(scaleanchor="x", scaleratio=1)
        # fig.update_xaxes(constrain='domain')
        st.write(fig_cm)

    with fig_col2:
        st.markdown("#### Precision Recall Curve")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines'))
        fig_pr.add_shape(
            type='line', line=dict(dash='dash', color='white'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig_pr.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_pr.update_xaxes(constrain='domain')
        fig_pr.update_layout(
            xaxis=dict(title='Recall'),
            yaxis=dict(title='Precision'),
            margin=dict(l=0, r=0, t=0, b=0),
            width=500,
            height=500
        )
        st.write(fig_pr)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("#### Receiver Operating Characteristic")
        fig_roc = go.Figure()
        fig_roc.add_scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'AUC: {auc:.2f}',
            line=dict(color='rgb(0, 118, 189)')
        )

        fig_roc.update_layout(
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            legend=dict(x=0.75, y=0.15, bgcolor='rgba(0, 0,0, 0)'),
            margin=dict(l=20, r=20, t=30, b=10),
            showlegend=True,
            width=500,
            height=500
        )

        st.plotly_chart(fig_roc)

    with fig_col2:
        st.markdown("#### Calibration Curve")
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideally Calibrated',line=dict(dash='dash', color='white') ))
        fig_calib.add_trace(go.Scatter(x=cali_curve_x, y=cali_curve_y, mode='lines', name='Model'))

        fig_calib.update_layout(
            xaxis=dict(title='Average Predicted Probability in each bin'),
            yaxis=dict(title='Ratio of Positives'),
            legend=dict(x=0.75, y=0.15, bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=20, r=20, t=30, b=10),
            width=500,
            height=500
        )

        st.plotly_chart(fig_calib)
