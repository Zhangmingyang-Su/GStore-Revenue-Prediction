import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import datetime

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from Data_transformation import load_df

train = load_df()

def data_visulization1(train):
    train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
    gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

    plt.figure(figsize=(8,6))
    plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('TransactionRevenue', fontsize=12)
    plt.show()

    num_of_consumption_record = pd.notnull(train["totals.transactionRevenue"]).sum()
    num_of_consumers = (gdf["totals.transactionRevenue"]>0).sum()

    print("数据集中有{}条消费记录，约占数据集的{:.2f}%".format(num_of_consumption_record, num_of_consumption_record/train.shape[0] * 100))
    print("数据集中有{}个消费者，约占数据集的{:.2f}%".format(num_of_consumers, num_of_consumers/gdf.shape[0] * 100))

# 可视化浏览量和消费之间的关系，并给出结论

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace


def data_visulization2(train):
    train['date'] = train['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    cnt_srs = train.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])
    cnt_srs.columns = ["count", "count of non-zero revenue"]
    cnt_srs = cnt_srs.sort_index()
    #cnt_srs.index = cnt_srs.index.astype('str')
    trace1 = scatter_plot(cnt_srs["count"], 'red')
    trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')

    fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                              subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
    py.iplot(fig, filename='date-plots')
