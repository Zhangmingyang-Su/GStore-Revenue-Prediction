import numpy as np
import pandas as pd
from Data_transformation import load_df
from Feature_engineering import feature_engineering
from Feature_engineering import data_preprocessing
from model import TabularDataset, FeedForwardNN


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


pd.options.display.max_columns = None



def embedding(cat_col_labels1, cat_col_labels2, max_values):
    emb_dims1 = []
    emb_dims2 = []
    for i in cat_col_labels1:
        emb_dims1.append((max_values[i], min((max_values[i]+1)//2, 50)))
    for i in cat_col_labels2:
        emb_dims2.append((max_values[i], min((max_values[i]+1)//2, 50)))
    return emb_dims1, emb_dims2



# Model Training
def model_application(model, train_dataloader, val_dataloder, epochs, criterion = torch.nn.MSELoss(), optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)):
    total_data = train_dataset.__len__()
    print_every = 500
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        model.train()
        for index, datas in enumerate(train_dataloader):
            steps += 1
            y, num_x, cat_x1, cat_x2 = datas
            cat_x1 = cat_x1.to(device)
            cat_x2 = cat_x2.to(device)
            num_x = num_x.to(device)
            y  = y.to(device)

            # Forward Pass
            optimizer.zero_grad()
            preds = model.forward(num_x, cat_x1, cat_x2)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for val_index, val_datas in enumerate(val_dataloder):
                        y, num_x, cat_x1, cat_x2 = val_datas
                        cat_x1 = cat_x1.to(device)
                        cat_x2 = cat_x2.to(device)
                        num_x = num_x.to(device)
                        y  = y.to(device)

                        out = model.forward(num_x, cat_x1, cat_x2)
                        batch_loss = criterion(out, y)
                        val_loss += batch_loss.item()

                print(f"Epoch {epoch+1}/{no_of_epochs}.."
                         f"Train loss:{running_loss/print_every:.3f}.."
                         f"Validation loss:{val_loss/len(val_dataloder):.3f}..")
                running_loss = 0
                model.train()

if if __name__ == '__main__':
    train = load_df()
    train, max_values = feature_engineering(train)
    x_train, x_val, y_train, y_val = data_preprocessing(train)
    cat_col_labels1 = ["channelGrouping", "device.deviceCategory", "device.operatingSystem", "geoNetwork.continent",
                       "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType",
                       "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot",
                       "trafficSource.campaign", "trafficSource.medium", "geoNetwork.region"]

    cat_col_labels2 = ["browser_category", "browser_operatingSystem", "source_country", "device.browser", "geoNetwork.city",
                       "trafficSource.source", "trafficSource.keyword", "trafficSource.adwordsClickInfo.gclId", "geoNetwork.networkDomain",
                       "geoNetwork.country", "geoNetwork.metro", "geoNetwork.region"]

    num_cols.remove("totals.transactionRevenue")
    emb_dims1, emb_dims2 = embedding(cat_col_labels1, cat_col_labels2, max_values)
    # model application
    train_dataset = TabularDataset(x_data=x_train, y_data=y_train, cat_cols1=cat_col_labels1, cat_cols2=cat_col_labels2, num_cols=num_cols)
    val_dataset = TabularDataset(x_data=x_val, y_data=y_val, cat_cols1=cat_col_labels1, cat_cols2=cat_col_labels2, num_cols=num_cols)

    batchsize = 64
    train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=0)
    val_dataloder = DataLoader(val_dataset, 64, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNN(emb_dims1=emb_dims1,
                          emb_dims2=emb_dims2,
                          no_of_num=len(num_cols),
                          lin_layer_sizes=[128,64,32,16],
                          output_size=1,
                          lin_layer_dropouts=[0.1, 0.1, 0.05],
                          emb_dropout=0.05).to(device)
    model_application(model, train_dataloader, val_dataloder, 100, criterion = torch.nn.MSELoss(), optimizer = torch.optim.Adam(model.parameters(), lr=0.0001))
