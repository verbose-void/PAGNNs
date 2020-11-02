import torch
import numpy as np
from torch import nn
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, device=torch.device('cpu')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_layer_size).to(device)

        self.linear = nn.Linear(hidden_layer_size, output_size).to(device)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size, device=device),
                            torch.zeros(1,1,self.hidden_layer_size, device=device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


if __name__ == '__main__':
    flight_data = sns.load_dataset('flights')
    
    # plot passengers over time
    """
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams['figure.figsize'] = fig_size
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x',tight=True)
    plt.plot(flight_data['passengers'])
    plt.show()
    """

    all_data = flight_data['passengers'].values.astype(float)
    test_data_size = 12
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    train_window = 12
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = LSTM(device=device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 500

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                            torch.zeros(1, 1, model.hidden_layer_size, device=device))
    
            y_pred = model(seq)
    
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
    
        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    fut_pred = 12

    test_inputs = train_data_normalized[-train_window:].tolist()

    model.eval()
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                            torch.zeros(1, 1, model.hidden_layer_size, device=device))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

    x = np.arange(132, 144, 1)
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(flight_data['passengers'], label='Ground Truth')
    plt.plot(x,actual_predictions, label='LSTM predictions')
    plt.legend()
    plt.show()

    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    plt.plot(flight_data['passengers'][-train_window:], label='Ground Truth')
    plt.plot(x,actual_predictions, label='LSTM predictions')
    plt.legend()
    plt.show()
