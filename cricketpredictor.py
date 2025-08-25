import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
import zipfile
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def read_games_from_zip(zip_file_path):

    """
    Function that reads in the zip file of all the games from a given team
    and returns a pandas dataframe of all the data within the zip file

    """

    data = []
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename.endswith('.json'):
                    with zip_ref.open(zip_info.filename) as file:
                        json_data = json.load(file)
                        data.append(json_data)
                else:
                    print(f"Skipping unsupported file: {zip_info.filename}")  

    except zipfile.BadZipFile as e:
        raise ValueError(f"Error opening zip file: {e}") from e

    if not data:
        raise ValueError("No JSON files found in the zip archive.")

    return pd.DataFrame(data)  

def calculate_cricket_stats(innings_data):
    """
    Calculates runs/over, total runs, runs/wicket and runs between wickets for an innings.

    This was done as more of a warm up as i was unsure what direction to go in at
    the start so made this just to get the ball rolling and is messy and not very relevant

    """

    runs_per_over = []  
    total_runs = [0]
    runs_per_wicket = []
    runs_since_wicket = 0
    runs_between_wickets = [0]
    wickets = 0
    wides = 0

    for over in innings_data['overs']:
        no_overs = 0
        over_runs = []
        for ball in over['deliveries']:
            runs = ball.get('runs', {}).get('total', 0) 
            wide = ball.get('extras', {}).get('wides', 0) 
            wides += wide
            over_runs.append(runs)
            total_runs.append(total_runs[-1] + runs)
            runs_per_wicket.append(total_runs[-1] / (wickets+1))
            runs_since_wicket += runs
            if ball.get('wickets', None):  
                wickets += 1
                runs_between_wickets.append(runs_since_wicket)
                runs_since_wicket = 0

        no_overs += 1

        runs_by_over = np.sum(over_runs)/no_overs
        runs_per_over.append(runs_by_over.item())


    return runs_per_over, total_runs[1:], runs_per_wicket, runs_between_wickets[1:], wickets

def plot_cricket_stats(runs_p_over, total_runs, runs_p_wicket, wickets, labels, title):
    """
    Plots statistics that are calculated in the calculate_cricket_stats function. Only the runs per over and total runs 
    are plotted due to the fact that these were actually used in the other parts of the code.
    """

    fig_1, axes = plt.subplots(2, 1, figsize=(12, 6))
    plt.subplots_adjust(left=0.15, bottom=0.04, right=0.9, top=0.96)

    axes[0].plot(runs_p_over, label=labels[0])
    axes[0].set_title(labels[0])
    axes[0].set_xlabel('Over')
    axes[0].set_ylabel('Runs')
    for i in range(0, len(runs_p_over)):
        axes[0].axvline(x=i,linestyle='--', linewidth=0.2)


    axes[1].plot(total_runs, label=labels[1])
    axes[1].set_title(labels[1])
    axes[1].set_xlabel('Ball')
    axes[1].set_ylabel('Cumulative Runs')
    for i in range(0, len(total_runs),10):
        axes[1].axvline(x=i,linestyle='--', linewidth=0.2)


    plt.tight_layout()
    plt.show()

    return 0

def pad_tensor(tensor, desired_length, padding_value):
    """
    This function pads the tensors that are going into the model as all 'sequences' must be the same length and not
    ever game had the same number of balls due to wides/no balls etc.

    It just duplicates the final innings score up to the length of the longest game in the set and returns a tensor

    """
    padding_length = desired_length - len(tensor)
    padding_tensor = torch.full((padding_length,), padding_value)
    return padding_tensor

class WeightedMSELoss(nn.Module):
    """

    This class defines a custom loss function that adjusts the standard mean squared error and adjusts the weighting
    so that predicted scores later in a match hold more weight to try and make the model prioritise getting these values correct. 

    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, input, target):
        loss = (input - target)**2
        weighted_loss = loss * self.weights
        return weighted_loss.mean()

def extract_team_data(main_data,desired_team):  
    """
    This function goes through the data obtained through the zip file and gets all of the data for a desired team. In this case,
    the Rajasthan Royals were selected. The function also checks that there were 2 innings in the match and that only innings in
    which all 20 overs were played are used. 

    """  
    royals_innings = []
    for i in range(len(main_data)):
        innings_s = main_data.loc[i]['innings']
        if len(innings_s) == 2:
            if innings_s[0]['team'] == desired_team and len(innings_s[0]['overs']) == 20:
                royals_innings.append(innings_s[0])
            if innings_s[1]['team'] == desired_team and len(innings_s[1]['overs']) == 20:
                royals_innings.append(innings_s[1])
        else:
            continue        

    return royals_innings

def model_data_preparation(data):
    """
    This function extracts data using the calculate_cricket_stats function and then converts all of it into
    tensors so that it can be fed into the model

    """

    all_runs_per_over = []
    all_total_no_runs_accum = []
    all_runs_per_wicket = []
    all_runs_separating_wickets = []

    for i in range(len(data)):
        runs_per_over, total_no_runs_accum, runs_per_wicket, runs_separating_wickets, wickets = calculate_cricket_stats(data[i])
        all_runs_per_over.append(torch.tensor(runs_per_over))
        all_total_no_runs_accum.append(torch.concat((torch.tensor(total_no_runs_accum),pad_tensor(total_no_runs_accum,133,total_no_runs_accum[-1]))))
        all_runs_per_wicket.append(runs_per_wicket)
        all_runs_separating_wickets.append(runs_separating_wickets)

    all_runs_per_over_tensor = torch.stack(all_runs_per_over, dim=0)
    all_total_no_runs_accum_tensor = torch.stack(all_total_no_runs_accum, dim=0)

    return all_runs_per_over_tensor, all_total_no_runs_accum_tensor, all_runs_per_wicket, all_runs_separating_wickets

class ScorePredictorRNN(nn.Module):
    """
    
    This is the model. A simple LSTM that takes in data of an innings up to any given ball and then predicts 
    the final run score. 

    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ScorePredictorRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.linear = nn.Linear(input_size, output_size) 
        self.dropout = nn.Dropout(p=0.4)
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.linear.weight.data.normal_(mean=4.0, std=2.5)
        self.linear.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  
        out = self.dropout(out)
        out = torch.relu(out)
        out = self.fc(out)
        return out
    

def train_cycle(model,model_data,num_epochs,no_seen_balls,batch_size):
    """
    
    This function trains the model 
    
    """

    no_test_batches = 1
    no_batches = int(np.floor(model_data.shape[0]/batch_size)) - no_test_batches
    cutoff = no_batches*batch_size

    no_unseen_balls = model_data.shape[1] - no_seen_balls
    input_data = model_data[:int(cutoff), :no_seen_balls,]
    targets = model_data[:int(cutoff), no_seen_balls:,]

    batched_input_data = input_data.reshape((no_batches,batch_size,no_seen_balls,1))
    batched_targets = targets.reshape((no_batches,batch_size,no_unseen_balls))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = WeightedMSELoss(weights = torch.exp(torch.linspace(0, 1, no_unseen_balls)))
    #criterion = nn.MSELoss()

    outputs = []
    known_outputs = []
    losses = []
    for epoch in range(num_epochs):
        model.train()
        
        for i in range(batched_input_data.shape[0]):

            output = model(batched_input_data[i].float())
            loss = criterion(output, batched_targets[i].float())

            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs.append(output)
            known_outputs.append(batched_targets[i])

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if np.round(loss.item(), 4) == 0:
            print('Training complete at', epoch)
            break

    return outputs, known_outputs, np.min(losses)

def test_cycle(model, model_data, no_seen_balls, batch_size):
    """
    
    This function tests the model meaning that it gives the model an unseen dataset and assesses its output
    without then adjusting its parameters
    
    """

    model.eval()

    no_unseen_balls = model_data.shape[1] - no_seen_balls
    test_data = model_data[:,:no_seen_balls,]
    targets = model_data[:,no_seen_balls:,]
    no_test_batches = 1

    criterion = WeightedMSELoss(weights = torch.exp(torch.linspace(0, 1, no_unseen_balls)))

    no_batches = int(test_data.shape[0]/batch_size)
    batched_test_data = test_data[-batch_size*no_test_batches:].reshape((no_test_batches,batch_size,no_seen_balls,1))
    batched_test_targets = targets[-batch_size*no_test_batches:].reshape((no_test_batches,batch_size,no_unseen_balls))

    with torch.no_grad():

        test_output = model(batched_test_data[0].float())

    loss = criterion(test_output, batched_test_targets[0].float())  

    return test_output, batched_test_targets[0]

def LSTM_main(input_size=1, hidden_size=64, num_layers=3, num_epochs=2000, no_seen_balls=40):
    """
    
    This function controls the model. It allows you to adjust hyperparameters and prints out the 
    results of the model into the terminal
    
    A small sample of training data is printed and then the testing data is printed for the test batch. The code is written
    in a way that only one batch is used fo testing. This is due to lack of time.

    NOTE: Batch size 1 has given the best results so that is how its been left but this was discovered late so the code is written
    assuming larger batch sizes

    """


    game_zip = "rajasthan_royals_male_json.zip"
    data = read_games_from_zip(game_zip)
    desired_team = 'Rajasthan Royals'
    batch_size = 1

    royals_data = extract_team_data(data,desired_team)
    _,model_input,_,_ = model_data_preparation(royals_data)
    output_size = len(model_input[1]) - no_seen_balls
    model = ScorePredictorRNN(input_size, hidden_size, num_layers, output_size)
    output, desired_output  = train_cycle(model, model_input, num_epochs, no_seen_balls, batch_size)
    test_output, desired_test_output = test_cycle(model,model_input,no_seen_balls,batch_size)
    print("A random batch from the end of training and the difference in the predicted and actual final runs is listed for each game below:")
    for i in range(batch_size):
        print((torch.round(output[-2:][0][i][-1]) - desired_output[-2:][0][i][-1]).item())
    print("The test data is shown below for the test batch after training:")
    for i in range(batch_size):    
        print("The test data aimed for a final run score of", torch.round(desired_test_output[i][-1]).item(),"and output a score of", torch.round(test_output[i][-1]).item())


def analysis():
    """
    
    This function controls the data analysis and the plots. The plots are of the 
     average values over all royals matches. This, again, is not very relevant.
    
    """
    game_zip = "rajasthan_royals_male_json.zip"
    data = read_games_from_zip(game_zip)
    desired_team = 'Rajasthan Royals'

    royals_data = extract_team_data(data, desired_team)
    all_rpo = []
    all_runs = []
    all_rpw = []
    all_rbw = []
    all_wickets = []
    for i in range(len(royals_data)):
        runs_per_over, accumulated_runs, runs_per_wicket, runs_between_wickets, wickets = calculate_cricket_stats(royals_data[i])
        all_rpo.append(runs_per_over)
        all_runs.append(accumulated_runs[:115])
        all_rpw.append(runs_per_wicket[:115])
        #all_rbw.append(runs_between_wickets[:2])
        all_wickets.append(wickets)
    avg_rpo = np.mean(np.array(all_rpo),axis=0)
    avg_runs = np.mean(np.array(all_runs), axis=0)
    avg_rpw = np.mean(all_rpw)
    #avg_rbw = np.mean(np.array(all_rbw),axis=0)
    avg_wickets = np.mean(all_wickets)

    plot_cricket_stats(avg_rpo, avg_runs, avg_rpw, avg_wickets, labels = ["Runs per Over", "Total Runs Scored", "Runs between wickets"], title='Runs analysis' )
    print("Average run per over:", np.round(avg_rpo,2))
    print("Average total runs at each ball:", np.round(avg_runs,2))
    print("Average runs per wicket:", np.round(avg_rpw,2))
    print("Average wickets conceded per game:", np.round(avg_wickets))

def find_most_similar(query_point, data):
    """
    
    This function does the clustering for game_search using KMeans
    
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_query = scaler.transform(query_point)  

    kmeans = KMeans(n_clusters=100)  
    kmeans.fit(scaled_data)
    cluster_centers = kmeans.cluster_centers_

    distances = [np.linalg.norm(scaled_query - center) for center in cluster_centers]
    closest_cluster = np.argmin(distances)
    similar_matches = data[kmeans.labels_ == closest_cluster]
    similar_indices = similar_matches.index

    return similar_matches, similar_indices

def game_search(no_overs_completed):
    """
    
    This function uses clustering techniques to group games together based on similarity. The idea here
    is for the user to input how many overs they are into a game and the function will assess all games of the
    desired team and find the most similar games. It will then show those games and the total run score for those games
    giving an indication as to what the final run score may be. Due to lack of time not much analysis is done on these to give 
    just one final score prediction but the number of clusters can be adjusted to get the most similar game.
    
    """

    game_zip = "rajasthan_royals_male_json.zip"
    data = read_games_from_zip(game_zip)
    desired_team = 'Rajasthan Royals'

    royals_data = extract_team_data(data, desired_team)
    all_rpo,_,_,_ = model_data_preparation(royals_data)
    played_overs = all_rpo[:,:no_overs_completed,]

    played_overs = pd.DataFrame(played_overs.numpy())
    current_game_overs = []
    for i in range(no_overs_completed):
        current_game_overs.append(input(f"Runs in over {i+1}: "))

    current_game_overs = pd.DataFrame(current_game_overs).transpose()
    all_rpo = pd.DataFrame(all_rpo)

    matches,indices = find_most_similar(current_game_overs,played_overs)

    if len(matches) == 1: 
        print(all_rpo.iloc[matches[0]].sum())

    else:
        print("The final runs total of the similar games:")
        print(all_rpo.iloc[indices].sum(axis=1))

    while True:
        response = input("Inspect runs per over for game(s)? (y/n): ")

        if response == "y":
            print(all_rpo.iloc[indices])
            break  

        elif response == "n":
            break  

        else:
            print("Invalid response. Please type 'y' or 'n'.")


def main_menu():
    """
    
    This is the main_menu that acts as an interface to make the different prediction models easy to access. The assignment
    was quite open-ended which is why i decided to do a couple different things.
    
    """

    while True:
        print("=================")
        print("Main Menu\n")
        print("1. Analysis")
        print("2. Train LSTM")
        print("3. Game Search")
        print("4. Exit")
        print("==================")

        choice = input("Enter your choice: ")

        if choice == '1':
            analysis()
        elif choice == '2':
            LSTM_main()
        elif choice == '3':
            overs = int(input("Enter number of overs played: "))
            game_search(overs)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

main_menu()