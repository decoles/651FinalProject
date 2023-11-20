import networkx as nx
import numpy as np
import ndlib.models.ModelConfig as mc
import matplotlib.pyplot as plt
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_networkx

#GITHUB COPILOT WAS USED IN THIS PROJECT FOR OUTPUT OF THE GCN MODEL
#https://github.com/njmarko/machine-learning-with-graphs/blob/master/MyAttempts/CS224W_Colab_0.ipynb For GCN MODEL EXAMPLE

def GenerateSirModel(N):
    #g = nx.erdos_renyi_graph(N, 0.2, seed=42)
    g = nx.erdos_renyi_graph(N, 0.2)

    #add attributes to nodes
    for i in range(N):
        g.nodes[i]['state'] = 0

    #keep node layout
    pos = nx.spring_layout(g)

    #get random node to infect graph
    randomInfectedNode = np.random.randint(0, N)

    model = ep.SIRModel(g)

    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.toarray())
    largestEigvalue = max(e) #largest eigenvalue of the graph for lambda

    #similar configurations as the barbasi paper
    R0 = 1.2
    lambda1 = largestEigvalue
    gamma = 0.4
    alpha = 0.5
    fraction_infected = 0.001
    beta = R0 * gamma/largestEigvalue
    ITERATIONS = 15

    '''
    CODE FOR SIER MODEL
    '''
    #pick a random number from 0 to ITERATIONS to slect as a 
    # # Model Configuration
    # cfg = mc.Configuration()
    # cfg.add_model_parameter('beta', beta)
    # cfg.add_model_parameter('lambda1', lambda1)
    # cfg.add_model_parameter('gamma', gamma)
    # cfg.add_model_parameter('alpha', alpha)
    # infected_nodes = [node]
    # cfg.add_model_initial_configuration("Infected", infected_nodes)

    cfgSIR = mc.Configuration()
    cfgSIR.add_model_parameter('beta', beta) 
    cfgSIR.add_model_parameter('gamma', gamma)
    infected_nodes = [randomInfectedNode]
    cfgSIR.add_model_initial_configuration("Infected", infected_nodes)

    model.set_initial_status(cfgSIR)
    #model2.set_initial_status(cfg)

    # Simulation execution
    iterations = model.iteration_bunch(ITERATIONS)

    #represent all node attributes at given time (NEED TO UPDATE TO FILL IN BLANK DATA)
    NodeFeatures = []

    for i in iterations:
        #get status of each node
        status = i['status']
        allNodeStatus = i['node_count']
        #update Graph nodes
        for j in status:
            g.nodes[j]['state'] = status[j]
        
        NodeFeatures.append(status)


    #draw graph
    tempDict = {i: 0 for i in range(N)}
    #for allowing node fill in between empty spots in data

    randomFeature = []

    #get random set of features
    randomFeatureNumber = np.random.randint(0, ITERATIONS)
    #fill in missing spots of data
    for i in range(1, len(NodeFeatures)):
        for j in range(N):
            previous_value = None
            if j not in NodeFeatures[i]: #if key not present update temp with a previous value
                previous_value = NodeFeatures[i-1][j]
                tempDict[j] = previous_value
                #NodeFeatures[i][j] = previous_value
            else:
                value = NodeFeatures[i][j]
                tempDict[j] = value
        NodeFeatures[i] = tempDict
        

    #maintain a feature of the infected and original, original for x, infected for y

    randomFeature = list(NodeFeatures[randomFeatureNumber].values())
    #set node of feature to -1 to show the original infected node
    randomFeatureInfected = randomFeature.copy()
    randomFeatureInfected[randomInfectedNode] = 1
    return g, randomFeature, randomFeatureInfected, randomInfectedNode

    

'''
    CODE FOR GCN MODEL TO PREDICT NODES
'''
def GCN(train_loader, val_loader, test_loader, num_epochs):
    #INITIALIZE MODEL
    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(1, 16) # 1 input feature
            self.conv2 = GCNConv(16, 4) 

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x

    model_gnn = GCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_gnn.parameters(), lr=0.01)

    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model_gnn.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model_gnn(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        model_gnn.eval()
        correct_train = 0
        total_train = 0
        correct_test = 0

        '''
            Code for testing, training, and validation metrics boilerplate code from COPILOT
        '''
        with torch.no_grad():
            for data in train_loader: #for each data piece
                output_train = model_gnn(data)
                _, predicted_train = output_train.max(1)
                total_train += data.y.size(0)
                correct_train += predicted_train.eq(data.y).sum().item()

        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in val_loader:
                output_val = model_gnn(data)
                _, predicted_val = output_val.max(1)
                total_val += data.y.size(0)
                correct_val += predicted_val.eq(data.y).sum().item()

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in test_loader:
                output_test = model_gnn(data)
                _, predicted_test = output_test.max(1)
                total_test += data.y.size(0)
                correct_test += predicted_test.eq(data.y).sum().item()
                # nx_graph = to_networkx(data)

                # # Visualize the graph with node colors based on predictions
                # plt.figure(figsize=(8, 8))
                # pos = nx.spring_layout(nx_graph)
                # nx.draw_networkx_nodes(nx_graph, pos, node_color=predicted_test.numpy(), cmap=plt.cm.tab10)
                # nx.draw_networkx_edges(nx_graph, pos)
                # plt.show()

        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        test_accuracy = 100 * correct_test / total_test

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #visualize in a graph

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
        print(test_accuracy)

    #Plot the accuracy over epochs
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    #save graph
    #plt.savefig('accuracy.png')
    
    #nx.draw_networkx(to_networkx(data), node_color=output_train.argmax(dim=1), cmap='coolwarm')


'''
    CODE FOR GENERATING DATA
'''
def create_data(g, randomFeature, randomFeatureInfected, randomInfectedNode):
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    #y is the selected features of the graph at random time
    y = torch.tensor(randomFeature).long()
    #x is the selected features with the infected node present to show correct prediction
    x = torch.tensor(randomFeatureInfected).unsqueeze(1).float()
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def main():
    #CONSTANTS
    DATASETSIZE = 200 #how many datasets to get
    N = 100 #Graph size
    EPOCHS = 100

    graphDataset = [GenerateSirModel(N) for i in range(DATASETSIZE)]
    
    #80% train, 10% val, 10% test
    graphDataLST = [create_data(*graph) for graph in graphDataset]
    train_data, test_data = train_test_split(graphDataLST, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    #print(list(train_loader))
    GCN(train_loader, val_loader, test_loader, EPOCHS )

if __name__ == "__main__":
    main()

