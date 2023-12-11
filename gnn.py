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
import torch.nn.functional as F

#GITHUB COPILOT WAS USED IN THIS PROJECT FOR OUTPUT OF THE GCN MODEL
#https://github.com/njmarko/machine-learning-with-graphs/blob/master/MyAttempts/CS224W_Colab_0.ipynb For GCN MODEL EXAMPLE

def TestGraph(N):
    #g = nx.erdos_renyi_graph(N, 0.2)
    g = nx.barabasi_albert_graph(N, 3)

    #add attributes to nodes
    for i in range(N):
        g.nodes[i]['state'] = 0

    #keep node layout
    pos = nx.spring_layout(g)

    #get random node to infect graph
    randomInfectedNode = np.random.randint(0, N)

    model = ep.SEIRModel(g)

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
    ITERATIONS = 30

    '''
    CODE FOR SIER MODEL
    '''
    #pick a random number from 0 to ITERATIONS to slect as a 
    # # Model Configuration
    cfgSIR = mc.Configuration()
    # cfg = mc.Configuration()
    cfgSIR.add_model_parameter('beta', beta)
    cfgSIR.add_model_parameter('lambda1', lambda1)
    cfgSIR.add_model_parameter('gamma', gamma)
    cfgSIR.add_model_parameter('alpha', alpha)
    infected_nodes = [randomInfectedNode]
    cfgSIR.add_model_initial_configuration("Infected", infected_nodes)

    model.set_initial_status(cfgSIR)

    # Simulation execution
    iterations = model.iteration_bunch(ITERATIONS)

    #represent all node attributes at given time (NEED TO UPDATE TO FILL IN BLANK DATA)
    NodeFeatures = []

    blankEntryCounter = 0
    CompletedIterationsCounter = 0
    for i in iterations:
        #get status of each node
        status = i['status']
        #update Graph nodes
        for j in status:
            g.nodes[j]['state'] = status[j]
        # if len(status) == 0:
        #     blankEntryCounter += 1
        #     if blankEntryCounter == 2:
        #         break
        # else:
        #     blankEntryCounter = 0

        CompletedIterationsCounter += 1
        NodeFeatures.append(status)


    #start at 1 because we dont need to modify the first row
    for i in range(1, len(NodeFeatures)): 
        #now we scan the dicitonary for missing values
        for j in range(N):
            if j not in NodeFeatures[i]:
                NodeFeatures[i][j] = NodeFeatures[i-1].get(j)

    #sort the dictionary keys by ascending order
    for i in range(0, len(NodeFeatures)):
        NodeFeatures[i] = dict(sorted(NodeFeatures[i].items()))


    return g, NodeFeatures, randomInfectedNode, pos

def GenerateSirModel(N, iter):
    randomeNodeCount = np.random.randint(50, N)
    N = randomeNodeCount
    #g = nx.erdos_renyi_graph(N, 0.2, seed=42)
    raondlyPickGraph = np.random.randint(0, 2)
    if raondlyPickGraph == 0:
        g = nx.erdos_renyi_graph(N, 0.2)
    elif raondlyPickGraph == 1:
        g = nx.barabasi_albert_graph(N, 3)
    elif raondlyPickGraph == 2:
        g = nx.random_geometric_graph(N, 0.3)
    #g = nx.erdos_renyi_graph(N, 0.2)
    #g = nx.barabasi_albert_graph(N, 2)

    #add attributes to nodes
    for i in range(N):
        g.nodes[i]['state'] = 0

    #keep node layout
    pos = nx.spring_layout(g)

    #get random node to infect graph
    randomInfectedNode = np.random.randint(0, N)

    model = ep.SEIRModel(g)

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
    ITERATIONS = iter

    '''
    CODE FOR SIER MODEL
    '''
    #pick a random number from 0 to ITERATIONS to slect as a 
    # # Model Configuration
    cfgSIR = mc.Configuration()
    # cfg = mc.Configuration()
    cfgSIR.add_model_parameter('beta', beta)
    cfgSIR.add_model_parameter('lambda1', lambda1)
    cfgSIR.add_model_parameter('gamma', gamma)
    cfgSIR.add_model_parameter('alpha', alpha)
    infected_nodes = [randomInfectedNode]
    cfgSIR.add_model_initial_configuration("Infected", infected_nodes)

    model.set_initial_status(cfgSIR)

    # Simulation execution
    iterations = model.iteration_bunch(ITERATIONS)

    #represent all node attributes at given time (NEED TO UPDATE TO FILL IN BLANK DATA)
    NodeFeatures = []

    blankEntryCounter = 0
    CompletedIterationsCounter = 0
    for i in iterations:
        #get status of each node
        status = i['status']
        #update Graph nodes
        for j in status:
            g.nodes[j]['state'] = status[j]
        # if len(status) == 0:
        #     blankEntryCounter += 1
        #     if blankEntryCounter == 2:
        #         break
        # else:
        #     blankEntryCounter = 0

        CompletedIterationsCounter += 1
        NodeFeatures.append(status)


    randomFeatureNumber = np.random.randint(0, CompletedIterationsCounter)

    #start at 1 because we dont need to modify the first row
    for i in range(1, len(NodeFeatures)): 
        #now we scan the dicitonary for missing values
        for j in range(N):
            if j not in NodeFeatures[i]:
                NodeFeatures[i][j] = NodeFeatures[i-1].get(j)

    #sort the dictionary keys by ascending order
    for i in range(0, len(NodeFeatures)):
        NodeFeatures[i] = dict(sorted(NodeFeatures[i].items()))

    randomFeature = []
    randomFeature = list(NodeFeatures[randomFeatureNumber].values())
    #set node of feature to -1 to show the original infected node
    randomFeatureInfected = randomFeature.copy()
    randomFeatureInfected[randomInfectedNode] = 4
    #randomFeature[randomInfectedNode] = 1
    #print(randomFeature)

    return g, randomFeature, randomFeatureInfected, randomInfectedNode

'''
    CODE FOR GCN MODEL TO PREDICT NODES
'''
def GCN(train_loader, val_loader, test_loader, num_epochs):
    #INITIALIZE MODEL
    torch.manual_seed(42)
    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(2, 16) # 2  input feature
            self.conv2 = GCNConv(16, 5) 

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            
            return x

    model_gnn = GCN()
    weight = torch.tensor([0.1,0.1,0.1,0.1,0.99]).float()
    #weight = torch.tensor([0, 0.98,0.02,0.02]).float()
    #weight = torch.tensor([0.05,0.05,0.05,0.05,0.99]).float()


    criterion = nn.CrossEntropyLoss(weight=weight)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_gnn.parameters(), lr=0.001)
    #sources = torch.tensor([data.source for data in train_loader]).long()

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
                loss = criterion(output_train, data.y)
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
                


        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val
        test_accuracy = 100 * correct_test / total_test

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        #visualize in a graph

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Loss: {loss.item():.4f}')

    g, NodeFeatures, randomInfectedNode, pos = TestGraph(100)
    for i in range(len(NodeFeatures)):
        NodeFeatures[i] = list(NodeFeatures[i].values())
        data = create_data(g, NodeFeatures[i], NodeFeatures[i], randomInfectedNode)
        model_gnn.eval()
        with torch.no_grad():
            output = model_gnn(data)
            test, predicted = output.max(1)
            probabilites = F.softmax(output, dim=1) 
            #print(probabilites)
            #print(probabilites)
            #print(predicted)
            #print(randomInfectedNode)
            #get nodes with high probability to show
            for j in range(len(probabilites[0])):
                if probabilites[0][j] > 0.40:
                    print(j)
                    print(predicted)
            print("")
            
        colors = ['red', 'green', 'blue', 'yellow', 'black']

        

        plt.figure(figsize=(8, 8))
        plt.title(f'Predicted Graph at Time {i} with Infected Node {randomInfectedNode}')
        plt.tight_layout()
        nx.draw_networkx(g, pos = pos, node_color=predicted.numpy(), cmap='coolwarm')
        #save
        plt.savefig(f'./images/{i}.png')
        print("")
        #print(NodeFeatures[i])
    torch.save(model_gnn.state_dict(), './model.pth')

    


    print("")

    #Plot the accuracy over epochs

    plt.figure(figsize=(8, 8))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()


'''
    CODE FOR GENERATING DATA
'''
def create_data(g, randomFeature, randomFeatureInfected, randomInfectedNode):
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    #y is the selected features of the graph at random time
    #x is the selected features with the infected node present to show correct prediction
    arrayTemp = []
    for i in range(len(randomFeatureInfected)):
        arrayTemp.append(0)
    arrayTemp[randomInfectedNode] = 1
    ResultFeatures = [[index, value] for index, value in enumerate(randomFeature)]
    x = torch.tensor(ResultFeatures).float()
    #x= torch.tensor(randomFeature).unsqueeze(1).float()
    y = torch.tensor(randomFeatureInfected).long()
    #y = torch.tensor(arrayTemp).long()


    #source = torch.tensor(source).long()
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def main():
    #CONSTANTS
    DATASETSIZE = 1200 #how many datasets to get
    N = 150 #Graph size
    EPOCHS = 100
    ITERATIONS = 30

    GenerateSirModel(N, ITERATIONS)
    graphDataset = [GenerateSirModel(N, ITERATIONS) for i in range(DATASETSIZE)]
    
    #80% train, 10% val, 10% test
    graphDataLST = [create_data(*graph) for graph in graphDataset]
    train_data, test_data = train_test_split(graphDataLST, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    #print(list(train_loader))
    GCN(train_loader, val_loader, test_loader, EPOCHS )

if __name__ == "__main__":
    main()

