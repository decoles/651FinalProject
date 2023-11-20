import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Person:
    def __init__(self, id, state, time):
        self.id = id
        self.state = "S"
        self.time = time
        self.color = "blue"
        self.I = 0 #infected
        self.S = 0 #susceptible
        self.R = 0 #recovered

def createGraph(N, kAvg, s0):
    G = nx.erdos_renyi_graph(N, kAvg/N, seed=34)

    #initialize nodes
    for node in G.nodes():
        person = Person(node, "S", 0)
        person.S = 1.0
        G.nodes[node]['person'] = person 

    #give one random node an infection
    np.random.seed(34)
    infected = np.random.choice(G.nodes())
    G.nodes[infected]['person'].state = "I"
    G.nodes[infected]['person'].color = "red"
    G.nodes[infected]['person'].I = 0.99
    G.nodes[infected]['person'].S = 0.01
    global position
    position = nx.spring_layout(G) #tracks position of nodes
    return G

def simulateInfection(N, kAvg, gamma, beta, R0, t, I, R, G):
    clock = 1
    #while clock < t: #start clock
    for i in G.nodes(): #scan through every node at current time
        Si = 0
        Ri = 0
        test = 0
        CurIS = G.nodes[i]['person'].S
        CurII = G.nodes[i]['person'].I
        CurIR = G.nodes[i]['person'].R
        print("CurIS: ", CurIS)
        print("CurII: ", CurII)
        print("CurIR: ", CurIR)

        for j in G.nodes(): 
            CurJI = G.nodes[j]['person'].I

            Aij = 0
            if G.has_edge(i, j):
                Aij = 1
            

            Si += ( Aij * CurJI * CurIS)
        G.nodes[i]['person'].S = -(beta/N) * CurIS * CurII
        Ri = gamma * CurII
        G.nodes[i]['person'].R = Ri
        G.nodes[i]['person'].I = CurII + (beta/N) * Si - gamma * CurII

        print("Si: ", Si)
        print("Ri: ", Ri)
        print("S: ", G.nodes[i]['person'].S)
        print("I: ", G.nodes[i]['person'].I)
        print("R: ", G.nodes[i]['person'].R)

        print("Total: ", G.nodes[i]['person'].S + G.nodes[i]['person'].I + G.nodes[i]['person'].R)
        print("")



        if G.nodes[i]['person'].I > G.nodes[i]['person'].S:
            G.nodes[i]['person'].color = "red"
        elif G.nodes[i]['person'].S >  G.nodes[i]['person'].I:
            G.nodes[i]['person'].color = "blue"
        elif G.nodes[i]['person'].R >  G.nodes[i]['person'].I:
            G.nodes[i]['person'].color = "green"
        elif G.nodes[i]['person'].R >  G.nodes[i]['person'].S:
            G.nodes[i]['person'].color = "green"
    

        node_colors = [G.nodes[i]['person'].color for i in G.nodes()]
        plt.figure(figsize=(20, 20))
        nx.draw(G, position, node_color=node_colors, with_labels=True, node_size=1000)
        plt.savefig("images/" + str(i) + ".png")
        clock += 1

#sucetible = blue, infected = red, recovered = green
def main():
    kAvg = 8 #average degree REDUDANT?
    N = 50 #population
    gamma = 0.2 #recovery rate STANDARD 0.93
    beta = 0.89 #infection rate
    R0 = 3 #reproductive number INTERCHANGEABLE WITH BETA
    t = 10 #time
    I = 1 #infected initial
    R = 0 #recovered
    #S0 = N - N/ - R #susceptible population
    G = createGraph(N, kAvg, 0)

    # tspread = np.linspace(0, t, t+1)
    # print(tspread)

    # node_colors = [G.nodes[i]['person'].color for i in G.nodes()]
    # plt.figure(figsize=(20, 20))
    # nx.draw(G, position, node_color=node_colors, with_labels=True, node_size=1000)
    simulateInfection(N, kAvg, gamma, beta, R0, t, I, R, G)

    #plot G neighbors
    # for i in G.node

if __name__ == "__main__":
    main()



#Si = -beta * S * I * I/N
#Ii = beta * S * I * I/N - gama * I
#Ri = gama * I

#recovery rate is 98.2% GAMMA
#reproductive number R0 is 3 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7751056/#:~:text=R0%20of%20COVID%2D19,-R0%20of&text=They%20found%20a%20final%20mean,of%20modeling%2C%20and%20estimation%20procedures.
#infection rate Beta(B) is  0.89 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10126528/





# A mathematical model for reproduction number and herd immunity

# It = (R0) t/SI
# It—the number of cases at the time
# R0—reproduction number.
# SI—serial interval—the time between the onset of the primary and secondary case (as per previous studies done in Hong Kong, SI has taken as 4.4).5
# t—prediction time.