from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class Graph:

    def __init__(self, n: int, edges: list[list[int]]):
        #setting size of the list based on how many nodes we have
        self.vertex_size = [0] * n
        
        #creating nested dictionary which works like this: {"node": {
        #                                                            "neighbour node" : "cost of traversal to this node",
        #                                                            "neighbour node" : "cost of traversal to this node",
        #                                                            ...
        #                                                            }
        #                                                   "node": {...}
        #                                                  }
        self.graph_edges = defaultdict(dict)
        for x in edges:
            self.graph_edges[x[0]][x[1]] = x[2]

    #method that handles adding new paths between nodes
    def addEdge(self, edge: list[int]) -> None:
        self.graph_edges[edge[0]][edge[1]] = edge[2]
        
    #visualisation method
    def visualize(self):
        G = nx.DiGraph()
        for node, neighbors in self.graph_edges.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)
    
        pos = nx.circular_layout(G, scale=1.5)
    
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, font_color='black')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()
        
    #method that handels Dijkstra's algorithm and returns smallest cost possible to get from node1 to node2
    def shortestPath(self, node1: int, node2: int) -> int:
        
        #in vertex list, indexes are nodes and values of each node are costs of traversing to it
        vertex = self.vertex_size
        #starting node
        curr = node1
        #prev_cost is added up cost of all distances we traversed. We haven't traversed any so its set to 0
        prev_cost = 0
        visited = set()
        print(vertex)
        while True:
            #if node we wanted to get to is reached we return value (calculated smallest cost) of index (node)
            
            if curr == node2:
                return vertex[curr]
            
            #saving current node to visited 
            visited.add(curr)
            #calculating cost of traversing from current node to corresponding nodes
            #starting for loop based on current node in nested dictionary
            for corresponding, cost in self.graph_edges[curr].items():
                
                #if corresponding nodes were visited we skip
                if corresponding not in visited:
                    
                    #if saved cost of corresponding node is higher than cost from current node + sum of previous distances cost we swap the values 
                    if vertex[corresponding] > cost + prev_cost:
                        vertex[corresponding] = cost + prev_cost
                    #else we add the cost of traversing to corresponding nodes with addition of sum of previous distances cost
                    else:
                        vertex[corresponding] += cost + prev_cost
                        
            #moving on to the next node which cost of traversing is the smallest and adding up cost to prev_cost
            #added error handling because if error occured it means that there is no such path and it returns -1
            try:
                next_node = min(self.graph_edges[curr], key=self.graph_edges[curr].get)
                if next_node in visited:
                    #break the loop if a cycle is detected
                    break
                prev_cost += self.graph_edges[curr][min(self.graph_edges[curr], key=self.graph_edges[curr].get)]
                curr = next_node
            except:
                return -1
            

        

                
        

obj = Graph(4,[[0,2,5],[0,1,2],[1,2,1],[3,0,3]])
obj2 = Graph(5, [[0,1,10],[0,2,5],[1,3,1],[1,2,2],[2,1,3],[2,4,2],[2,3,9],[3,4,4],[4,3,6],[4,0,7]])

print("Shortest path from 3 to 2 is: " + str(obj.shortestPath(3,2)))
obj.visualize()

print("Shortest path from 1 to 0 is: " + str(obj2.shortestPath(2,3)))
obj2.visualize()


# Variables:
# edge = ["from which node you are making connection", "to what node you are making connection", "weight of the connection"]
# edges = [edge, edge, edge, ...]
# n = "number of nodes you want created"
#
# Your Graph object need to be instantiated and called as such:
# obj = Graph(n, edges)
# obj.addEdge(edge)
# obj.shortestPath(node1,node2)