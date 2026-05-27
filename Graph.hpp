#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "utility.hpp"
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <queue>
#include <cmath>
#include <set>

// NodeInfo is the class which describes the qualities of a Node.
// The graph contains a collection of NodeInfo
struct NodeInfo {
    public:
        NodeInfo();
        NodeInfo(std::string activationFunction, double value, double bias);

        bool operator==(const NodeInfo& other);
        friend std::ostream& operator<<(std::ostream& out, const NodeInfo& n);

        // evaluate the activation function at the preActivation node value and store it in postActivationValue
        double activate();
        // evaluate the activation function's derivative at the preActivation node value.
        double derive();

        // Function pointers are used so that the type of function used may be dynamic. 
        // The Activation Function of this Node. Either identity, ReLU, or sigmoid
        FuncSig activationFunction;
        // The Derivative of the Activation Function of this node.
        FuncSig activationDerivative;
        
        // The value of this Node before the activation function is applied.
        // This is the weighted sum of inputs plus the bias: z = (w1*x1 + w2*x2 + ... + bias)
        // We need to store this separately because the activation derivative during backprop
        // is evaluated at this value: derive() uses preActivationValue, not postActivationValue.
        double preActivationValue;

        // The value of this Node after the activation function is applied: a = activate(z)
        // This is the value that gets passed forward to the next layer as input.
        // Both values must be stored because the forward pass needs postActivationValue
        // and the backward pass needs preActivationValue.
        double postActivationValue;

        // The bias of this Node. Added to the weighted sum before activation.
        double bias;

        // The accumulated gradient for this node's bias.
        // During backpropagation, each training example contributes to this value.
        // It accumulates (adds up) across a batch of examples and is only applied
        // to the bias when update() is called: bias = bias - (learningRate * delta)
        // After the update, delta is reset to zero for the next batch.
        double delta;
};

// The Connection Class is responsible for describing connections between Nodes in the graph. 
// node refers to the node at the receiving end of the directed edge, 
// and weight refers to the weight of the connection. 
struct Connection {

    public:
        Connection();
        Connection(int source, int dest, double weight);

        bool operator<(const Connection& other);
        bool operator==(const Connection& other);
        friend std::ostream& operator<<(std::ostream& out, const Connection& c);

        // The emitting end of the connection.
        int source;
        // Node at the receiving end of the directed edge.
        int dest;
        // Weight between the source of the directed edge and destination node.
        double weight;

        // The accumulated gradient for this connection's weight.
        // Analogous to delta in NodeInfo, but for the weight rather than the bias.
        // Accumulates across a batch and is applied in update(): weight = weight - (learningRate * delta)
        // Reset to zero after each update.
        double delta;

};

// AdjList becomes an alias for std::vector<std::unordered_map<int, Connection> >
// * Anywhere you see AdjList, you can interpret it as std::vector<std::unordered_map<int, Connection> > *
typedef std::vector<std::unordered_map<int, Connection> > AdjList;


// Graph is the generic graph structure you learned about in class. It contains a definitive list of nodes, indexed by their id.
// The adjacency list maps a node's id to a Connection object. 
// We store the addresses of the nodes and put the nodes on the heap so that the information stored in the node may be arbitrarily large
// But the only thing on the stack is the id of each node. 
class Graph {

    public:
        Graph();
        Graph(int size);
        Graph(const Graph& other);
        Graph& operator=(const Graph& other);
        ~Graph();


        // TODO: graph methods
        void updateNode(int id, NodeInfo n);
        NodeInfo* getNode(int id) const;
        void updateConnection(int v, int u, double w);

        

        AdjList& getAdjacencyList();

        friend std::ostream& operator<<(std::ostream& out, const Graph& g);
        void resize(int size);

    protected:
        // protected to give NeuralNetwork access

        // adjacency list containing weights for edges.
        AdjList adjacencyList;

        // nodes serves as the canonical (the most trustworthy) version of a node in the graph. Do not rely on anything else 
        // to be an accurate representation of this node. Use node id to index into this vector. 
        std::vector<NodeInfo*> nodes;


        // maintains number of nodes just for bookkeeping purposes
        int size;

        std::vector<NodeInfo*> getNodes() const;

        // TODO: graph methods
        void clear();

        
};

#endif