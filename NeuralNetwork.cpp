// includes
#include <queue>
#include <unordered_set>
#include "NeuralNetwork.hpp"
#include "Trace.hpp"

using namespace std;



// NeuralNetwork -----------------------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::eval() {
    evaluating = true;
}

void NeuralNetwork::train() {
    evaluating = false;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

void NeuralNetwork::setInputNodeIds(std::vector<int> inputNodeIds) {
    this->inputNodeIds = inputNodeIds;
}

void NeuralNetwork::setOutputNodeIds(std::vector<int> outputNodeIds) {
    this->outputNodeIds = outputNodeIds;
}

vector<int> NeuralNetwork::getInputNodeIds() const {
    return inputNodeIds;
}

vector<int> NeuralNetwork::getOutputNodeIds() const {
    return outputNodeIds;
}

vector<double> NeuralNetwork::predict(DataInstance instance) {

    vector<double> input = instance.x;

    // error checking : size mismatch
    if (input.size() != inputNodeIds.size()) {
        cerr << "input size mismatch." << endl;
        cerr << "\tNeuralNet expected input size: " << inputNodeIds.size() << endl;
        cerr << "\tBut got: " << input.size() << endl;
        return vector<double>();
    }

    // Load input values directly into input nodes — no activation applied
    for (int i = 0; i < (int)inputNodeIds.size(); i++) {
        nodes[inputNodeIds[i]]->postActivationValue = input[i];
    }

    // BFT — start from input nodes' neighbors, not the input nodes themselves
    queue<int> q;
    unordered_set<int> visited;

    for (int id : inputNodeIds) {
        visited.insert(id);
        for (auto& [destId, conn] : adjacencyList[id]) {
            visitPredictNeighbor(conn);
            if (visited.find(destId) == visited.end()) {
                visited.insert(destId);
                q.push(destId);
            }
        }
    }

    while (!q.empty()) {
        int vId = q.front();
        q.pop();

        visitPredictNode(vId);

        for (auto& [destId, conn] : adjacencyList[vId]) {
            visitPredictNeighbor(conn);
            if (visited.find(destId) == visited.end()) {
                visited.insert(destId);
                q.push(destId);
            }
        }
    }

    vector<double> output;
    for (int i = 0; i < (int)outputNodeIds.size(); i++) {
        int dest = outputNodeIds.at(i);
        NodeInfo* outputNode = nodes.at(dest);
        output.push_back(outputNode->postActivationValue);
    }

    if (evaluating) {
        flush();
    } else {
        batchSize++;
        contribute(instance.y, output.at(0));
        // clear node values and contributions map for next predict,
        // but leave batchSize and deltas intact for update()
        for (int i = 0; i < (int)nodes.size(); i++) {
            nodes.at(i)->postActivationValue = 0;
            nodes.at(i)->preActivationValue = 0;
        }
        contributions.clear();
    }
    return output;
}

bool NeuralNetwork::contribute(double y, double p) {
    for (int id : inputNodeIds) {
        contribute(id, y, p);
    }
    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    visitContributeStart(nodeId); // don't remove this line, used for visualization

    double incomingContribution = 0;
    double outgoingContribution = 0;

    // If already computed, return stored value (handles multiple paths to same node)
    if (contributions.find(nodeId) != contributions.end()) {
        return contributions.at(nodeId);
    }

    if (adjacencyList.at(nodeId).empty()) {
        // Base case: output node — seed the backward pass with the initial error signal.
        // Clamp p to avoid division by zero when p is exactly 0 or 1.
        const double eps = 1e-9;
        double p_clamped = max(eps, min(1.0 - eps, p));
        outgoingContribution = -1 * ((y - p_clamped) / (p_clamped * (1.0 - p_clamped)));
    } else {
        // Recursive case: recurse into neighbors first, then visit this node
        for (auto& [destId, conn] : adjacencyList.at(nodeId)) {
            incomingContribution = contribute(destId, y, p);
            visitContributeNeighbor(conn, incomingContribution, outgoingContribution);
        }
        // Input nodes do not have a bias to update
        bool isInputNode = false;
        for (int id : inputNodeIds) {
            if (id == nodeId) { isInputNode = true; break; }
        }
        if (!isInputNode) {
            visitContributeNode(nodeId, outgoingContribution);
        }
    }

    contributions[nodeId] = outgoingContribution;
    return outgoingContribution;
}

bool NeuralNetwork::update() {
    for (int i = 0; i < (int)nodes.size(); i++) {
        nodes[i]->bias -= learningRate * (nodes[i]->delta / batchSize);
        nodes[i]->delta = 0;

        for (auto& [destId, conn] : adjacencyList[i]) {
            conn.weight -= learningRate * (conn.delta / batchSize);
            conn.delta = 0;
        }
    }

    flush();
    return true;
}




// Feel free to explore the remaining code, but no need to implement past this point

// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------


// Constructors
NeuralNetwork::NeuralNetwork() : Graph(0) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(string filename) : Graph() {
    ifstream fin(filename);
    if (fin.fail()) {
        cerr << "Could not open " << filename << " for reading. " << endl;
        exit(1);
    }
    loadNetwork(fin);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
    fin.close();
}

NeuralNetwork::NeuralNetwork(istream& in) : Graph() {
    loadNetwork(in);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

const vector<vector<int> >& NeuralNetwork::getLayers() const {
    return layers;
}

void NeuralNetwork::loadNetwork(istream& in) {
    int numLayers(0), totalNodes(0), numNodes(0), weightModifications(0), biasModifications(0);
    string activationMethod = "identity";
    string junk;
    in >> numLayers; in >> totalNodes; getline(in, junk);
    if (numLayers <= 1) {
        cerr << "Neural Network must have at least 2 layers, but got " << numLayers << " layers" << endl;
        exit(1);
    }

    resize(totalNodes);
    this->size = totalNodes;

    int currentNodeId(0);
    vector<int> previousLayer;
    vector<int> currentLayer;

    for (int i = 0; i < numLayers; i++) {
        currentLayer.clear();
        in >> numNodes; in >> activationMethod; getline(in, junk);

        for (int j = 0; j < numNodes; j++) {
            updateNode(currentNodeId, NodeInfo(activationMethod, 0, 0));
            currentLayer.push_back(currentNodeId++);
        }

        if (i != 0) {
            for (int k = 0; k < previousLayer.size(); k++) {
                for (int w = 0; w < currentLayer.size(); w++) {
                    updateConnection(previousLayer.at(k), currentLayer.at(w), sample());
                }
            }
        }

        previousLayer = currentLayer;
        layers.push_back(currentLayer);
    }

    in >> weightModifications; getline(in, junk);
    int v(0), u(0); double w(0), b(0);

    for (int i = 0; i < weightModifications; i++) {
        in >> v; in >> u; in >> w; getline(in, junk);
        updateConnection(v, u, w);
    }

    in >> biasModifications; getline(in, junk);

    for (int i = 0; i < biasModifications; i++) {
        in >> v; in >> b; getline(in, junk);
        NodeInfo* thisNode = getNode(v);
        thisNode->bias = b;
    }

    setInputNodeIds(layers.at(0));
    setOutputNodeIds(layers.at(layers.size()-1));
}

void NeuralNetwork::visitPredictNode(int vId) {
    NodeInfo* v = nodes.at(vId);
    v->preActivationValue += v->bias;
    v->activate();
    if (viz::isTracing()) {
        viz::traceNodeState(0, "forward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "current");
    }
}

void NeuralNetwork::visitPredictNeighbor(Connection c) {
    NodeInfo* v = nodes.at(c.source);
    NodeInfo* u = nodes.at(c.dest);
    double w = c.weight;
    u->preActivationValue += v->postActivationValue * w;
    if (viz::isTracing()) {
        viz::traceEdgeState(0, "forward", c.source, c.dest, c.weight, c.delta);
        viz::traceNodeState(0, "forward", c.dest,
                            u->preActivationValue,
                            u->postActivationValue,
                            u->bias,
                            u->delta,
                            "neighbor");
    }
}

void NeuralNetwork::visitContributeStart(int vId) {
    NodeInfo* v = nodes.at(vId);
    if (viz::isTracing()) {
        viz::traceNodeState(0, "backward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "stack");
    }
}

void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    NodeInfo* v = nodes.at(vId);
    outgoingContribution *= v->derive();
    v->delta += outgoingContribution;
    if (viz::isTracing()) {
        viz::traceNodeState(0, "backward", vId,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "current");
    }
}

void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    NodeInfo* v = nodes.at(c.source);
    outgoingContribution += c.weight * incomingContribution;
    c.delta += incomingContribution * v->postActivationValue;
    if (viz::isTracing()) {
        viz::traceEdgeState(0, "backward", c.source, c.dest, c.weight, c.delta);
        viz::traceNodeState(0, "backward", c.source,
                            v->preActivationValue,
                            v->postActivationValue,
                            v->bias,
                            v->delta,
                            "neighbor");
    }
}

void NeuralNetwork::flush() {
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->postActivationValue = 0;
        nodes.at(i)->preActivationValue = 0;
    }
    contributions.clear();
    batchSize = 0;
}

double NeuralNetwork::assess(string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

double NeuralNetwork::assess(DataLoader dl) {
    bool stateBefore = evaluating;
    evaluating = true;
    double count(0);
    double correct(0);
    vector<double> output;
    for (int i = 0; i < dl.getData().size(); i++) {
        DataInstance di = dl.getData().at(i);
        output = predict(di);
        if (static_cast<int>(round(output.at(0))) == di.y) {
            correct++;
        }
        count++;
    }
    if (dl.getData().empty()) {
        cerr << "Cannot assess accuracy on an empty dataset" << endl;
        exit(1);
    }
    evaluating = stateBefore;
    return correct / count;
}

void NeuralNetwork::saveModel(string filename) {
    ofstream fout(filename);
    fout << layers.size() << " " << getNodes().size() << endl;
    for (int i = 0; i < layers.size(); i++) {
        NodeInfo* layerNode = getNodes().at(layers.at(i).at(0));
        string activationType = getActivationIdentifier(layerNode->activationFunction);
        fout << layers.at(i).size() << " " << activationType << endl;
    }

    int numWeights = 0;
    int numBias = 0;
    stringstream weightStream;
    stringstream biasStream;
    for (int i = 0; i < nodes.size(); i++) {
        numBias++;
        biasStream << i << " " << nodes.at(i)->bias << endl;
        for (auto j = adjacencyList.at(i).begin(); j != adjacencyList.at(i).end(); j++) {
            numWeights++;
            weightStream << j->second.source << " " << j->second.dest << " " << j->second.weight << endl;
        }
    }

    fout << numWeights << endl;
    fout << weightStream.str();
    fout << numBias << endl;
    fout << biasStream.str();
    fout.close();
}

ostream& operator<<(ostream& out, const NeuralNetwork& nn) {
    for (int i = 0; i < nn.layers.size(); i++) {
        out << "layer " << i << ": ";
        for (int j = 0; j < nn.layers.at(i).size(); j++) {
            out << nn.layers.at(i).at(j) << " ";
        }
        out << endl;
    }
    out << static_cast<const Graph&>(nn) << endl;
    return out;
}