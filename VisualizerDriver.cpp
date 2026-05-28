#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "NeuralNetwork.hpp"
#include "DataLoader.hpp"
#include "Trace.hpp"
#include "utility.hpp"

using namespace std;

namespace {

std::string escapeJsonString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    return out;
}

std::string vectorToJsonArray(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

std::string labelToJson(double y) {
    std::ostringstream oss;
    oss << y;
    return oss.str();
}

double binaryCrossEntropy(double y, double p) {
    const double eps = 1e-9;
    p = std::max(std::min(p, 1.0 - eps), eps);  // clamp for numerical stability
    return -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
}

void emitInitialGraph(NeuralNetwork& nn) {
    const std::vector<std::vector<int> >& layers = nn.getLayers();
    std::ostringstream nodesJson;
    nodesJson << "[";
    bool firstNode = true;
    for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
        for (int nodeId : layers[layerIdx]) {
            NodeInfo* info = nn.getNode(nodeId);
            if (info) {
                if (!firstNode) nodesJson << ",";
                firstNode = false;
                std::string act = getActivationIdentifier(info->activationFunction);
                nodesJson << "{\"id\":" << nodeId << ",\"layer\":" << layerIdx
                         << ",\"activation\":\"" << escapeJsonString(act) << "\"}";
            }
        }
    }
    nodesJson << "]";

    AdjList& adj = nn.getAdjacencyList();
    std::ostringstream edgesJson;
    edgesJson << "[";
    bool firstEdge = true;
    for (size_t src = 0; src < adj.size(); src++) {
        for (auto& kv : adj[src]) {
            const Connection& c = kv.second;
            if (!firstEdge) edgesJson << ",";
            firstEdge = false;
            edgesJson << "{\"source\":" << src << ",\"dest\":" << c.dest
                     << ",\"initialWeight\":" << c.weight << "}";
        }
    }
    edgesJson << "]";
    viz::traceInitialGraph(nodesJson.str(), edgesJson.str());
}

void emitUpdateStepWithGraph(NeuralNetwork& nn, int step, double batchSize) {
    std::ostringstream nodesJson;
    nodesJson << "[";
    bool firstNode = true;
    const std::vector<std::vector<int> >& layers = nn.getLayers();
    for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
        for (int nodeId : layers[layerIdx]) {
            NodeInfo* info = nn.getNode(nodeId);
            if (info) {
                if (!firstNode) nodesJson << ",";
                firstNode = false;
                nodesJson << "{\"id\":" << nodeId
                         << ",\"pre\":" << info->preActivationValue
                         << ",\"post\":" << info->postActivationValue
                         << ",\"bias\":" << info->bias
                         << ",\"delta\":" << info->delta << "}";
            }
        }
    }
    nodesJson << "]";

    std::ostringstream edgesJson;
    edgesJson << "[";
    bool firstEdge = true;
    AdjList& adj = nn.getAdjacencyList();
    for (size_t src = 0; src < adj.size(); src++) {
        for (auto& kv : adj[src]) {
            const Connection& c = kv.second;
            if (!firstEdge) edgesJson << ",";
            firstEdge = false;
            edgesJson << "{\"source\":" << src << ",\"dest\":" << c.dest
                     << ",\"weight\":" << c.weight << ",\"delta\":" << c.delta << "}";
        }
    }
    edgesJson << "]";

    viz::traceUpdateStepWithGraph(step, "update", batchSize, nodesJson.str(), edgesJson.str());
}

} // namespace

int main(int argc, char* argv[]) {
    const string networkFile = "./models/sample.init";
    const string dataFile = "./data/sample.csv";
    const string traceDir = "./web-viz/";

    // Trace only the first N input rows (e.g. 1 for a single-instance trace).
    const size_t numTraceRows = 1;

    DataLoader dl(dataFile);
    const size_t numInstances = dl.getData().size();
    if (numInstances == 0) {
        cerr << "No data in " << dataFile << endl;
        return 1;
    }
    const size_t n = min(numTraceRows, numInstances);

    const int numEpochs = 1;
    const double lr = 0.001;

    // --- Eval trace ---
    {
        viz::initTrace(traceDir + "eval.trace");
        viz::enableTracing(true);
        NeuralNetwork nn(networkFile);
        nn.setLearningRate(lr);
        viz::traceRunStart(networkFile, dataFile, dataFile, lr);
        emitInitialGraph(nn);
        nn.eval();
        int globalStep = 0;
        for (size_t j = 0; j < n; j++) {
            DataInstance di = dl.getData().at(j);
            viz::traceStepStart(globalStep, "forward",
                                vectorToJsonArray(di.x), labelToJson(di.y));
            vector<double> output = nn.predict(di);
            if (!output.empty()) {
                viz::traceLoss(globalStep, "instance", binaryCrossEntropy(di.y, output.at(0)));
            }
            globalStep++;
        }
        viz::enableTracing(false);
        double evalAccuracy = nn.assess(dataFile);
        viz::enableTracing(true);
        viz::traceRunEnd(0, evalAccuracy);
        viz::closeTrace();
        cout << "Trace written to " << traceDir << "eval.trace" << endl;
    }

    // --- Train trace ---
    {
        viz::initTrace(traceDir + "train.trace");
        viz::enableTracing(true);
        NeuralNetwork nn(networkFile);
        nn.setLearningRate(lr);
        viz::traceRunStart(networkFile, dataFile, dataFile, lr);
        emitInitialGraph(nn);
        nn.train();
        int globalStep = 0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double epochLoss = 0.0;
            int count = 0;
            for (size_t j = 0; j < n; j++) {
                DataInstance di = dl.getData().at(j);
                viz::traceStepStart(globalStep, "forward",
                                    vectorToJsonArray(di.x), labelToJson(di.y));
                vector<double> output = nn.predict(di);
                if (!output.empty()) {
                    double p = output.at(0);
                    epochLoss += binaryCrossEntropy(di.y, p);
                    count++;
                    viz::traceLoss(globalStep, "instance", binaryCrossEntropy(di.y, p));
                }
                globalStep++;
            }
            nn.update();
            if (count > 0) {
                viz::traceLoss(epoch, "epoch", epochLoss / count);
            }
            emitUpdateStepWithGraph(nn, epoch, static_cast<double>(count));
        }
        viz::enableTracing(false);
        double trainAccuracy = nn.assess(dataFile);
        viz::enableTracing(true);
        viz::traceRunEnd(numEpochs, trainAccuracy);
        viz::closeTrace();
        cout << "Trace written to " << traceDir << "train.trace" << endl;
    }

    return 0;
}

