#include "Trace.hpp"

#include <fstream>
#include <iomanip>

namespace viz {

namespace {

struct TraceState {
    std::ofstream out;
    bool enabled = false;
};

TraceState& state() {
    static TraceState s;
    return s;
}

bool ready() {
    TraceState& s = state();
    return s.enabled && s.out.is_open() && s.out.good();
}

// Very lightweight string escaping for JSON strings.
std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '\"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    return out;
}

} // namespace

void initTrace(const std::string& filename) {
    TraceState& s = state();
    if (s.out.is_open()) {
        s.out.close();
    }
    s.out.open(filename, std::ios::out | std::ios::trunc);
    s.out << std::setprecision(10);
    s.enabled = s.out.good();
}

void closeTrace() {
    TraceState& s = state();
    if (s.out.is_open()) {
        s.out.flush();
        s.out.close();
    }
    s.enabled = false;
}

void enableTracing(bool enabled) {
    state().enabled = enabled && state().out.is_open() && state().out.good();
}

bool isTracing() {
    return ready();
}

void traceRunStart(const std::string& networkFile,
                   const std::string& trainFile,
                   const std::string& testFile,
                   double learningRate) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"run_start\","
          << "\"meta\":{"
          << "\"networkFile\":\"" << escape(networkFile) << "\","
          << "\"trainFile\":\"" << escape(trainFile) << "\","
          << "\"testFile\":\"" << escape(testFile) << "\","
          << "\"learningRate\":" << learningRate
          << "}"
          << "}"
          << std::endl;
}

void traceRunEnd(int epochs, double finalAccuracy) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"run_end\","
          << "\"summary\":{"
          << "\"epochs\":" << epochs << ","
          << "\"finalAccuracy\":" << finalAccuracy
          << "}"
          << "}"
          << std::endl;
}

void traceInitialGraph(const std::string& nodesJson,
                       const std::string& edgesJson) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{\"type\":\"initial_graph\",\"nodes\":" << nodesJson
          << ",\"edges\":" << edgesJson << "}"
          << std::endl;
}

void traceStepStart(int step,
                    const std::string& phase,
                    const std::string& inputJsonArray,
                    const std::string& labelJson) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"step_start\","
          << "\"step\":" << step << ","
          << "\"phase\":\"" << escape(phase) << "\","
          << "\"input\":" << inputJsonArray << ","
          << "\"label\":" << labelJson
          << "}"
          << std::endl;
}

void traceNodeState(int step,
                    const std::string& phase,
                    int nodeId,
                    double pre,
                    double post,
                    double bias,
                    double delta,
                    const std::string& highlight) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"node_state\","
          << "\"step\":" << step << ","
          << "\"phase\":\"" << escape(phase) << "\","
          << "\"id\":" << nodeId << ","
          << "\"pre\":" << pre << ","
          << "\"post\":" << post << ","
          << "\"bias\":" << bias << ","
          << "\"delta\":" << delta;
    if (!highlight.empty()) {
        s.out << ",\"highlight\":\"" << escape(highlight) << "\"";
    }
    s.out << "}"
          << std::endl;
}

void traceEdgeState(int step,
                    const std::string& phase,
                    int source,
                    int dest,
                    double weight,
                    double delta) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"edge_state\","
          << "\"step\":" << step << ","
          << "\"phase\":\"" << escape(phase) << "\","
          << "\"source\":" << source << ","
          << "\"dest\":" << dest << ","
          << "\"weight\":" << weight << ","
          << "\"delta\":" << delta
          << "}"
          << std::endl;
}

void traceLoss(int step,
               const std::string& scope,
               double value) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"loss\","
          << "\"step\":" << step << ","
          << "\"scope\":\"" << escape(scope) << "\","
          << "\"value\":" << value
          << "}"
          << std::endl;
}

void traceUpdateStepWithGraph(int step,
                              const std::string& phase,
                              double batchSize,
                              const std::string& nodesJson,
                              const std::string& edgesJson) {
    if (!ready()) return;
    TraceState& s = state();
    s.out << "{"
          << "\"type\":\"update_step\","
          << "\"step\":" << step << ","
          << "\"phase\":\"" << escape(phase) << "\","
          << "\"batchSize\":" << batchSize << ","
          << "\"nodes\":" << nodesJson << ","
          << "\"edges\":" << edgesJson
          << "}"
          << std::endl;
}

} // namespace viz

