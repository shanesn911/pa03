#ifndef TRACE_HPP
#define TRACE_HPP

#include <string>

// Simple tracing API for pa03 visualization.
// Writes line-delimited JSON events as specified in VIZ_FORMAT.md.
//
// Usage pattern:
//   viz::initTrace("out.trace");
//   viz::traceRunStart(...);
//   ... emit more trace* events ...
//   viz::closeTrace();
//
// All trace functions are no-ops if tracing is not enabled.

namespace viz {

// Initialize tracing to a given filename. Opens (overwrites) the file.
void initTrace(const std::string& filename);

// Close the trace file (safe to call multiple times).
void closeTrace();

// Enable or disable tracing at runtime.
void enableTracing(bool enabled);

// Returns true if tracing is currently enabled and the stream is good.
bool isTracing();

// --- Event helpers (mirror VIZ_FORMAT.md) ---

void traceRunStart(const std::string& networkFile,
                   const std::string& trainFile,
                   const std::string& testFile,
                   double learningRate);

void traceRunEnd(int epochs, double finalAccuracy);

// Emit the full initial graph as one object (nodes and edges as JSON arrays).
void traceInitialGraph(const std::string& nodesJson,
                       const std::string& edgesJson);

void traceStepStart(int step,
                    const std::string& phase,
                    const std::string& inputJsonArray,
                    const std::string& labelJson);

// highlight: "current" or "neighbor" (optional; omit or empty for backward compatibility).
void traceNodeState(int step,
                    const std::string& phase,
                    int nodeId,
                    double pre,
                    double post,
                    double bias,
                    double delta,
                    const std::string& highlight = "");

void traceEdgeState(int step,
                    const std::string& phase,
                    int source,
                    int dest,
                    double weight,
                    double delta);

void traceLoss(int step,
               const std::string& scope,
               double value);

// Emit update_step with full graph state (nodes and edges arrays).
void traceUpdateStepWithGraph(int step,
                              const std::string& phase,
                              double batchSize,
                              const std::string& nodesJson,
                              const std::string& edgesJson);

} // namespace viz

#endif

