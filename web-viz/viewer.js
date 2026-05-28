(function () {
  const TRACE_FILES = {
    eval: { default: "default_eval.trace", user: "eval.trace" },
    train: { default: "default_train.trace", user: "train.trace" }
  };

  const INITIAL_STATE_INDEX = 1;

  const statusEl = document.getElementById("status");
  const btnEval = document.getElementById("btnEval");
  const btnTrain = document.getElementById("btnTrain");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const resetBtn = document.getElementById("resetBtn");
  const eventCounter = document.getElementById("eventCounter");
  const phaseIndicatorLeft = document.getElementById("phaseIndicatorLeft");
  const phaseIndicatorRight = document.getElementById("phaseIndicatorRight");
  const graphLeft = document.getElementById("graphLeft");
  const graphRight = document.getElementById("graphRight");

  const REFERENCE_WIDTH = 900;
  const LAYOUT_DEFAULTS = {
    LAYER_WIDTH: 210,
    LAYER_PADDING: 60,
    LAYER_PADDING_LEFT: 20,
    LAYER_PADDING_TOP: -40,
    NODE_R: 60,
    BOX_RX: 12,
    BOX_TITLE_H: 100
  };

  function getLayoutConstants(containerWidth) {
    if (!containerWidth || containerWidth <= 0) containerWidth = REFERENCE_WIDTH;
    const scale = Math.max(0.4, Math.min(1.2, containerWidth / REFERENCE_WIDTH));
    return {
      LAYER_WIDTH: Math.round(LAYOUT_DEFAULTS.LAYER_WIDTH * scale),
      LAYER_PADDING: Math.round(LAYOUT_DEFAULTS.LAYER_PADDING * scale),
      LAYER_PADDING_LEFT: Math.round(LAYOUT_DEFAULTS.LAYER_PADDING_LEFT * scale),
      LAYER_PADDING_TOP: Math.round(LAYOUT_DEFAULTS.LAYER_PADDING_TOP * scale),
      NODE_R: Math.round(LAYOUT_DEFAULTS.NODE_R * scale),
      BOX_RX: Math.round(LAYOUT_DEFAULTS.BOX_RX * scale),
      BOX_TITLE_H: Math.round(LAYOUT_DEFAULTS.BOX_TITLE_H * scale)
    };
  }

  let mode = "eval";
  let eventsDefault = [];
  let eventsUser = [];
  let currentIndex = INITIAL_STATE_INDEX;

  function replayToState(events, endIndex) {
    const nodes = new Map();
    const edges = [];
    const edgeMap = new Map();
    let numLayers = 0;
    let forwardCurrentNodeList = [];
    let lastNeighborNodeId = null;
    let lastEdgeState = null;
    let backwardStackNodeList = [];
    let backwardCurrentNodeList = [];
    let lastBackwardEdgeState = null;
    let lastPhase = "forward";
    let phaseLabel = "predict";
    const capped = Math.min(endIndex, events.length - 1);

    for (let i = 0; i <= capped && i < events.length; i++) {
      const ev = events[i];
      if (ev.type === "step_start") {
        forwardCurrentNodeList = [];
        lastNeighborNodeId = null;
        lastEdgeState = null;
        backwardStackNodeList = [];
        backwardCurrentNodeList = [];
        lastBackwardEdgeState = null;
        lastPhase = "forward";
        phaseLabel = "predict";
        continue;
      }
      if (ev.type === "update_step") {
        phaseLabel = "update";
        if (ev.nodes && Array.isArray(ev.nodes)) {
          ev.nodes.forEach((nd) => {
            const n = nodes.get(nd.id);
            if (n) {
              if (nd.pre != null) n.pre = nd.pre;
              if (nd.post != null) n.post = nd.post;
              if (nd.bias != null) n.bias = nd.bias;
              if (nd.delta != null) n.delta = nd.delta;
            }
          });
        }
        if (ev.edges && Array.isArray(ev.edges)) {
          ev.edges.forEach((e) => {
            const key = e.source + "," + e.dest;
            const edge = edgeMap.get(key);
            if (edge) {
              if (e.weight != null) edge.weight = e.weight;
              if (e.delta != null) edge.delta = e.delta;
            }
          });
        }
        continue;
      }
      if (lastNeighborNodeId !== null) lastEdgeState = null;
      if (lastNeighborNodeId !== null) {
        lastBackwardEdgeState = null;
      }
      lastNeighborNodeId = null;
      if (ev.type === "initial_graph") {
        if (ev.nodes && Array.isArray(ev.nodes)) {
          ev.nodes.forEach((nd) => {
            nodes.set(nd.id, {
              id: nd.id,
              layer: nd.layer ?? 0,
              activation: nd.activation || "identity",
              pre: 0,
              post: 0,
              bias: 0,
              delta: 0
            });
            numLayers = Math.max(numLayers, (nd.layer ?? 0) + 1);
          });
        }
        if (ev.edges && Array.isArray(ev.edges)) {
          ev.edges.forEach((e) => {
            const key = e.source + "," + e.dest;
            if (!edgeMap.has(key)) {
              const edge = { source: e.source, dest: e.dest, weight: e.initialWeight ?? e.weight ?? 0, delta: 0 };
              edgeMap.set(key, edge);
              edges.push(edge);
            }
          });
        }
      } else if (ev.type === "graph_node") {
        nodes.set(ev.id, {
          id: ev.id,
          layer: ev.layer,
          activation: ev.activation || "identity",
          pre: 0,
          post: 0,
          bias: 0,
          delta: 0
        });
        numLayers = Math.max(numLayers, (ev.layer || 0) + 1);
      } else if (ev.type === "graph_edge") {
        const key = ev.source + "," + ev.dest;
        if (!edgeMap.has(key)) {
          const e = { source: ev.source, dest: ev.dest, weight: ev.initialWeight ?? ev.weight ?? 0, delta: 0 };
          edgeMap.set(key, e);
          edges.push(e);
        }
      } else if (ev.type === "node_state") {
        const n = nodes.get(ev.id);
        if (n) {
          if (ev.pre != null) n.pre = ev.pre;
          if (ev.post != null) n.post = ev.post;
          if (ev.bias != null) n.bias = ev.bias;
          if (ev.delta != null) n.delta = ev.delta;
        }
        const phase = ev.phase || "forward";
        if (phase === "update") {
          phaseLabel = "update";
          
        } else if (phase === "backward") {
          lastPhase = "backward";
          phaseLabel = "contribute";
          const h = ev.highlight || "";
          if (h === "stack") backwardStackNodeList.push(ev.id);
          else if (h === "current") {
            backwardStackNodeList.pop();
            backwardCurrentNodeList.push(ev.id);
          } else if (h === "neighbor") {
            lastNeighborNodeId = ev.id;
          }
        } else {
          lastPhase = "forward";
          phaseLabel = "predict";
          const h = ev.highlight || "current";
          if (h === "current") forwardCurrentNodeList.push(ev.id);
          else if (h === "neighbor") lastNeighborNodeId = ev.id;
        }
      } else if (ev.type === "edge_state") {
        const key = ev.source + "," + ev.dest;
        const e = edgeMap.get(key);
        if (e) {
          if (ev.weight != null) e.weight = ev.weight;
          if (ev.delta != null) e.delta = ev.delta;
        }
        const phase = ev.phase || "forward";
        if (phase === "update") {
          phaseLabel = "update";
        } else if (phase === "backward") {
          lastPhase = "backward";
          phaseLabel = "contribute";
          lastBackwardEdgeState = { source: ev.source, dest: ev.dest };
        } else {
          lastPhase = "forward";
          phaseLabel = "predict";
          lastEdgeState = { source: ev.source, dest: ev.dest };
        }
      }
    }

    if (nodes.size === 0) {
      for (let i = 0; i <= capped && i < events.length; i++) {
        const ev = events[i];
        if (ev.type === "node_state" && ev.id != null) {
          const layer = ev.layer != null ? ev.layer : 0;
          if (!nodes.has(ev.id)) {
            nodes.set(ev.id, {
              id: ev.id,
              layer: layer,
              activation: "identity",
              pre: ev.pre ?? 0,
              post: ev.post ?? 0,
              bias: ev.bias ?? 0,
              delta: ev.delta ?? 0
            });
          } else {
            const n = nodes.get(ev.id);
            if (ev.pre != null) n.pre = ev.pre;
            if (ev.post != null) n.post = ev.post;
            if (ev.bias != null) n.bias = ev.bias;
            if (ev.delta != null) n.delta = ev.delta;
          }
          numLayers = Math.max(numLayers, layer + 1);
        }
      }
    }
    if (numLayers === 0) numLayers = 1;

    const layers = [];
    for (let l = 0; l < numLayers; l++) layers.push([]);
    nodes.forEach((n) => {
      const l = Math.min(n.layer, layers.length - 1);
      layers[l].push(n);
    });
    layers.forEach((arr) => arr.sort((a, b) => a.id - b.id));

    return {
      nodes: Array.from(nodes.values()),
      edges,
      layers,
      numLayers,
      forwardCurrentNodeList,
      lastNeighborNodeId,
      lastEdgeState,
      backwardStackNodeList,
      backwardCurrentNodeList,
      lastBackwardEdgeState,
      lastPhase,
      phaseLabel
    };
  }

  function formatNum(x) {
    if (x == null || Number.isNaN(x)) return "—";
    if (Math.abs(x) < 1e-4 && x !== 0) return x.toExponential(2);
    return Number.isInteger(x) ? String(x) : x.toFixed(4);
  }

  function render(graphEl, state) {
    const {
      layers,
      edges,
      numLayers,
      forwardCurrentNodeList,
      lastNeighborNodeId,
      lastEdgeState,
      backwardStackNodeList,
      backwardCurrentNodeList,
      lastBackwardEdgeState,
      lastPhase
    } = state;
    if (numLayers === 0 || layers.every((l) => l.length === 0)) {
      graphEl.innerHTML = "<text x=\"20\" y=\"30\" fill=\"#7f8c8d\">No graph in trace.</text>";
      return;
    }

    const containerWidth = graphEl.parentElement ? graphEl.parentElement.clientWidth : window.innerWidth;
    const { LAYER_WIDTH, LAYER_PADDING, LAYER_PADDING_LEFT, LAYER_PADDING_TOP, NODE_R, BOX_RX, BOX_TITLE_H } = getLayoutConstants(containerWidth);

    const maxNodes = Math.max(1, ...layers.map((l) => l.length));
    const boxHeight = Math.max(280, maxNodes * (NODE_R * 3.2));
    const totalW = LAYER_PADDING_LEFT + numLayers * (LAYER_WIDTH + LAYER_PADDING) + LAYER_PADDING;
    const totalH = LAYER_PADDING_TOP + BOX_TITLE_H + boxHeight + LAYER_PADDING;

    graphEl.setAttribute("width", totalW);
    graphEl.setAttribute("height", totalH);

    const nodePos = new Map();
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    marker.setAttribute("id", "arrowhead-" + (graphEl.id || "g"));
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("refX", "4.8");
    marker.setAttribute("refY", "3");
    marker.setAttribute("orient", "auto");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", "0 0, 6 3, 0 6");
    poly.setAttribute("fill", "#2c3e50");
    marker.appendChild(poly);
    defs.appendChild(marker);
    graphEl.innerHTML = "";
    graphEl.appendChild(defs);

    layers.forEach((layerNodes, layerIdx) => {
      const isInput = layerIdx === 0;
      const isOutput = layerIdx === numLayers - 1;
      const boxX = LAYER_PADDING_LEFT + layerIdx * (LAYER_WIDTH + LAYER_PADDING);
      const boxY = LAYER_PADDING_TOP + BOX_TITLE_H;
      const layerLabel = isInput ? "Input Layer" : isOutput ? "Output Layer" : "Hidden Layer";
      const activation = layerNodes.length ? (layerNodes[0].activation || "identity") : "identity";
      const className = isInput || isOutput ? "input" : "hidden";
      if (isOutput && layerNodes.length) {
        layerNodes[0].activation = layerNodes[0].activation || "sigmoid";
      }

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", boxX);
      rect.setAttribute("y", boxY);
      rect.setAttribute("width", LAYER_WIDTH);
      rect.setAttribute("height", boxHeight);
      rect.setAttribute("rx", BOX_RX);
      rect.setAttribute("class", "layer-box " + className);
      g.appendChild(rect);

      const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
      title.setAttribute("x", boxX + LAYER_WIDTH / 2);
      title.setAttribute("y", boxY - 20);
      title.setAttribute("text-anchor", "middle");
      title.setAttribute("class", "layer-title");
      title.textContent = layerLabel;
      g.appendChild(title);

      const actText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      actText.setAttribute("x", boxX + LAYER_WIDTH / 2);
      actText.setAttribute("y", boxY - 5);
      actText.setAttribute("text-anchor", "middle");
      actText.setAttribute("class", "layer-activation");
      actText.textContent = "Activation: " + (activation === "identity" ? "Identity" : activation);
      g.appendChild(actText);

      const nodeStep = boxHeight / (layerNodes.length + 1) + 10;
      layerNodes.forEach((n, i) => {
        const cx = boxX + LAYER_WIDTH / 2;
        const cy = boxY + (i + 1) * nodeStep;
        nodePos.set(n.id, { x: cx, y: cy });

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", NODE_R);
        circle.setAttribute("class", "node-circle");
        circle.setAttribute("data-node-id", String(n.id));
        g.appendChild(circle);

        const idLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        idLabel.setAttribute("x", cx);
        idLabel.setAttribute("y", cy - 14);
        idLabel.setAttribute("text-anchor", "middle");
        idLabel.setAttribute("class", "node-label");
        idLabel.textContent = "id: " + n.id;
        g.appendChild(idLabel);

        const preLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        preLabel.setAttribute("x", cx);
        preLabel.setAttribute("y", cy - 2);
        preLabel.setAttribute("text-anchor", "middle");
        preLabel.setAttribute("class", "node-value");
        preLabel.textContent = "pre_val: " + formatNum(n.pre);
        g.appendChild(preLabel);

        const postLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        postLabel.setAttribute("x", cx);
        postLabel.setAttribute("y", cy + 10);
        postLabel.setAttribute("text-anchor", "middle");
        postLabel.setAttribute("class", "node-value");
        postLabel.textContent = "post_val: " + formatNum(n.post);
        g.appendChild(postLabel);

        if (!isInput) {
          const biasLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
          biasLabel.setAttribute("x", cx);
          biasLabel.setAttribute("y", cy + 22);
          biasLabel.setAttribute("text-anchor", "middle");
          biasLabel.setAttribute("class", "node-value");
          biasLabel.textContent = "bias: " + formatNum(n.bias);
          g.appendChild(biasLabel);
          const deltaLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
          deltaLabel.setAttribute("x", cx);
          deltaLabel.setAttribute("y", cy + 34);
          deltaLabel.setAttribute("text-anchor", "middle");
          deltaLabel.setAttribute("class", "node-value");
          deltaLabel.textContent = "delta: " + formatNum(n.delta);
          g.appendChild(deltaLabel);
        }
      });

      graphEl.appendChild(g);
    });

    edges.forEach((e) => {
      const from = nodePos.get(e.source);
      const to = nodePos.get(e.dest);
      if (!from || !to) return;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      const x1 = from.x + ux * NODE_R;
      const y1 = from.y + uy * NODE_R;
      const x2 = to.x - ux * NODE_R;
      const y2 = to.y - uy * NODE_R;
      path.setAttribute("d", "M " + x1 + " " + y1 + " L " + x2 + " " + y2);
      path.setAttribute("class", "edge-line");
      path.setAttribute("data-edge-key", e.source + "," + e.dest);
      path.setAttribute("marker-end", "url(#arrowhead-" + (graphEl.id || "g") + ")");
      graphEl.appendChild(path);
      const wx = (4 * x1 + x2) / 5;
      const wy = (4 * y1 + y2) / 5;
      const wText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      wText.setAttribute("x", wx);
      wText.setAttribute("y", wy - 6);
      wText.setAttribute("text-anchor", "middle");
      wText.setAttribute("class", "edge-weight");
      wText.textContent = "w: " + formatNum(e.weight);
      graphEl.appendChild(wText);
      const delx = (x1 + 3 * x2) / 4;
      const dely = (y1 + 3 * y2) / 4;
      const dText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      dText.setAttribute("x", delx);
      dText.setAttribute("y", dely - 6);
      dText.setAttribute("text-anchor", "middle");
      dText.setAttribute("class", "edge-weight");
      dText.textContent = "delta: " + formatNum(e.delta);
      graphEl.appendChild(dText);
    });

    graphEl.querySelectorAll(".node-circle").forEach((el) =>
      el.classList.remove("node-highlight-current", "node-highlight-neighbor", "node-highlight-contribute-current")
    );
    graphEl.querySelectorAll(".edge-line").forEach((el) => el.classList.remove("edge-highlight", "edge-highlight-backward"));
    if (lastPhase === "backward") {
      backwardStackNodeList.forEach((nodeId) => {
        if (nodeId === lastNeighborNodeId) return;
        const circle = graphEl.querySelector('.node-circle[data-node-id="' + nodeId + '"]');
        if (circle) circle.classList.add("node-highlight-stack");
      });
      if (lastNeighborNodeId !== null) {
        const circle = graphEl.querySelector('.node-circle[data-node-id="' + lastNeighborNodeId + '"]');
        if (circle) circle.classList.add("node-highlight-neighbor");
      }
      backwardCurrentNodeList.forEach((nodeId) => {
        const circle = graphEl.querySelector('.node-circle[data-node-id="' + nodeId + '"]');
        if (circle) circle.classList.add("node-highlight-contribute-current");
      });
      if (lastBackwardEdgeState !== null) {
        const key = lastBackwardEdgeState.source + "," + lastBackwardEdgeState.dest;
        const path = graphEl.querySelector('.edge-line[data-edge-key="' + key + '"]');
        if (path) path.classList.add("edge-highlight-backward");
      }
    } else {
      forwardCurrentNodeList.forEach((nodeId) => {
        const circle = graphEl.querySelector('.node-circle[data-node-id="' + nodeId + '"]');
        if (circle) circle.classList.add("node-highlight-current");
      });
      if (lastNeighborNodeId !== null) {
        const circle = graphEl.querySelector('.node-circle[data-node-id="' + lastNeighborNodeId + '"]');
        if (circle) circle.classList.add("node-highlight-neighbor");
      }
      if (lastEdgeState !== null) {
        const key = lastEdgeState.source + "," + lastEdgeState.dest;
        const path = graphEl.querySelector('.edge-line[data-edge-key="' + key + '"]');
        if (path) path.classList.add("edge-highlight");
      }
    }
  }

  function parseTraceText(text) {
    const events = [];
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (!trimmed) continue;
      try {
        events.push(JSON.parse(trimmed));
      } catch (e) {}
    }
    return events;
  }

  function updateToolbar() {
    const maxLen = Math.max(eventsDefault.length, eventsUser.length);
    eventCounter.textContent = "Event " + currentIndex + " of " + Math.max(0, maxLen - 1);
    prevBtn.disabled = currentIndex <= 1;
    nextBtn.disabled = maxLen === 0 || currentIndex >= maxLen - 1;
  }

  function refreshBothPanels() {
    const stateLeft = replayToState(eventsDefault, currentIndex);
    const stateRight = replayToState(eventsUser, currentIndex);
    render(graphLeft, stateLeft);
    render(graphRight, stateRight);
    const phaseLeft = stateLeft.phaseLabel || "predict";
    const phaseRight = stateRight.phaseLabel || "predict";
    phaseIndicatorLeft.textContent = "Phase: " + phaseLeft.charAt(0).toUpperCase() + phaseLeft.slice(1);
    phaseIndicatorLeft.className = "phaseIndicator " + phaseLeft;
    phaseIndicatorRight.textContent = "Phase: " + phaseRight.charAt(0).toUpperCase() + phaseRight.slice(1);
    phaseIndicatorRight.className = "phaseIndicator " + phaseRight;
    updateToolbar();
  }

  function setMode(newMode) {
    mode = newMode;
    btnEval.classList.toggle("active", mode === "eval");
    btnTrain.classList.toggle("active", mode === "train");
    currentIndex = INITIAL_STATE_INDEX;
    loadTracesForMode();
  }

  function loadTracesForMode() {
    const files = TRACE_FILES[mode];
    statusEl.textContent = "Loading " + mode + " traces…";
    Promise.all([
      fetch(files.default).then((r) => r.ok ? r.text() : Promise.reject(new Error(files.default + " failed"))),
      fetch(files.user).then((r) => r.ok ? r.text() : Promise.reject(new Error(files.user + " failed")))
    ]).then(([textDefault, textUser]) => {
      eventsDefault = parseTraceText(textDefault);
      eventsUser = parseTraceText(textUser);
      currentIndex = INITIAL_STATE_INDEX;
      refreshBothPanels();
      statusEl.textContent = mode === "eval" ? "Evaluation mode. Use Previous / Next / Reset." : "Training mode. Use Previous / Next / Reset.";
    }).catch((err) => {
      statusEl.textContent = "Error: " + (err.message || "Could not load trace files. Run ./pa03_viz first.");
      eventsDefault = [];
      eventsUser = [];
      currentIndex = 0;
      updateToolbar();
      render(graphLeft, { layers: [], edges: [], numLayers: 0 });
      render(graphRight, { layers: [], edges: [], numLayers: 0 });
    });
  }

  btnEval.addEventListener("click", function () { setMode("eval"); });
  btnTrain.addEventListener("click", function () { setMode("train"); });
  prevBtn.addEventListener("click", function () {
    if (currentIndex > 1) {
      currentIndex--;
      refreshBothPanels();
    }
  });
  nextBtn.addEventListener("click", function () {
    const maxLen = Math.max(eventsDefault.length, eventsUser.length);
    if (currentIndex < maxLen - 1) {
      currentIndex++;
      refreshBothPanels();
    }
  });
  resetBtn.addEventListener("click", function () {
    currentIndex = INITIAL_STATE_INDEX;
    refreshBothPanels();
  });

  let resizeTimeout;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(refreshBothPanels, 100);
  });

  loadTracesForMode();
})();
