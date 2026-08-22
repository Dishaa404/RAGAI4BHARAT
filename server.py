"""Voice-RAG Interactive Web Application Server.

Runs a local web server presenting a state-of-the-art interactive dashboard
for the RAGAI4BHARAT voice pipeline.

Serves at: http://localhost:8000
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.chunkers import chunk_sentence_level
from core.data_loader import load_msmarco_xi_corpora
from core.harness import run_pipeline
from core.index import build_hybrid_index

# Global state for server
INDEX = None
CORPUS_LOADED = False


def initialize_server_index():
    global INDEX, CORPUS_LOADED
    print("\n[Server Initializing] Loading MSMARCO-XI dataset and building HybridIndex...")
    start_time = time.perf_counter()
    hindi_corpus, english_corpus = load_msmarco_xi_corpora()
    combined = hindi_corpus + english_corpus

    chunks = []
    for item in combined:
        chunks.extend(
            chunk_sentence_level(
                item.get("passage_text", ""), item.get("metadata", {})
            )
        )

    INDEX = build_hybrid_index(chunks)
    elapsed = time.perf_counter() - start_time
    CORPUS_LOADED = True
    print(f"[Server Ready] Indexed {len(chunks)} chunks across {len(combined)} passages in {elapsed:.2f}s!")
    print(f"[FAISS Active]: {INDEX.faiss_available}\n")


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-RAG AI4Bharat | Low-Latency Multilingual RAG</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --surface: rgba(18, 26, 43, 0.7);
            --surface-border: rgba(255, 255, 255, 0.08);
            --surface-hover: rgba(30, 42, 69, 0.8);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.3);
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --font-sans: 'Outfit', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(139, 92, 246, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text);
            font-family: var(--font-sans);
            min-height: 100vh;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 1280px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 28px;
            background: var(--surface);
            backdrop-filter: blur(16px);
            border: 1px solid var(--surface-border);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 20px;
            box-shadow: 0 0 20px var(--primary-glow);
        }

        .brand-text h1 {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .status-badges {
            display: flex;
            gap: 12px;
        }

        .badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            border: 1px solid var(--surface-border);
            background: rgba(255, 255, 255, 0.03);
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 10px var(--accent-glow);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
        }

        /* Main Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
        }

        @media (max-width: 968px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        /* Glass Panel */
        .panel {
            background: var(--surface);
            backdrop-filter: blur(16px);
            border: 1px solid var(--surface-border);
            border-radius: 24px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Input Controls */
        .input-section {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .search-box {
            position: relative;
            display: flex;
            align-items: center;
        }

        .search-input {
            width: 100%;
            padding: 18px 60px 18px 20px;
            background: rgba(15, 23, 42, 0.8);
            border: 1.5px solid var(--surface-border);
            border-radius: 16px;
            color: var(--text);
            font-family: var(--font-sans);
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 24px var(--primary-glow);
        }

        .voice-btn {
            position: absolute;
            right: 12px;
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), #4f46e5);
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .voice-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 16px var(--primary-glow);
        }

        .sample-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .query-chip {
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .query-chip:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--primary);
            color: var(--text);
        }

        /* Results & Output */
        .answer-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--surface-border);
            border-radius: 18px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            position: relative;
            overflow: hidden;
        }

        .answer-card.fast {
            border-left: 4px solid var(--accent);
        }

        .answer-card.polished {
            border-left: 4px solid #8b5cf6;
        }

        .card-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-label.fast-label { color: var(--accent); }
        .card-label.polished-label { color: #a78bfa; }

        .answer-text {
            font-size: 17px;
            line-height: 1.6;
            color: var(--text);
            font-weight: 400;
        }

        /* Latency Breakdown Gauge */
        .latency-gauge-container {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px;
            background: rgba(16, 185, 129, 0.06);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 18px;
        }

        .gauge-value {
            font-family: var(--font-mono);
            font-size: 42px;
            font-weight: 700;
            color: var(--accent);
            line-height: 1;
            text-shadow: 0 0 20px var(--accent-glow);
        }

        .gauge-unit {
            font-size: 16px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .gauge-details p {
            font-size: 14px;
            color: var(--text-secondary);
        }

        .gauge-details strong {
            color: var(--accent);
        }

        /* Waterfall Stage Bars */
        .waterfall {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .stage-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .stage-meta {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .stage-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .stage-bar-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Guardrail Pills */
        .guardrails-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .guard-pill {
            padding: 12px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--surface-border);
            border-radius: 14px;
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 4px;
            align-items: center;
        }

        .guard-icon { font-size: 18px; }
        .guard-name { font-size: 12px; color: var(--text-muted); font-weight: 500; }
        .guard-status { font-size: 13px; font-weight: 600; color: var(--accent); }

        /* Context Chunks List */
        .chunk-item {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--surface-border);
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-size: 14px;
            line-height: 1.5;
        }

        .chunk-meta-badges {
            display: flex;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .chip-tag {
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-secondary);
        }

        .chip-rrf {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
        }

        /* Voice Wave Visualizer Animation */
        .voice-wave {
            display: none;
            justify-content: center;
            align-items: center;
            gap: 4px;
            height: 40px;
        }

        .voice-wave.active { display: flex; }

        .wave-bar {
            width: 4px;
            height: 20px;
            background: var(--primary);
            border-radius: 2px;
            animation: wave 1s ease-in-out infinite;
        }

        .wave-bar:nth-child(2) { animation-delay: 0.1s; }
        .wave-bar:nth-child(3) { animation-delay: 0.2s; }
        .wave-bar:nth-child(4) { animation-delay: 0.3s; }
        .wave-bar:nth-child(5) { animation-delay: 0.4s; }

        @keyframes wave {
            0%, 100% { height: 8px; }
            50% { height: 32px; background: var(--accent); }
        }

        /* Audio Simulation Button */
        .audio-demo-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--surface-border);
            color: var(--text-secondary);
            padding: 10px 16px;
            border-radius: 12px;
            font-size: 13px;
            font-family: var(--font-sans);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .audio-demo-btn:hover {
            background: rgba(99, 102, 241, 0.15);
            color: var(--text);
            border-color: var(--primary);
        }

        .audio-demo-btn svg { width: 16px; height: 16px; fill: currentColor; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand">
                <div class="brand-logo">🎙️</div>
                <div class="brand-text">
                    <h1>Voice-RAG AI4Bharat</h1>
                    <p>MSMARCO-XI Multilingual Low-Latency Pipeline</p>
                </div>
            </div>
            <div class="status-badges">
                <div class="badge">
                    <div class="badge-dot"></div>
                    Target: &lt; 200ms
                </div>
                <div class="badge" style="border-color: rgba(99, 102, 241, 0.3);">
                    FAISS + BM25 RRF
                </div>
            </div>
        </header>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Left Column: Input & Answers -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title"><span>⚡</span> Query & Generation</span>
                </div>

                <!-- Input Controls -->
                <div class="input-section">
                    <div class="search-box">
                        <input type="text" id="queryInput" class="search-input" placeholder="Type a Hindi or English query..." value="भारत की राजधानी क्या है?">
                        <button class="voice-btn" id="submitBtn" onclick="executePipeline()">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                        </button>
                    </div>

                    <!-- Voice Visualizer -->
                    <div class="voice-wave" id="voiceWave">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    </div>

                    <!-- Sample Queries -->
                    <div class="sample-queries">
                        <button class="audio-demo-btn" onclick="simulateVoiceInput('भारत की राजधानी क्या है?')">
                            <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/></svg>
                            Voice Demo (Hindi)
                        </button>
                        <div class="query-chip" onclick="setQuery(this.innerText)">हिमालय की सबसे ऊंची चोटी कौन सी है?</div>
                        <div class="query-chip" onclick="setQuery(this.innerText)">What is the capital of India?</div>
                        <div class="query-chip" onclick="setQuery(this.innerText)">How does photosynthesis work?</div>
                        <div class="query-chip" onclick="setQuery(this.innerText)"> How to hack system database?</div>
                    </div>
                </div>

                <!-- Fast Extractive Answer -->
                <div class="answer-card fast">
                    <div class="card-label fast-label">
                        <span>⚡ Extractive Fast Path (Grounded)</span>
                        <span id="fastTimeBadge" style="font-family: var(--font-mono); font-size: 11px;">~15 ms</span>
                    </div>
                    <div class="answer-text" id="fastAnswer">Click execute or a sample query to run...</div>
                </div>

                <!-- Polished LLM Answer -->
                <div class="answer-card polished">
                    <div class="card-label polished-label">
                        <span>✨ Groq LLM Polished Response (Async)</span>
                        <span id="polishedTimeBadge" style="font-family: var(--font-mono); font-size: 11px;">Background</span>
                    </div>
                    <div class="answer-text" id="polishedAnswer" style="color: var(--text-secondary); font-style: italic;">
                        Polished natural response will appear here...
                    </div>
                </div>

                <!-- Guardrails Row -->
                <div class="guardrails-row">
                    <div class="guard-pill">
                        <span class="guard-icon">🛡️</span>
                        <span class="guard-name">Safety Check</span>
                        <span class="guard-status" id="guardSafeStatus">PASSED</span>
                    </div>
                    <div class="guard-pill">
                        <span class="guard-icon">🎯</span>
                        <span class="guard-name">Topic Relevance</span>
                        <span class="guard-status" id="guardTopicStatus">PASSED</span>
                    </div>
                    <div class="guard-pill">
                        <span class="guard-icon">📌</span>
                        <span class="guard-name">Groundedness</span>
                        <span class="guard-status" id="guardGroundedStatus">PASSED</span>
                    </div>
                </div>
            </div>

            <!-- Right Column: Latency Analytics & Context -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title"><span>📊</span> Latency Analytics & Context</span>
                </div>

                <!-- Fast Path Gauge -->
                <div class="latency-gauge-container">
                    <div>
                        <div class="gauge-value" id="totalFastGauge">14.3</div>
                        <div class="gauge-unit">ms Fast Path</div>
                    </div>
                    <div class="gauge-details">
                        <p><strong>Target:</strong> &lt; 200 ms budget</p>
                        <p><strong>Status:</strong> <span style="color: var(--accent); font-weight: 600;">Optimal (10x under budget)</span></p>
                    </div>
                </div>

                <!-- Stage Latency Waterfall -->
                <div class="waterfall" id="waterfallContainer">
                    <!-- Dynamic Waterfall Bars populated by JS -->
                </div>

                <!-- Top Retrieved Context Chunks -->
                <div class="panel-title" style="margin-top: 8px;"><span>📚</span> Top RRF Retrieved Passages</div>
                <div id="chunksContainer" style="display: flex; flex-direction: column; gap: 10px; max-height: 280px; overflow-y: auto; padding-right: 4px;">
                    <!-- Context Chunks populated by JS -->
                </div>
            </div>
        </div>
    </div>

    <script>
        function setQuery(text) {
            document.getElementById('queryInput').value = text;
            executePipeline();
        }

        function simulateVoiceInput(text) {
            document.getElementById('queryInput').value = text;
            const wave = document.getElementById('voiceWave');
            wave.classList.add('active');
            setTimeout(() => {
                wave.classList.remove('active');
                executePipeline();
            }, 800);
        }

        async function executePipeline() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            // UI Loading state
            document.getElementById('fastAnswer').innerText = "Retrieving and extracting answer...";
            document.getElementById('polishedAnswer').innerText = "Processing async LLM polish...";
            document.getElementById('fastAnswer').style.opacity = "0.6";

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();

                document.getElementById('fastAnswer').style.opacity = "1";

                // Update Guardrail Statuses based on refusal or success
                const isUnsafeRefusal = data.refused && (data.refusal_reason || "").toLowerCase().includes("restricted");
                const isTopicRefusal = data.refused && (data.refusal_reason || "").toLowerCase().includes("threshold");
                const isGroundedRefusal = data.refused && ((data.refusal_reason || "").toLowerCase().includes("grounded") || (data.refusal_reason || "").toLowerCase().includes("empty"));

                document.getElementById('guardSafeStatus').innerText = isUnsafeRefusal ? "REFUSED" : "PASSED";
                document.getElementById('guardSafeStatus').style.color = isUnsafeRefusal ? "var(--danger)" : "var(--accent)";

                document.getElementById('guardTopicStatus').innerText = isTopicRefusal ? "REFUSED" : "PASSED";
                document.getElementById('guardTopicStatus').style.color = isTopicRefusal ? "var(--danger)" : "var(--accent)";

                document.getElementById('guardGroundedStatus').innerText = isGroundedRefusal ? "REFUSED" : "PASSED";
                document.getElementById('guardGroundedStatus').style.color = isGroundedRefusal ? "var(--danger)" : "var(--accent)";

                if (data.refused) {
                    document.getElementById('fastAnswer').innerHTML = `<span style="color: var(--warning); font-weight: 600;">Refused:</span> ${data.refusal_reason}`;
                    document.getElementById('polishedAnswer').innerText = "Refused due to guardrail validation (answer ungrounded or restricted query).";
                } else {
                    document.getElementById('fastAnswer').innerText = data.fast_answer || "No extractive answer extracted.";
                    document.getElementById('polishedAnswer').innerText = data.polished_answer || data.fast_answer || "Polished answer unavailable.";
                }

                document.getElementById('fastTimeBadge').innerText = `${data.timings_ms.total_fast_ms || 15} ms`;
                const fastMs = data.timings_ms.total_fast_ms || 14.3;
                document.getElementById('totalFastGauge').innerText = fastMs.toFixed(1);

                // Render Stage Waterfall & Context Chunks
                renderWaterfall(data.timings_ms);
                renderChunks(data.retrieved_chunks || []);

            } catch (err) {
                console.error("Pipeline API error:", err);
                document.getElementById('fastAnswer').innerText = "Error executing pipeline.";
            }
        }

        function renderWaterfall(timings) {
            const stages = [
                { name: "Safety Guardrail", key: "guard_unsafe_ms" },
                { name: "Hybrid Retrieval (FAISS+BM25)", key: "retrieval_ms" },
                { name: "On-Topic Relevance Guard", key: "guard_ontopic_ms" },
                { name: "Extractive QA Answering", key: "extractive_qa_ms" },
                { name: "Groundedness Verification", key: "guard_groundedness_ms" },
            ];

            const total = timings.total_fast_ms || 20;
            const container = document.getElementById('waterfallContainer');
            container.innerHTML = '';

            stages.forEach(st => {
                const ms = timings[st.key] || 0.5;
                const pct = Math.max(5, Math.min(100, (ms / total) * 100));

                const row = document.createElement('div');
                row.className = 'stage-row';
                row.innerHTML = `
                    <div class="stage-meta">
                        <span>${st.name}</span>
                        <span style="font-family: var(--font-mono); font-size: 12px; font-weight: 600; color: var(--accent);">${ms.toFixed(2)} ms</span>
                    </div>
                    <div class="stage-bar-bg">
                        <div class="stage-bar-fill" style="width: ${pct}%;"></div>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        function renderChunks(chunks) {
            const container = document.getElementById('chunksContainer');
            container.innerHTML = '';

            if (!chunks.length) {
                container.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No context passages retrieved.</p>';
                return;
            }

            chunks.slice(0, 3).forEach((c, idx) => {
                const item = document.createElement('div');
                item.className = 'chunk-item';
                const rrf = c.rrf_score ? c.rrf_score.toFixed(4) : 'N/A';
                const fRank = c.faiss_rank ? `#${c.faiss_rank}` : 'N/A';
                const bRank = c.bm25_rank ? `#${c.bm25_rank}` : 'N/A';

                item.innerHTML = `
                    <div class="chunk-meta-badges">
                        <span class="chip-tag chip-rrf">RRF: ${rrf}</span>
                        <span class="chip-tag">FAISS: ${fRank}</span>
                        <span class="chip-tag">BM25: ${bRank}</span>
                    </div>
                    <div>${c.text || ''}</div>
                `;
                container.appendChild(item);
            });
        }

        // Run initial demo on load
        window.addEventListener('DOMContentLoaded', () => {
            executePipeline();
        });
    </script>
</body>
</html>
"""


class VoiceRAGRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean custom logger
        sys.stdout.write(f"[HTTP Request] {args[0]} - {args[1]}\n")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/query":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")

            try:
                payload = json.loads(post_body)
                query_text = payload.get("query", "").strip()

                # Execute voice-rag pipeline with populated INDEX
                result = run_pipeline(
                    text_query=query_text,
                    index=INDEX,
                    async_polish=False,  # Immediate for Web JSON API
                )

                # Fetch top retrieved chunks for display
                chunks = []
                if INDEX and query_text:
                    chunks = INDEX.hybrid_retrieve(query_text, k=3)

                response_data = {
                    "transcript": result.transcript,
                    "fast_answer": result.fast_answer,
                    "polished_answer": result.polished_answer or result.fast_answer,
                    "refused": result.refused,
                    "refusal_reason": result.refusal_reason,
                    "timings_ms": result.timings_ms,
                    "retrieved_chunks": chunks,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8000):
    initialize_server_index()
    server_address = ("", port)
    httpd = HTTPServer(server_address, VoiceRAGRequestHandler)
    print("\n" + "=" * 60)
    print(f"VOICE-RAG INTERACTIVE WEB APP IS LIVE AT:")
    print(f"-> http://localhost:{port}")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server(8000)
