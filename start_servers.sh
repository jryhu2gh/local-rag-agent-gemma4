#!/bin/bash
# Start local llama-server instances for the RAG agent.
# Chat server on :8080 (Gemma 4), Embedding server on :8081 (EmbeddingGemma).
# Usage: ./start_servers.sh [chat|embed|both]   (default: both)

LLAMA_DIR=${LLAMA_DIR:-~/local-llm/llama.cpp}
SERVER=$LLAMA_DIR/build/bin/llama-server
MODEL=${CHAT_MODEL_PATH:-~/local-llm/gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf}
TEMPLATE=$LLAMA_DIR/models/templates/gemma4.jinja
EMBED_MODEL=${EMBED_MODEL_PATH:-~/local-llm/gemma4/embeddinggemma-300M-qat-Q4_0.gguf}

MODE=${1:-both}

start_chat() {
    echo "Starting chat server on :8080..."
    $SERVER \
        -m "$MODEL" \
        --port 8080 \
        --jinja \
        --chat-template-file "$TEMPLATE" \
        -fa on \
        -c 8192 \
        --no-webui &
    echo "Chat server PID: $!"
}

start_embed() {
    echo "Starting embedding server on :8081..."
    $SERVER \
        -m "$EMBED_MODEL" \
        --embeddings \
        --port 8081 \
        --no-webui &
    echo "Embedding server PID: $!"
}

case $MODE in
    chat)  start_chat ;;
    embed) start_embed ;;
    both)  start_chat; start_embed ;;
    *)     echo "Usage: $0 [chat|embed|both]"; exit 1 ;;
esac

echo ""
echo "Waiting for servers to start..."
sleep 5

if [[ "$MODE" == "both" || "$MODE" == "chat" ]]; then
    if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
        echo "Chat server (:8080) is ready."
    else
        echo "Chat server (:8080) not responding yet — may still be loading the model."
    fi
fi

if [[ "$MODE" == "both" || "$MODE" == "embed" ]]; then
    if curl -s http://localhost:8081/v1/models > /dev/null 2>&1; then
        echo "Embedding server (:8081) is ready."
    else
        echo "Embedding server (:8081) not responding yet — may still be loading/downloading the model."
    fi
fi

echo ""
echo "Servers running in background. Use 'pkill llama-server' to stop all."
wait
