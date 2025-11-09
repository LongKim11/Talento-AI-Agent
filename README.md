uv venv --python 3.11

source .venv/bin/activate

uv add <package_name>

uv remove <package_name>

uv tree

uv run main.py

utils -> chunk (text, metadata) -> embedding to VectorDB

tools -> search schema -> search tool

tools -> web search company info

curl -s --request POST \
 --url "http://localhost:2024/runs/stream" \
 --header 'Content-Type: application/json' \
 --data "{
\"assistant_id\": \"agent\",
\"input\": {
\"messages\": [
{
\"role\": \"human\",
\"content\": \"What is LangGraph?\"
}
]
},
\"stream_mode\": \"messages-tuple\"
}"
