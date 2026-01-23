# Talento AI Agent

## Setup

### 1. Create virtual environment

```bash
uv venv --python 3.11
```

### 2. Activate virtual environment

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install .
```

## Manage dependencies

### Add new package

```bash
uv add <package_name>
```

### Remove package

```bash
uv remove <package_name>
```

### List dependencies in tree view

```bash
uv tree
```

## Run project

```bash
uv run langgraph dev
```
