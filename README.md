# Talento AI Agent

## Setup

### 1. Tạo virtual environment

```bash
uv venv --python 3.11
```

### 2. Kích hoạt virtual environment

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
uv pip install .
```

## Quản lý dependencies

### Thêm package mới

```bash
uv add <package_name>
```

### Xóa package

```bash
uv remove <package_name>
```

### Xem cây dependencies

```bash
uv tree
```

## Chạy project

```bash
uv run langgraph dev
```
