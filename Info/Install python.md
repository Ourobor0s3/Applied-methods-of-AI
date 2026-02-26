# Install python and dependences

## 1. Установка uv

### Через pip

```bash
pip install uv
```

- **macOS/Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Установка необходимой версии Python

```bash
uv python install 3.12
```

## 3. Установка зависимостей

Установите все необходимые пакеты из файла `uv.lock`:

```bash
uv sync
```
