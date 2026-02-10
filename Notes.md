## Общие заметки по проекту

**Общие команды**
```bash
venv/scripts/activate
```
```bash
cd src/tests
```
```bash
python -m test_... .py
```

**Тестирование модулей**

Код для импорта модулей в тесты
```python
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
```
