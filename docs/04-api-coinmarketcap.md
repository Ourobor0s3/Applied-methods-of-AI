### `docs/04-api-coinmarketcap.md`

```markdown
# API CoinMarketCap (Pro)

**CoinMarketCap** — ведущая платформа аналитики криптовалютного рынка.

## Endpoint: Latest Quotes

**Метод:** `GET`  
**URL:** `https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest`

⚠️ **Аутентификация:** заголовок `X-CMC_PRO_API_KEY` с вашим API-ключом.

### Параметры запроса

| Параметр | Обязательность | Описание |
|----------|----------------|----------|
| `id` | ✅ Обязательный | ID криптовалюты (например, `1` для BTC) |
| `aux` | ❌ Необязательный | Дополнительные метрики |
| `convert` | ❌ (default: USD) | Валюта котировок |
| `skip_invalid` | ❌ (default: false) | Пропускать невалидные ID |

### Структура ответа

```json
{
  "status": {
    "timestamp": "2026-03-01T09:28:09.275Z",
    "error_code": 0,
    "elapsed": 17,
    "credit_count": 1
  },
  "data": {
    "1": {
      "id": 1,
      "name": "Bitcoin",
      "symbol": "BTC",
      "cmc_rank": 1,
      "quote": {
        "EUR": {
          "price": 56378.09,
          "volume_24h": 37030972078.71,
          "volume_change_24h": 4.6852,
          "percent_change_1h": -0.9499,
          "percent_change_24h": 4.2376,
          "market_cap": 1127357602424.33,
          "market_cap_dominance": 57.8478
        }
      }
    }
  }
}

Основные поля Quote Object


Поле	                Тип	    Описание
price	                number	Текущая цена
volume_24h	          number	Объём торгов за 24 часа
volume_change_24h	    number	Изменение объёма за 24ч (%)
percent_change_1h	    number	Изменение цены за 1 час (%)
percent_change_24h	  number	Изменение цены за 24 часа (%)
market_cap	          number	Рыночная капитализация
market_cap_dominance	number	Доля рынка (%)
