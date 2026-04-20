# API CryptoPanic

⚠️ **Аутентификация:** во всех запросах необходимо указывать параметр `auth_token`.

**Базовый URL:**
https://cryptopanic.com/api/developer/v2


## Параметры запроса

### `currencies`
| Параметр | Значение |
|----------|----------|
| **Обязательность** | Необязательный |
| **Описание** | Фильтрация новостей по кодам криптовалют |
| **Пример** | `/api/developer/v2/posts/?auth_token=YOUR_TOKEN&currencies=BTC,ETH` |

### `filter`
| Параметр | Значение |
|----------|----------|
| **Обязательность** | Необязательный |
| **Описание** | Фильтрация по категориям: `rising`, `hot`, `bullish`, `bearish`, `important`, `saved`, `lol` |
| **Пример** | `/api/developer/v2/posts/?auth_token=YOUR_TOKEN&filter=rising` |

### `search` (только Enterprise)
| Параметр | Значение |
|----------|----------|
| **Обязательность** | Необязательный |
| **Описание** | Поиск новостей по ключевому слову |

## Структура ответа

```json
{
  "next": "string | null",
  "previous": "string | null",
  "results": [ ... ]
}

Основные поля Item Object

Поле	        Тип	    Описание
id	          integer	Уникальный идентификатор публикации
title	        string	Полный заголовок новости
description	  string	Краткое содержание
published_at	string  (ISO 8601)	Дата публикации
kind	        string	Тип контента: news, 
                      media, blog, twitter, reddit
votes	        object	Счётчики реакций 
                      (positive, negative,     important и др.)
panic_score	  integer (0–100)	Оценка важности новости
instruments	  array	  Список упомянутых криптовалют