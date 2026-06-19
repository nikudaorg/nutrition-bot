# Nutrition Bot

Telegram bot for logging meals from free-form text and photos, estimating nutrients with an OpenAI GPT model, tracking start-of-day boundaries, and reporting totals against goals.

## Features

- Logs meal descriptions and optional photos.
- Estimates meal time and nutrients with OpenAI.
- Stores all data locally in `data.json`.
- Supports `/check`, `/begin`, `/remove_begin`, and `/goals`.
- Includes a `systemd` unit for Linux VPS deployment.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `BOT_TIMEZONE`
- `ALLOWED_USER_IDS`

## Run

```bash
. .venv/bin/activate
python main.py
```

## Deploy with systemd

1. Copy the project to a server location such as `/opt/nutrition-bot`.
2. Create a venv there and install requirements.
3. Copy `.env.example` to `.env` and fill it.
4. Edit `nutrition-bot.service` so `User`, `WorkingDirectory`, and `ExecStart` match the server.
5. Move the unit file into `/etc/systemd/system/`.
6. Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nutrition-bot.service
sudo systemctl status nutrition-bot.service
```

## Notes

- Photos are stored in `photos/`.
- Only Telegram users listed in `ALLOWED_USER_IDS` can use the bot.
- Data is global and shared across allowed users.
- Nutrient units:
  - calories: `kcal`
  - proteins, fats, carbohydrates, sugar, fibres: `g`
  - sodium, potassium, calcium, iron, vitamin c: `mg`
  - vitamin a: `mcg RAE`
