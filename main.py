import base64
import json
import logging
import os
import re
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import telebot
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from telebot import types




BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.json"
PHOTOS_DIR = BASE_DIR / "photos"
PHOTOS_DIR.mkdir(exist_ok=True)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
BOT_TIMEZONE = os.environ.get("BOT_TIMEZONE", "Asia/Jerusalem")
ALLOWED_USER_IDS = {
    int(value.strip())
    for value in os.environ["ALLOWED_USER_IDS"].split(",")
    if value.strip()
}

TZ = ZoneInfo(BOT_TIMEZONE)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("nutrition-bot")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
client = OpenAI(api_key=OPENAI_API_KEY)


NUTRIENTS = [
    ("calories", "Calories", "kcal"),
    ("proteins", "Proteins", "g"),
    ("fats", "Fats", "g"),
    ("carbohydrates", "Carbohydrates", "g"),
    ("sugar", "Sugar", "g"),
    ("fibres", "Fibres", "g"),
    ("sodium", "Sodium", "mg"),
    ("potassium", "Potassium", "mg"),
    ("calcium", "Calcium", "mg"),
    ("iron", "Iron", "mg"),
    ("vitamin_a", "Vitamin A", "mcg RAE"),
    ("vitamin_c", "Vitamin C", "mg"),
]

NUTRIENT_META = {key: {"label": label, "unit": unit} for key, label, unit in NUTRIENTS}
BASIC_REPORT_KEYS = ["proteins", "calories", "sugar"]


def now_local() -> datetime:
    return datetime.now(TZ)


def isoformat_local(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=TZ)
    return value.astimezone(TZ).isoformat()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(TZ)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def ensure_data_shape(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("entries", [])
    data.setdefault("starts", [])
    data.setdefault("goals", {})
    data.setdefault("pending_goals", {})
    legacy_users = data.pop("users", None)
    if isinstance(legacy_users, dict):
        for legacy_bucket in legacy_users.values():
            if not isinstance(legacy_bucket, dict):
                continue
            data["entries"].extend(legacy_bucket.get("entries", []))
            data["starts"].extend(legacy_bucket.get("starts", []))
            for key, value in legacy_bucket.get("goals", {}).items():
                data["goals"].setdefault(key, value)
    return data


class JsonStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        if not self.path.exists():
            self._write(ensure_data_shape({}))

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return ensure_data_shape(json.load(handle))

    def _write(self, data: dict[str, Any]) -> None:
        temp_path = self.path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        temp_path.replace(self.path)

    def get_data(self) -> dict[str, Any]:
        with self.lock:
            return deepcopy(self._read())

    def update(self, updater) -> Any:
        with self.lock:
            data = self._read()
            result = updater(data)
            self._write(data)
            return result


store = JsonStore(DATA_FILE)


def pending_goal_for(data: dict[str, Any], user_id: int) -> str | None:
    return data.get("pending_goals", {}).get(str(user_id))


def set_pending_goal(data: dict[str, Any], user_id: int, nutrient: str | None) -> None:
    pending_goals = data.setdefault("pending_goals", {})
    user_key = str(user_id)
    if nutrient is None:
        pending_goals.pop(user_key, None)
    else:
        pending_goals[user_key] = nutrient


def is_authorized_user(user_id: int | None) -> bool:
    return user_id is not None and user_id in ALLOWED_USER_IDS


def reject_if_unauthorized(message: types.Message) -> bool:
    if is_authorized_user(message.from_user.id if message.from_user else None):
        return False
    bot.reply_to(message, "You are not allowed to use this bot.")
    return True


def reject_callback_if_unauthorized(call: types.CallbackQuery) -> bool:
    if is_authorized_user(call.from_user.id if call.from_user else None):
        return False
    bot.answer_callback_query(call.id, "You are not allowed to use this bot.", show_alert=True)
    return True


class Nutrients(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calories: float = Field(ge=0)
    proteins: float = Field(ge=0)
    fats: float = Field(ge=0)
    carbohydrates: float = Field(ge=0)
    sugar: float = Field(ge=0)
    fibres: float = Field(ge=0)
    sodium: float = Field(ge=0)
    potassium: float = Field(ge=0)
    calcium: float = Field(ge=0)
    iron: float = Field(ge=0)
    vitamin_a: float = Field(ge=0)
    vitamin_c: float = Field(ge=0)


class MealEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description_summary: str
    eaten_at_iso: str
    assumptions: list[str]
    nutrients: Nutrients


class TimeParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iso_datetime: str
    explanation: str


def mechanistic_parse_time(raw_text: str, reference: datetime) -> datetime | None:
    text = raw_text.strip()
    if not text:
        return None

    normalized = text.lower().replace(",", " ").strip()
    normalized = re.sub(r"\s+", " ", normalized)

    full_formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H.%M",
        "%Y/%m/%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%d.%m.%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%d.%m.%y %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d-%m-%Y",
    ]
    for fmt in full_formats:
        try:
            parsed = datetime.strptime(normalized, fmt)
            if "H" not in fmt:
                parsed = parsed.replace(hour=0, minute=0)
            return parsed.replace(tzinfo=TZ)
        except ValueError:
            continue

    compact_time_patterns = [
        "%H:%M",
        "%H.%M",
        "%H",
        "%I:%M%p",
        "%I%p",
        "%I %p",
        "%I:%M %p",
    ]
    for fmt in compact_time_patterns:
        try:
            parsed = datetime.strptime(normalized.replace(" ", ""), fmt.replace(" ", ""))
            candidate = reference.replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0,
            )
            if candidate > reference:
                candidate -= timedelta(days=1)
            return candidate
        except ValueError:
            continue

    match = re.fullmatch(r"(\d{1,2})(?::|\.)(\d{2})", normalized)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour < 24 and minute < 60:
            candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate > reference:
                candidate -= timedelta(days=1)
            return candidate

    return None


def llm_parse_time(raw_text: str, reference: datetime) -> TimeParseResult:
    prompt = (
        "Parse the user's intended time for a 'start of the day' record.\n"
        f"Reference time: {isoformat_local(reference)}\n"
        f"Timezone: {BOT_TIMEZONE}\n"
        "Interpret the user's text as a past moment. If the user gives only a time of day, "
        "choose the most recent occurrence of that time not after the reference time. "
        "Return a full ISO 8601 datetime with timezone."
    )
    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_text},
        ],
        text_format=TimeParseResult,
    )
    return response.output_parsed


def parse_start_time(raw_text: str, reference: datetime) -> tuple[datetime, str]:
    parsed = mechanistic_parse_time(raw_text, reference)
    if parsed is not None:
        return parsed, "mechanistic"
    llm_result = llm_parse_time(raw_text, reference)
    return parse_iso(llm_result.iso_datetime), f"llm: {llm_result.explanation}"


def get_file_ext(path: str | None) -> str:
    if not path or "." not in path:
        return ".jpg"
    return "." + path.rsplit(".", 1)[-1].lower()


def save_photo(message: types.Message) -> dict[str, Any]:
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    file_bytes = bot.download_file(file_info.file_path)
    extension = get_file_ext(file_info.file_path)
    filename = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}{extension}"
    local_path = PHOTOS_DIR / filename
    local_path.write_bytes(file_bytes)
    return {
        "telegram_file_id": photo.file_id,
        "telegram_file_unique_id": photo.file_unique_id,
        "telegram_file_path": file_info.file_path,
        "local_path": str(local_path.relative_to(BASE_DIR)),
    }


def build_meal_input_text(message: types.Message, photo_saved: bool) -> str:
    text = (message.caption or message.text or "").strip()
    if not text:
        text = "No description provided."
    sent_at = datetime.fromtimestamp(message.date, TZ)
    parts = [
        f"Message sent at: {isoformat_local(sent_at)}",
        f"Timezone: {BOT_TIMEZONE}",
        f"Photo attached: {'yes' if photo_saved else 'no'}",
        f"User description: {text}",
    ]
    return "\n".join(parts)


def estimate_meal(description: str, photo_path: Path | None) -> MealEstimate:
    system_prompt = (
        "You extract structured nutrition logs from a user's meal message. "
        "Estimate nutrients and the meal time from the message text and image when present. "
        "Use the message send time when the user does not specify a different time. "
        "Return conservative but useful estimates. All nutrient values must be non-negative.\n"
        f"Timezone: {BOT_TIMEZONE}\n"
        "Units: calories=kcal, proteins/fats/carbohydrates/sugar/fibres=g, "
        "sodium/potassium/calcium/iron/vitamin_c=mg, vitamin_a=mcg RAE."
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": description}]
    if photo_path is not None:
        image_bytes = photo_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/jpeg"
        suffix = photo_path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".webp":
            mime_type = "image/webp"
        elif suffix == ".gif":
            mime_type = "image/gif"
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            }
        )

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        text_format=MealEstimate,
    )
    return response.output_parsed


def init_commands() -> None:
    commands = [
        types.BotCommand("start", "Record the start of the day, optionally with a time"),
        types.BotCommand("check", "Show nutrient totals"),
        types.BotCommand("remove_start", "Remove the latest start-of-day record"),
        types.BotCommand("goals", "Set nutrient goals"),
    ]
    bot.set_my_commands(commands)


def sum_entries(entries: list[dict[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in NUTRIENT_META}
    for entry in entries:
        for key in totals:
            totals[key] += float(entry["nutrients"].get(key, 0.0))
    return totals


def get_windows(data: dict[str, Any], reference: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime | None]:
    entries = []
    for entry in data["entries"]:
        eaten_at = parse_iso(entry["eaten_at_iso"])
        if eaten_at <= reference:
            entries.append((eaten_at, entry))

    last_24h_start = reference - timedelta(hours=24)
    entries_24h = [entry for eaten_at, entry in entries if eaten_at >= last_24h_start]

    starts = sorted((parse_iso(value) for value in data["starts"]), reverse=True)
    active_start = next((value for value in starts if value <= reference), None)
    entries_since_start = [entry for eaten_at, entry in entries if active_start and eaten_at >= active_start]

    return entries_24h, entries_since_start, active_start


def goal_text(value: float, key: str, goals: dict[str, Any]) -> str:
    goal = goals.get(key)
    unit = NUTRIENT_META[key]["unit"]
    if goal is None:
        return f"{format_number(value)} {unit}"
    return f"{format_number(value)} / {format_number(float(goal))} {unit}"


def render_report(data: dict[str, Any], expanded: bool) -> str:
    reference = now_local()
    entries_24h, entries_since_start, active_start = get_windows(data, reference)
    totals_24h = sum_entries(entries_24h)
    totals_since_start = sum_entries(entries_since_start)
    goals = data.get("goals", {})

    keys = list(NUTRIENT_META) if expanded else BASIC_REPORT_KEYS
    header = [
        "<b>Nutrition Check</b>",
        f"<i>Updated:</i> {escape(reference.strftime('%Y-%m-%d %H:%M %Z'))}",
        f"<i>Last 24h entries:</i> {len(entries_24h)}",
    ]
    if active_start is None:
        header.append("<i>Start of day:</i> none recorded")
    else:
        header.append(f"<i>Start of day:</i> {escape(active_start.strftime('%Y-%m-%d %H:%M %Z'))}")

    rows = []
    rows.append(f"{'Nutrient':<16} {'Last 24h':<16} Since start")
    rows.append("-" * 52)
    for key in keys:
        label = NUTRIENT_META[key]["label"]
        left = goal_text(totals_24h[key], key, goals)
        right = goal_text(totals_since_start[key], key, goals)
        rows.append(f"{label:<16} {left:<16} {right}")

    body = "<pre>" + escape("\n".join(rows)) + "</pre>"
    return "\n".join(header + [body])


def report_markup(expanded: bool) -> types.InlineKeyboardMarkup:
    keyboard = types.InlineKeyboardMarkup()
    toggle_text = "Show less" if expanded else "Show more"
    toggle_value = "less" if expanded else "more"
    keyboard.add(types.InlineKeyboardButton(toggle_text, callback_data=f"check:{toggle_value}"))
    return keyboard


def goals_markup(data: dict[str, Any]) -> types.InlineKeyboardMarkup:
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    for key, label, unit in NUTRIENTS:
        goal = data.get("goals", {}).get(key)
        suffix = f" ({format_number(float(goal))} {unit})" if goal is not None else ""
        buttons.append(types.InlineKeyboardButton(f"{label}{suffix}", callback_data=f"goal:{key}"))
    keyboard.add(*buttons)
    keyboard.add(types.InlineKeyboardButton("Clear a goal", callback_data="goal:clear_menu"))
    return keyboard


def clear_goals_markup(data: dict[str, Any]) -> types.InlineKeyboardMarkup:
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    for key, label, unit in NUTRIENTS:
        if data.get("goals", {}).get(key) is not None:
            buttons.append(types.InlineKeyboardButton(f"Remove {label}", callback_data=f"goalclear:{key}"))
    if buttons:
        keyboard.add(*buttons)
    keyboard.add(types.InlineKeyboardButton("Back", callback_data="goal:back"))
    return keyboard


@dataclass
class StoredEntry:
    message_id: int
    description: str
    photo: dict[str, Any] | None
    estimate: MealEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "description": self.description,
            "photo": self.photo,
            "description_summary": self.estimate.description_summary,
            "eaten_at_iso": self.estimate.eaten_at_iso,
            "assumptions": self.estimate.assumptions,
            "nutrients": self.estimate.nutrients.model_dump(),
            "recorded_at_iso": isoformat_local(now_local()),
        }


def store_entry(entry: StoredEntry) -> None:
    def updater(data: dict[str, Any]) -> None:
        data["entries"].append(entry.to_dict())

    store.update(updater)


def send_processing_error(chat_id: int, exc: Exception) -> None:
    LOGGER.exception("Processing failed", exc_info=exc)
    bot.send_message(
        chat_id,
        "Could not process that meal entry. Check bot logs and API credentials, then try again.",
    )


def handle_meal_message(message: types.Message) -> None:
    if reject_if_unauthorized(message):
        return
    try:
        photo_meta = save_photo(message) if message.content_type == "photo" else None
        content_text = build_meal_input_text(message, photo_meta is not None)
        photo_path = BASE_DIR / photo_meta["local_path"] if photo_meta else None
        estimate = estimate_meal(content_text, photo_path)
        entry = StoredEntry(
            message_id=message.message_id,
            description=(message.caption or message.text or "").strip(),
            photo=photo_meta,
            estimate=estimate,
        )
        store_entry(entry)

        summary_lines = [
            "<b>Meal recorded</b>",
            f"<i>Time:</i> {escape(parse_iso(estimate.eaten_at_iso).strftime('%Y-%m-%d %H:%M %Z'))}",
            f"<i>Summary:</i> {escape(estimate.description_summary)}",
            f"<i>Calories:</i> {format_number(estimate.nutrients.calories)} kcal",
            f"<i>Proteins:</i> {format_number(estimate.nutrients.proteins)} g",
            f"<i>Sugar:</i> {format_number(estimate.nutrients.sugar)} g",
        ]
        if estimate.assumptions:
            summary_lines.append("<i>Assumptions:</i> " + escape("; ".join(estimate.assumptions[:3])))
        bot.reply_to(message, "\n".join(summary_lines))
    except Exception as exc:
        send_processing_error(message.chat.id, exc)


@bot.message_handler(commands=["start"])
def handle_start_command(message: types.Message) -> None:
    if reject_if_unauthorized(message):
        return
    command_text = message.text or "/start"
    raw_arg = command_text.split(maxsplit=1)[1].strip() if " " in command_text else ""
    reference = datetime.fromtimestamp(message.date, TZ)

    if raw_arg:
        try:
            parsed_time, source = parse_start_time(raw_arg, reference)
        except Exception as exc:
            send_processing_error(message.chat.id, exc)
            return
    else:
        parsed_time, source = reference, "message time"

    def updater(data: dict[str, Any]) -> str:
        data["starts"].append(isoformat_local(parsed_time))
        return isoformat_local(parsed_time)

    stored_value = store.update(updater)
    bot.reply_to(
        message,
        "\n".join(
            [
                "<b>Start of day recorded</b>",
                f"<i>When:</i> {escape(parse_iso(stored_value).strftime('%Y-%m-%d %H:%M %Z'))}",
                f"<i>Source:</i> {escape(source)}",
            ]
        ),
    )


@bot.message_handler(commands=["remove_start"])
def handle_remove_start(message: types.Message) -> None:
    if reject_if_unauthorized(message):
        return
    def updater(data: dict[str, Any]) -> str | None:
        if not data["starts"]:
            return None
        return data["starts"].pop()

    removed = store.update(updater)
    if removed is None:
        bot.reply_to(message, "No start-of-day record exists yet.")
        return
    bot.reply_to(
        message,
        f"<b>Removed</b>\n<i>Start of day:</i> {escape(parse_iso(removed).strftime('%Y-%m-%d %H:%M %Z'))}",
    )


@bot.message_handler(commands=["check"])
def handle_check(message: types.Message) -> None:
    if reject_if_unauthorized(message):
        return
    data = store.get_data()
    bot.send_message(
        message.chat.id,
        render_report(data, expanded=False),
        reply_markup=report_markup(expanded=False),
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("check:"))
def handle_check_toggle(call: types.CallbackQuery) -> None:
    if reject_callback_if_unauthorized(call):
        return
    expanded = call.data == "check:more"
    data = store.get_data()
    bot.edit_message_text(
        render_report(data, expanded=expanded),
        call.message.chat.id,
        call.message.message_id,
        reply_markup=report_markup(expanded=expanded),
    )
    bot.answer_callback_query(call.id)


@bot.message_handler(commands=["goals"])
def handle_goals(message: types.Message) -> None:
    if reject_if_unauthorized(message):
        return
    data = store.get_data()
    bot.send_message(
        message.chat.id,
        "<b>Goals</b>\nChoose a nutrient to set or update its goal.",
        reply_markup=goals_markup(data),
    )


@bot.callback_query_handler(func=lambda call: call.data.startswith("goal:") or call.data.startswith("goalclear:"))
def handle_goal_callbacks(call: types.CallbackQuery) -> None:
    if reject_callback_if_unauthorized(call):
        return
    if call.data == "goal:clear_menu":
        data = store.get_data()
        bot.edit_message_text(
            "<b>Goals</b>\nChoose a goal to remove.",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=clear_goals_markup(data),
        )
        bot.answer_callback_query(call.id)
        return

    if call.data == "goal:back":
        data = store.get_data()
        bot.edit_message_text(
            "<b>Goals</b>\nChoose a nutrient to set or update its goal.",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=goals_markup(data),
        )
        bot.answer_callback_query(call.id)
        return

    if call.data.startswith("goalclear:"):
        nutrient = call.data.split(":", 1)[1]

        def updater(data: dict[str, Any]) -> None:
            data["goals"].pop(nutrient, None)

        store.update(updater)
        data = store.get_data()
        bot.edit_message_text(
            "<b>Goals</b>\nChoose a goal to remove.",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=clear_goals_markup(data),
        )
        bot.answer_callback_query(call.id, f"Removed goal for {NUTRIENT_META[nutrient]['label']}")
        return

    nutrient = call.data.split(":", 1)[1]
    label = NUTRIENT_META[nutrient]["label"]
    unit = NUTRIENT_META[nutrient]["unit"]

    def updater(data: dict[str, Any]) -> None:
        set_pending_goal(data, call.from_user.id, nutrient)

    store.update(updater)
    bot.answer_callback_query(call.id)
    bot.send_message(
        call.message.chat.id,
        f"Send the goal for <b>{escape(label)}</b> as a number in {escape(unit)}.",
    )


@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_text_message(message: types.Message) -> None:
    if (message.text or "").startswith("/"):
        return
    if reject_if_unauthorized(message):
        return

    data = store.get_data()
    pending_goal = pending_goal_for(data, message.from_user.id)
    if pending_goal:
        raw_value = (message.text or "").strip().replace(",", ".")
        try:
            value = float(raw_value)
            if value < 0:
                raise ValueError("goal must be non-negative")
        except ValueError:
            bot.reply_to(message, "Send a non-negative number.")
            return

        def updater(data: dict[str, Any]) -> None:
            data["goals"][pending_goal] = value
            set_pending_goal(data, message.from_user.id, None)

        store.update(updater)
        meta = NUTRIENT_META[pending_goal]
        bot.reply_to(
            message,
            f"<b>Goal saved</b>\n{escape(meta['label'])}: {format_number(value)} {escape(meta['unit'])}",
        )
        return

    handle_meal_message(message)


@bot.message_handler(content_types=["photo"])
def handle_photo_message(message: types.Message) -> None:
    if (message.caption or "").startswith("/"):
        return
    if reject_if_unauthorized(message):
        return
    handle_meal_message(message)


def main() -> None:
    init_commands()
    LOGGER.info("Starting bot with model=%s timezone=%s", OPENAI_MODEL, BOT_TIMEZONE)
    bot.infinity_polling(timeout=30, long_polling_timeout=30)


if __name__ == "__main__":
    main()
