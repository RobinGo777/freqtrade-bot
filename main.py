"""
English A2 Telegram Bot — v2.0
Glassmorphism дизайн + Native Telegram Quiz + Розумна система фонів
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, date
from io import BytesIO
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from playwright.async_api import async_playwright

# ──────────────────────────────────────────────
# НАЛАШТУВАННЯ ЛОГІВ
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# ЗМІННІ СЕРЕДОВИЩА
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID     = os.environ["TELEGRAM_CHAT_ID"]
# Три Gemini ключі з різних Google акаунтів — незалежні квоти
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]  # прибираємо порожні
UNSPLASH_ACCESS_KEY  = os.environ.get("UNSPLASH_ACCESS_KEY", "")
PEXELS_API_KEY       = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY      = os.environ.get("PIXABAY_API_KEY", "")

# ──────────────────────────────────────────────
# МОДЕЛІ
# ──────────────────────────────────────────────
# ── Gemini моделі (замініть на актуальні) ──
# Зараз підтверджено працює: gemini-2.0-flash-lite
# Після верифікації картки замініть на кращі моделі
# Замініть на актуальні моделі після верифікації картки
GEMINI_MODELS = [
    "gemini-3.1-flash-lite-preview",    # 500 RPD — основна
    "gemini-2.5-flash-lite",            # 20 RPD — запасна
    "gemini-2.5-flash",                 # 20 RPD — запасна
    "gemini-2.0-flash-lite",             # модель 3 — запасна
]

# ──────────────────────────────────────────────
# РОЗКЛАД
# ──────────────────────────────────────────────
SCHEDULE = {
    9:  "daily_phrase",
    10: "grammar_quiz",
    11: "vocabulary_quiz",
    12: "situation_phrases",
    13: "confusing_words_quiz",
    14: "prepositions_quiz",
    15: "quote_motivation",
}

QUIZ_RUBRICS = {"grammar_quiz", "vocabulary_quiz", "confusing_words_quiz", "prepositions_quiz"}
IMAGE_RUBRICS = {"daily_phrase", "situation_phrases", "quote_motivation"}

# ──────────────────────────────────────────────
# SITUATION PHRASES — 18 КАТЕГОРІЙ (РОТАЦІЯ)
# ──────────────────────────────────────────────
SITUATION_CATEGORIES = [
    # Travel & Transport
    {"name": "At the Airport", "emoji": "✈️", "photo_query": "airplane sky clouds dramatic dusk", "description": "Check-in, boarding pass, gate, security, luggage"},
    {"name": "On the Plane", "emoji": "🛫", "photo_query": "airplane window sky clouds cinematic", "description": "Aisle seat, turbulence, flight attendant, requests on board"},
    {"name": "Train Station", "emoji": "🚂", "photo_query": "train tracks sunset landscape dramatic golden dusk", "description": "Platform, timetable, one-way, return ticket"},
    {"name": "Public Transport", "emoji": "🚌", "photo_query": "city street evening golden bokeh dramatic", "description": "Bus stop, fare, transfer, next stop"},
    {"name": "Taxi & Uber", "emoji": "🚕", "photo_query": "city night lights bokeh dramatic dusk", "description": "Drop me off, take me to, estimated time, tip"},
    {"name": "Car Rental", "emoji": "🚗", "photo_query": "open road landscape dramatic golden dusk", "description": "Insurance, fill up, mileage, deposit"},
    {"name": "Asking for Directions", "emoji": "🗺️", "photo_query": "city street golden hour bokeh dramatic", "description": "Turn left, go straight, crossroads, landmark"},
    {"name": "Border Control", "emoji": "🛂", "photo_query": "airport dramatic moody cinematic dusk", "description": "Passport, visa, customs, declaration"},
    {"name": "Booking a Trip Online", "emoji": "💻", "photo_query": "travel planning dramatic golden cinematic moody", "description": "Refundable, availability, confirmation, cancel"},
    {"name": "Travel Problems", "emoji": "⚠️", "photo_query": "airport dramatic moody dark cinematic", "description": "Delayed flight, lost luggage, compensation, claim"},
    {"name": "Road Trip", "emoji": "🛣️", "photo_query": "open road dramatic golden sunset cinematic", "description": "Route, pit stop, gas station, scenic route"},
    # Accommodation
    {"name": "Hotel Check-in", "emoji": "🏨", "photo_query": "hotel exterior dramatic golden dusk cinematic", "description": "Reservation, room key, checkout time, receipt"},
    {"name": "Hotel Problems", "emoji": "🔧", "photo_query": "hotel corridor dramatic moody cinematic", "description": "Complain, room upgrade, noisy, maintenance"},
    {"name": "Renting an Apartment", "emoji": "🏠", "photo_query": "apartment building exterior dramatic golden dusk", "description": "Lease, deposit, landlord, utilities"},
    # Food & Dining
    {"name": "Making a Reservation", "emoji": "📞", "photo_query": "restaurant terrace evening lights bokeh dusk", "description": "Book a table, for two, outdoor seating, occasion"},
    {"name": "Ordering Food", "emoji": "🍽️", "photo_query": "restaurant evening warm bokeh dramatic", "description": "I'll have, medium rare, vegan, gluten-free, allergy"},
    {"name": "At the Cafe", "emoji": "☕", "photo_query": "cafe evening bokeh warm dramatic moody", "description": "Flat white, oat milk, to go, pastry, decaf"},
    {"name": "The Bill & Tipping", "emoji": "💳", "photo_query": "restaurant evening golden bokeh cinematic", "description": "Split the bill, tip, service charge, receipt"},
    {"name": "Supermarket Shopping", "emoji": "🛒", "photo_query": "market outdoor golden dramatic dusk", "description": "Aisle, best before, organic, on sale, checkout"},
    # Work & Career
    {"name": "Job Interview", "emoji": "💼", "photo_query": "city skyline business dramatic golden dusk", "description": "Strengths, weaknesses, experience, salary expectations"},
    {"name": "Office Life", "emoji": "🏢", "photo_query": "city office building dramatic golden dusk", "description": "Colleagues, printer, break room, overtime"},
    {"name": "Business Meeting", "emoji": "📊", "photo_query": "city business district dramatic golden dusk", "description": "Agenda, minutes, action points, follow up"},
    {"name": "Remote Work", "emoji": "🖥️", "photo_query": "nature peaceful morning dramatic cinematic", "description": "Home office, video call, mute, screen share"},
    {"name": "Asking for a Day Off", "emoji": "📅", "photo_query": "nature park morning peaceful dramatic", "description": "Annual leave, sick day, approve, cover"},
    # Health
    {"name": "At the Doctor", "emoji": "🏥", "photo_query": "nature green healing morning peaceful dramatic", "description": "Symptoms, diagnosis, prescription, appointment"},
    {"name": "At the Pharmacy", "emoji": "💊", "photo_query": "green nature bokeh peaceful cinematic moody", "description": "Prescription, dosage, side effects, over the counter"},
    {"name": "Describing Symptoms", "emoji": "🤒", "photo_query": "nature peaceful green morning cinematic", "description": "Sore throat, headache, fever, dizzy"},
    {"name": "Emergency Services", "emoji": "🚨", "photo_query": "city night dramatic moody dark cinematic", "description": "Call 999/112, ambulance, fire brigade, police"},
    # Shopping
    {"name": "Clothes Shopping", "emoji": "👗", "photo_query": "shopping street evening golden bokeh dramatic", "description": "Size, fitting room, try on, exchange, refund"},
    {"name": "Returns & Refunds", "emoji": "🔄", "photo_query": "city street evening dramatic golden dusk", "description": "Receipt, exchange, faulty, guarantee"},
    {"name": "At the Market", "emoji": "🛍️", "photo_query": "outdoor market golden dramatic dusk cinematic", "description": "Fresh produce, bargain, vendor, cash only"},
    # Social
    {"name": "Meeting Someone New", "emoji": "🤝", "photo_query": "friends outdoor golden hour dramatic dusk", "description": "Nice to meet you, what do you do, where are you from"},
    {"name": "Small Talk", "emoji": "💬", "photo_query": "park golden hour dramatic bokeh cinematic", "description": "Weather, weekend plans, local area, how was your day"},
    {"name": "Making Plans", "emoji": "📆", "photo_query": "city park golden dramatic dusk bokeh", "description": "Are you free, shall we, how about, let's meet at"},
    {"name": "Invitations", "emoji": "🎉", "photo_query": "celebration bokeh night dramatic colorful", "description": "You're invited, I'd love to come, unfortunately I can't"},
    {"name": "Apologies", "emoji": "🙏", "photo_query": "peaceful nature morning dramatic cinematic", "description": "I'm so sorry, my mistake, I didn't mean to"},
    # Education
    {"name": "In the Classroom", "emoji": "📚", "photo_query": "autumn park golden dramatic dusk cinematic", "description": "Could you repeat, I don't understand, homework, exam"},
    {"name": "Talking to a Teacher", "emoji": "🎓", "photo_query": "autumn leaves golden dramatic moody cinematic", "description": "Could you explain, I need help with, deadline, grade"},
    {"name": "Study Abroad", "emoji": "🌍", "photo_query": "autumn park path golden dramatic dusk", "description": "Enrol, campus, lecture, assignment, student visa"},
    # Nature
    {"name": "Talking About Weather", "emoji": "🌤️", "photo_query": "dramatic sky clouds landscape dark moody", "description": "Forecast, temperature, humid, sunny spells, showers"},
    {"name": "Outdoor Activities", "emoji": "🏔️", "photo_query": "mountain forest dramatic moody cinematic", "description": "Hiking trail, picnic, campfire, wildlife, scenic view"},
    # Personal
    {"name": "Describing Yourself", "emoji": "😊", "photo_query": "nature peaceful morning dramatic cinematic moody", "description": "Personality, hobbies, background, hometown"},
    {"name": "Talking About Family", "emoji": "👨‍👩‍👧", "photo_query": "nature golden bokeh dramatic dusk cinematic", "description": "Siblings, extended family, childhood, upbringing"},
    # Finance
    {"name": "At the Bank", "emoji": "🏦", "photo_query": "city financial district dramatic golden dusk", "description": "Open account, transfer, exchange rate, ATM"},
    {"name": "Paying & Splitting Bills", "emoji": "💰", "photo_query": "city street evening dramatic golden bokeh", "description": "Split, go Dutch, contactless, cash back"},
    # Digital
    {"name": "Online Shopping", "emoji": "🛒", "photo_query": "technology bokeh abstract night dramatic", "description": "Add to cart, checkout, track order, return policy"},
    {"name": "Tech Problems", "emoji": "💻", "photo_query": "city night lights bokeh abstract dramatic", "description": "Frozen screen, update, reboot, password reset"},
    # Character
    {"name": "Expressing Feelings", "emoji": "💭", "photo_query": "nature peaceful moody dramatic cinematic", "description": "I feel overwhelmed, I'm excited, I'm a bit nervous"},
    {"name": "Giving Compliments", "emoji": "⭐", "photo_query": "golden hour nature bokeh dramatic cinematic", "description": "That looks great on you, well done, I'm impressed"},
]

# ──────────────────────────────────────────────
# DAILY PHRASE TOPICS — ротація тем з атласу
# ──────────────────────────────────────────────
DAILY_PHRASE_TOPICS = [
    {"name": "Travel Vocabulary", "desc": "General travel words and phrases for any journey"},
    {"name": "Airport Vocabulary", "desc": "Words related to flying, check-in, boarding"},
    {"name": "Hotel Vocabulary", "desc": "Useful words for hotel stays and accommodation"},
    {"name": "Food Vocabulary", "desc": "Names of foods, dishes, cooking methods, flavours"},
    {"name": "Cafe Phrases", "desc": "Common phrases for ordering coffee and snacks"},
    {"name": "Restaurant Phrases", "desc": "Useful expressions for dining out"},
    {"name": "Food Idioms", "desc": "Popular English idioms related to food: piece of cake, spill the beans"},
    {"name": "Food Trends", "desc": "Modern vocabulary: plant-based, keto, meal prep"},
    {"name": "Work Vocabulary", "desc": "General workplace words and expressions"},
    {"name": "Business English", "desc": "Professional phrases for meetings and emails"},
    {"name": "Job Interview Phrases", "desc": "Common expressions for job interviews"},
    {"name": "Health Vocabulary", "desc": "Words for describing health, symptoms, wellbeing"},
    {"name": "Wellbeing & Self-care", "desc": "Phrases about mental health, relaxation, balance"},
    {"name": "Shopping Vocabulary", "desc": "Useful words and phrases for shopping"},
    {"name": "Small Talk Phrases", "desc": "Casual conversation starters and social expressions"},
    {"name": "Polite Requests", "desc": "How to ask for things politely in English"},
    {"name": "Expressing Opinions", "desc": "Phrases for agreeing, disagreeing, giving views"},
    {"name": "Complimenting People", "desc": "How to give and receive compliments in English"},
    {"name": "Apologising", "desc": "Different ways to say sorry and make amends"},
    {"name": "Study Skills Vocabulary", "desc": "Words for learning, studying, taking notes"},
    {"name": "Classroom Language", "desc": "Phrases teachers and students use every day"},
    {"name": "Nature Vocabulary", "desc": "Words for describing the natural world"},
    {"name": "Weather Expressions", "desc": "Phrases for talking about weather and seasons"},
    {"name": "Seasons & Time", "desc": "Vocabulary for time of year, months, seasons"},
    {"name": "Positive Personality Traits", "desc": "Words like generous, patient, honest, supportive"},
    {"name": "Negative Traits", "desc": "Words like stubborn, jealous, bossy, unreliable"},
    {"name": "Body Language", "desc": "Phrases for describing gestures and non-verbal communication"},
    {"name": "Expressing Emotions", "desc": "How to describe feelings and emotional states"},
    {"name": "Growth Mindset", "desc": "Motivational vocabulary: resilience, comfort zone, discipline"},
    {"name": "Habits & Routines", "desc": "Phrases for daily routines, consistency, productivity"},
    {"name": "Ambitions & Goals", "desc": "Talking about dreams, plans and aspirations"},
    {"name": "Home Vocabulary", "desc": "Words for rooms, furniture, household items"},
    {"name": "Housework Phrases", "desc": "Vocabulary for cleaning, cooking, daily chores"},
    {"name": "Money Vocabulary", "desc": "Basic financial words: budget, savings, expenses"},
    {"name": "Banking Phrases", "desc": "Useful expressions for banking and transactions"},
    {"name": "Technology Vocabulary", "desc": "Modern tech words: update, cloud, backup, bluetooth"},
    {"name": "Social Media Language", "desc": "Words for posting, sharing, commenting online"},
    {"name": "Online Communication", "desc": "Phrases for emails, messages, video calls"},
    {"name": "Transport Vocabulary", "desc": "Words for different types of transport and travel"},
    {"name": "Driving Vocabulary", "desc": "Phrases for driving, directions, road signs"},
    {"name": "British vs American English", "desc": "Key vocabulary differences between UK and US"},
    {"name": "Common Idioms", "desc": "Popular English idioms and their meanings"},
    {"name": "Phrasal Verbs", "desc": "Common phrasal verbs used in everyday English"},
    {"name": "Slang & Informal English", "desc": "Modern casual English expressions"},
    {"name": "Sports Vocabulary", "desc": "Words and phrases related to popular sports"},
    {"name": "Music & Arts", "desc": "Vocabulary for talking about music, art, film"},
    {"name": "Books & Reading", "desc": "Phrases for discussing books and literature"},
    {"name": "Humor & Jokes", "desc": "Light-hearted phrases and expressions for fun"},
]


# ──────────────────────────────────────────────
# СВЯТА
# ──────────────────────────────────────────────
HOLIDAYS = {
    (1, 1):   {"name": "New Year", "emoji": "🎆", "photo_query": "fireworks night celebration dramatic colorful", "situation_name": "New Year", "situation_description": "New Year wishes, resolutions, countdown phrases"},
    (2, 14):  {"name": "Valentine's Day", "emoji": "💝", "photo_query": "valentines day hearts romantic bokeh dramatic", "situation_name": "Valentine's Day", "situation_description": "Romantic phrases, love expressions, Valentine's Day"},
    (3, 8):   {"name": "Women's Day", "emoji": "🌸", "photo_query": "spring flowers bokeh golden dramatic", "situation_name": "Women's Day", "situation_description": "Appreciation phrases, congratulations, Women's Day"},
    (4, 1):   {"name": "April Fool's Day", "emoji": "🃏", "photo_query": "playful colorful bokeh dramatic fun cinematic", "situation_name": "April Fool's Day", "situation_description": "Jokes, pranks, funny expressions in English"},
    (4, 22):  {"name": "Earth Day", "emoji": "🌍", "photo_query": "nature green dramatic moody forest cinematic", "situation_name": "Earth Day", "situation_description": "Environment phrases, recycling, saving the planet"},
    (10, 31): {"name": "Halloween", "emoji": "🎃", "photo_query": "halloween pumpkin moody dramatic night", "situation_name": "Halloween", "situation_description": "Halloween phrases, costumes, trick or treat"},
    (12, 25): {"name": "Christmas", "emoji": "🎄", "photo_query": "christmas lights bokeh night dramatic snow", "situation_name": "Christmas", "situation_description": "Christmas greetings, gift phrases, festive expressions"},
    (12, 31): {"name": "New Year's Eve", "emoji": "🥂", "photo_query": "fireworks celebration night dramatic colorful", "situation_name": "New Year's Eve", "situation_description": "Countdown phrases, toasts, New Year wishes"},
}

# ──────────────────────────────────────────────
# СЕЗОНИ ТА МІСЯЦІ — ФОНИ ДЛЯ DAILY PHRASE
# ──────────────────────────────────────────────
SEASON_PHOTOS = {
    "spring": ["cherry blossom branch bokeh dramatic", "spring meadow sunrise dramatic golden hour", "green forest path morning light"],
    "summer": ["tropical beach sunset aerial dramatic golden", "sunflower field golden hour", "mountain lake reflection summer"],
    "autumn": ["maple leaves golden light forest", "misty autumn forest path", "vineyard autumn sunset"],
    "winter": ["snow covered pine forest sunrise", "frozen lake mountains winter", "cozy snow landscape blue hour"],
}

MONTH_PHOTOS = {
    1:  ["frozen lake blue hour winter", "snowy mountain peaks sunrise"],
    2:  ["misty forest winter morning", "ice crystals macro photography"],
    3:  ["cherry blossom pink bokeh", "green shoots spring soil macro"],
    4:  ["tulip field aerial colorful", "spring rain drops petals macro"],
    5:  ["lavender field purple sunset", "green forest canopy aerial view"],
    6:  ["golden wheat field sunset", "ocean waves aerial blue"],
    7:  ["tropical lagoon turquoise aerial", "desert dunes golden hour"],
    8:  ["sunflower field horizon", "mountain sunset alpenglow"],
    9:  ["misty forest golden autumn", "harvest field warm light"],
    10: ["maple forest peak foliage aerial", "pumpkin field autumn fog"],
    11: ["foggy forest moody autumn", "rain drops window bokeh"],
    12: ["snowy forest blue hour dramatic moody", "frozen river winter landscape"],
}

ATMOSPHERE_PHOTOS = [
    "minimalist coffee cup overhead flat lay",
    "city skyline golden hour aerial dramatic",
    "misty forest path morning rays",
    "clean desk setup natural light minimal",
    "mountain sunrise horizon dramatic golden moody",
    "architectural minimal concrete geometry",
]

QUOTE_PHOTOS = [
    "misty forest sunbeams dramatic golden moody",
    "mountain peak sunrise alpenglow dramatic",
    "ocean horizon sunrise golden dramatic",
    "pine forest fog morning rays",
    "waterfall long exposure nature",
    "starry sky milky way mountains",
]


def get_season(month: int) -> str:
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"


def get_today_holiday() -> dict | None:
    today = date.today()
    return HOLIDAYS.get((today.month, today.day))


def get_photo_query_for_daily_phrase() -> str:
    holiday = get_today_holiday()
    if holiday:
        log.info(f"🎉 Holiday detected: {holiday['name']}")
        return holiday["photo_query"]

    today = date.today()
    season = get_season(today.month)

    roll = random.random()
    if roll < 0.70:
        query = random.choice(SEASON_PHOTOS[season])
        log.info(f"🌿 Season photo: {season} → {query}")
        return query
    elif roll < 0.85:
        query = random.choice(MONTH_PHOTOS[today.month])
        log.info(f"📅 Month photo: month={today.month} → {query}")
        return query
    else:
        query = random.choice(ATMOSPHERE_PHOTOS)
        log.info(f"☁️ Atmosphere photo → {query}")
        return query


def get_photo_query_for_situation(category: dict) -> str:
    holiday = get_today_holiday()
    if holiday:
        log.info(f"🎉 Holiday situation photo: {holiday['name']}")
        return holiday["photo_query"] + " celebration"
    return category["photo_query"]


def get_photo_query_for_quote() -> str:
    return random.choice(QUOTE_PHOTOS)


async def get_daily_phrase_topic(history_mgr) -> dict:
    """Вибирає випадкову тему з атласу, уникаючи повторів."""
    try:
        used_raw = await history_mgr.r.lrange("used:daily_phrase_topics", 0, -1)
        used_set = set(used_raw) if used_raw else set()
        available = [t for t in DAILY_PHRASE_TOPICS if t["name"] not in used_set]
        if not available:
            # Всі теми використані — скидаємо
            log.info("🔄 Daily phrase topics exhausted — resetting")
            await history_mgr.r.delete("used:daily_phrase_topics")
            available = DAILY_PHRASE_TOPICS
        topic = random.choice(available)
        await history_mgr.r.lpush("used:daily_phrase_topics", topic["name"])
        await history_mgr.r.ltrim("used:daily_phrase_topics", 0, len(DAILY_PHRASE_TOPICS) - 1)
        log.info(f"📖 Daily phrase topic: {topic['name']}")
        return topic
    except Exception as e:
        log.error(f"❌ get_daily_phrase_topic error: {e} — using random")
        return random.choice(DAILY_PHRASE_TOPICS)

# ──────────────────────────────────────────────
# UPSTASH REDIS
# ──────────────────────────────────────────────
class UpstashRedis:
    def __init__(self):
        self.url   = os.environ["UPSTASH_REDIS_REST_URL"].rstrip("/")
        self.token = os.environ["UPSTASH_REDIS_REST_TOKEN"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def _cmd(self, *args):
        cmd_url = self.url + "/" + "/".join(str(a) for a in args)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(cmd_url, headers=self.headers)
                resp.raise_for_status()
                return resp.json().get("result")
        except Exception as e:
            log.error(f"❌ Redis command {args[0]} failed: {e}")
            raise

    async def ping(self):
        return await self._cmd("ping")

    async def lrange(self, key: str, start: int, end: int) -> list:
        try:
            result = await self._cmd("lrange", key, start, end)
            return result or []
        except Exception:
            return []

    async def lpush(self, key: str, value: str) -> int:
        return await self._cmd("lpush", key, value)

    async def ltrim(self, key: str, start: int, end: int):
        return await self._cmd("ltrim", key, start, end)

    async def set(self, key: str, value: str, nx: bool = False, ex: int = None):
        parts = ["set", key, value]
        if nx:  parts.append("nx")
        if ex:  parts += ["ex", ex]
        return await self._cmd(*parts)

    async def get(self, key: str):
        try:
            return await self._cmd("get", key)
        except Exception:
            return None

    async def delete(self, key: str):
        try:
            return await self._cmd("del", key)
        except Exception:
            return None

    async def incr(self, key: str):
        return await self._cmd("incr", key)


class HistoryManager:
    def __init__(self, redis_client: UpstashRedis):
        self.r = redis_client
        self.max_history = 90

    async def get_used(self, rubric: str) -> list:
        key = f"used:{rubric}"
        try:
            items = await self.r.lrange(key, 0, -1)
            log.info(f"📚 History [{rubric}]: {len(items)} items loaded from Redis")
            return [item if isinstance(item, str) else str(item) for item in items]
        except Exception as e:
            log.error(f"❌ Redis get_used error for [{rubric}]: {e} — continuing without history")
            return []

    async def add_used(self, rubric: str, value: str):
        key = f"used:{rubric}"
        try:
            await self.r.lpush(key, value)
            await self.r.ltrim(key, 0, self.max_history - 1)
            log.info(f"📝 History [{rubric}]: added '{value[:60]}'")
        except Exception as e:
            log.error(f"❌ Redis add_used error for [{rubric}]: {e}")

    async def acquire_lock(self, rubric: str, ttl: int = 300) -> bool:
        key = f"lock:{rubric}"
        try:
            result = await self.r.set(key, "1", nx=True, ex=ttl)
            acquired = result == "OK"
            log.info(f"🔒 Lock [{rubric}]: {'acquired' if acquired else 'already locked'}")
            return acquired
        except Exception as e:
            log.error(f"❌ Redis lock error for [{rubric}]: {e} — allowing execution")
            return True

    async def release_lock(self, rubric: str):
        key = f"lock:{rubric}"
        try:
            await self.r.delete(key)
            log.info(f"🔓 Lock [{rubric}]: released")
        except Exception as e:
            log.error(f"❌ Redis release_lock error for [{rubric}]: {e}")

    async def get_situation_index(self) -> int:
        try:
            val = await self.r.get("situation:rotation_index")
            idx = int(val) if val is not None else 0
            log.info(f"🔄 Situation rotation index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_situation_index error: {e} — using index 0")
            return 0

    async def advance_situation_index(self):
        try:
            current = await self.get_situation_index()
            next_idx = (current + 1) % len(SITUATION_CATEGORIES)
            await self.r.set("situation:rotation_index", str(next_idx))
            log.info(f"🔄 Situation index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_situation_index error: {e}")

# ──────────────────────────────────────────────
# ФОТО API
# ──────────────────────────────────────────────
async def fetch_photo_unsplash(query: str, use_topics: bool = True) -> str | None:
    if not UNSPLASH_ACCESS_KEY:
        log.warning("⚠️ UNSPLASH_ACCESS_KEY not set")
        return None
    try:
        # Отримуємо 5 фото і вибираємо найкраще за лайками
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "orientation": "portrait",
            "content_filter": "high",
            "per_page": 10,
        }
        if use_topics:
            params["collections"] = "bo8jQKTaE0Y,6sMVjTLSkeQ"
        headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            log.info(f"📷 Unsplash response: status={resp.status_code} query='{query}'")
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("results", [])
                if not photos:
                    log.warning(f"⚠️ Unsplash: no results for query '{query}'")
                    return None
                # Вибираємо фото з найбільшою кількістю лайків з перших 5
                best = max(photos[:5], key=lambda p: p.get("likes", 0))
                photo_url = best["urls"]["regular"]
                log.info(f"✅ Unsplash best photo: likes={best.get('likes',0)} url={photo_url[:60]}")
                return photo_url
            else:
                log.warning(f"⚠️ Unsplash failed: {resp.status_code} — {resp.text[:200]}")
                return None
    except Exception as e:
        log.error(f"❌ Unsplash exception: {e}")
        return None


async def fetch_photo_pexels(query: str) -> str | None:
    if not PEXELS_API_KEY:
        log.warning("⚠️ PEXELS_API_KEY not set")
        return None
    try:
        url = "https://api.pexels.com/v1/search"
        params = {
            "query": query,
            "orientation": "portrait",
            "per_page": 10,
            "size": "large",
        }
        headers = {"Authorization": PEXELS_API_KEY}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            log.info(f"📷 Pexels response: status={resp.status_code} query='{query}'")
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("photos", [])
                if photos:
                    photo = random.choice(photos[:5])
                    photo_url = photo["src"]["large"]
                    log.info(f"✅ Pexels photo found: {photo_url[:80]}")
                    return photo_url
                else:
                    log.warning(f"⚠️ Pexels: no photos found for query '{query}'")
                    return None
            else:
                log.warning(f"⚠️ Pexels failed: {resp.status_code} — {resp.text[:200]}")
                return None
    except Exception as e:
        log.error(f"❌ Pexels exception: {e}")
        return None


async def fetch_photo_pixabay(query: str) -> str | None:
    if not PIXABAY_API_KEY:
        log.warning("⚠️ PIXABAY_API_KEY not set")
        return None
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "orientation": "vertical",
            "image_type": "photo",
            "per_page": 10,
            "min_width": 1080,
            "safesearch": "true",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            log.info(f"📷 Pixabay response: status={resp.status_code} query='{query}'")
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", [])
                if hits:
                    photo = random.choice(hits[:5])
                    photo_url = photo["largeImageURL"]
                    log.info(f"✅ Pixabay photo found: {photo_url[:80]}")
                    return photo_url
                else:
                    log.warning(f"⚠️ Pixabay: no photos found for query '{query}'")
                    return None
            else:
                log.warning(f"⚠️ Pixabay failed: {resp.status_code} — {resp.text[:200]}")
                return None
    except Exception as e:
        log.error(f"❌ Pixabay exception: {e}")
        return None


async def fetch_photo(query: str, use_topics: bool = True) -> str | None:
    """Пробує Unsplash → Pexels → Pixabay. Повертає URL або None."""
    log.info(f"🔍 Fetching photo for query: '{query}' use_topics={use_topics}")

    # Спроба 1: Unsplash
    for attempt in range(1, 4):
        photo_url = await fetch_photo_unsplash(query, use_topics=use_topics)
        if photo_url:
            return photo_url
        log.warning(f"⚠️ Unsplash attempt {attempt}/3 failed")
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)

    # Спроба 2: Pexels
    for attempt in range(1, 4):
        photo_url = await fetch_photo_pexels(query)
        if photo_url:
            return photo_url
        log.warning(f"⚠️ Pexels attempt {attempt}/3 failed")
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)

    # Спроба 3: Pixabay
    for attempt in range(1, 4):
        photo_url = await fetch_photo_pixabay(query)
        if photo_url:
            return photo_url
        log.warning(f"⚠️ Pixabay attempt {attempt}/3 failed")
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)

    log.error(f"❌ ALL photo APIs failed for query '{query}' — skipping post")
    return None


async def download_photo(url: str) -> bytes | None:
    """Завантажує фото за URL і повертає bytes."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                size_kb = len(resp.content) / 1024
                log.info(f"✅ Photo downloaded: {size_kb:.0f} KB from {url[:60]}")
                return resp.content
            else:
                log.error(f"❌ Photo download failed: status={resp.status_code} url={url[:60]}")
                return None
    except Exception as e:
        log.error(f"❌ Photo download exception: {e} url={url[:60]}")
        return None

# ──────────────────────────────────────────────
# МОВНИЙ ЦЕНЗОР — СИСТЕМНИЙ ПРОМПТ
# ──────────────────────────────────────────────
LANGUAGE_CENSOR = """
Ukrainian Language Quality Rules (MUST FOLLOW):
- Use only authentic, natural Ukrainian language (not a word-for-word translation)
- Strictly avoid Russian-isms (surzhyk) and Russian grammar patterns
- Avoid the word "ви" — use natural verb endings instead
- Ensure the tone is friendly and educational
- Sound like a native speaker from Kyiv, not a machine translation
- Self-Correction step: First translate to Ukrainian, then re-read and ensure it sounds natural
- Fix any unnatural phrases before giving the final result
"""

# ──────────────────────────────────────────────
# ПРОМПТИ
# ──────────────────────────────────────────────
def get_prompt(rubric: str, used_history: list, extra: dict = None) -> str:
    history_note = (
        f"\nDo NOT repeat these recent topics/phrases: {used_history[-20:]}\n"
        if used_history else ""
    )
    extra = extra or {}

    if rubric == "daily_phrase":
        topic_name = extra.get("topic_name", "")
        topic_desc = extra.get("topic_desc", "")
        topic_line = f"Topic: {topic_name} — {topic_desc}\n" if topic_name else ""
        return f"""You are an English teacher. Generate a useful English phrase for A2 level students.
{topic_line}{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "phrase_en": "the English phrase related to the topic (minimum 5 words, max 80 characters)",
  "example_en": "one natural example sentence using the phrase in context (max 140 characters)",
  "example_ua": "Ukrainian translation of the example sentence (max 140 characters)",
  "photo_query": "3-5 keywords for stock photo: scene + mood + style (cinematic dramatic moody soft light)"
}}
Rules:
- Phrase must relate to the given topic
- Minimum 5 words — avoid very short phrases like "See you" or "Thank you"
- Simple A2 vocabulary, natural everyday conversation
- photo_query: visual scene related to the topic, add: cinematic dramatic soft light
{LANGUAGE_CENSOR}"""

    if rubric == "situation_phrases":
        situation_name = extra.get("situation_name", "Daily Life")
        situation_desc = extra.get("situation_description", "everyday situations")
        return f"""You are an English teacher. Generate 5 useful English phrases for A2 level students for this situation: {situation_name} — {situation_desc}
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "phrases": [
    {{"en": "english phrase (max 70 chars)", "ua": "ukrainian translation (max 80 chars)"}},
    {{"en": "english phrase (max 70 chars)", "ua": "ukrainian translation (max 80 chars)"}},
    {{"en": "english phrase (max 70 chars)", "ua": "ukrainian translation (max 80 chars)"}},
    {{"en": "english phrase (max 70 chars)", "ua": "ukrainian translation (max 80 chars)"}},
    {{"en": "english phrase (max 70 chars)", "ua": "ukrainian translation (max 80 chars)"}}
  ],
  "photo_query": "3-5 keywords for stock photo: scene + mood + style (minimal aesthetic cinematic soft light)"
}}
Rules:
- Practical A2 level phrases for the situation
- Each phrase must be different and useful
- photo_query: visual scene related to the situation, add: minimal aesthetic cinematic soft light
- photo_query examples: "airport runway night dramatic cinematic", "restaurant evening bokeh soft light aesthetic"
{LANGUAGE_CENSOR}"""

    if rubric == "quote_motivation":
        return f"""You are an English teacher. Generate a short motivational phrase for A2 level English students.
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "quote_en": "motivational phrase in English (minimum 6 words, maximum 12 words, simple A2 vocabulary)",
  "quote_ua": "Ukrainian translation (natural, not word-for-word)",
  "photo_query": "3-5 keywords for dark moody nature photo: forest, mountains, ocean, waterfall or green nature (NO sunrise, NO sunset, NO cities) + style (moody dramatic cinematic)"
}}
Rules:
- Minimum 6 words — NEVER generate phrases like 'Just do it', 'Dream big', 'Stay strong' (too short)
- Maximum 12 words
- Simple A2 vocabulary, positive and inspiring
- Good examples: 'Every day is a new chance to grow.' (9 words)
- Self-Correction: Make sure the phrase is at least 6 words long
- photo_query: ALWAYS use dark/moody nature scenes — never cities, people, sunrise or sunset
- NO author needed — this is original motivation, not a quote
{LANGUAGE_CENSOR}"""

    if rubric == "grammar_quiz":
        return f"""You are an English teacher. Create a grammar quiz question for A2 level students.
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Grammar question with a blank ___ (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation why this answer is correct (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- Always provide exactly 4 options
- correct_index is 0-based (0=A, 1=B, 2=C, 3=D)
- Test common A2 grammar: tenses, articles, prepositions with adjectives
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "vocabulary_quiz":
        return f"""You are an English teacher. Create a vocabulary quiz question for A2 level students.
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Vocabulary question with a blank ___ (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation of the correct word meaning (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- correct_index is 0-based
- Test everyday A2 vocabulary in context sentences
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "confusing_words_quiz":
        return f"""You are an English teacher. Create a quiz about commonly confused English words for A2 level students.
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Sentence with a blank ___ testing confusing words (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation of the difference (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- Test confusing pairs: look/see/watch, make/do, say/tell, bring/take, etc.
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "prepositions_quiz":
        return f"""You are an English teacher. Create a prepositions quiz question for A2 level students.
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Sentence with a missing preposition ___ (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation why this preposition is correct (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- Test common prepositions: in/on/at, to/for/of, with/by/from, etc.
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    raise ValueError(f"Unknown rubric: {rubric}")

# ──────────────────────────────────────────────
# GEMINI / GROQ — ГЕНЕРАЦІЯ КОНТЕНТУ
# ──────────────────────────────────────────────
CRITICAL_ERRORS = {
    "INVALID_ARGUMENT", "API_KEY_INVALID", "PERMISSION_DENIED",
    "invalid_api_key", "authentication_failed", "account_suspended",
}


def is_critical_error(error_text: str) -> bool:
    for ce in CRITICAL_ERRORS:
        if ce.lower() in error_text.lower():
            return True
    return False


async def call_gemini(api_key: str, prompt: str, model: str = "gemini-2.0-flash-lite") -> dict:
    """Викликає Gemini модель з конкретним API ключем."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": 1000},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, params={"key": api_key})
            key_hint = api_key[:8] + "..."
            log.info(f"🤖 Gemini {model} [{key_hint}]: status={resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                result = json.loads(text)
                log.info(f"✅ Gemini {model} [{key_hint}]: success")
                return result

            error_text = resp.text
            log.warning(f"⚠️ Gemini {model} [{key_hint}] HTTP {resp.status_code}: {error_text[:200]}")

            if is_critical_error(error_text):
                log.error(f"❌ Gemini [{key_hint}]: CRITICAL error — {error_text[:100]}")
                raise RuntimeError(f"CRITICAL: {error_text[:100]}")

            # 429 або 5xx — просто повертаємо None, спробуємо наступний ключ
            raise RuntimeError(f"HTTP {resp.status_code}")

    except RuntimeError:
        raise
    except json.JSONDecodeError as e:
        log.error(f"❌ Gemini [{api_key[:8]}...] JSON parse error: {e}")
        raise RuntimeError(f"JSON error: {e}")
    except Exception as e:
        log.error(f"❌ Gemini [{api_key[:8]}...] exception: {e}")
        raise RuntimeError(f"Exception: {e}")







async def generate_content(rubric: str, history: list, extra: dict = None) -> dict:
    prompt = get_prompt(rubric, history, extra)
    log.info(f"🤖 Generating content for [{rubric}] | keys available: {len(GEMINI_API_KEYS)}")

    if not GEMINI_API_KEYS:
        log.error("❌ No Gemini API keys configured! Set GEMINI_API_KEY_1/2/3")
        raise RuntimeError("No Gemini API keys configured")

    for model in GEMINI_MODELS:
        for i, api_key in enumerate(GEMINI_API_KEYS, 1):
            try:
                log.info(f"🔑 Trying {model} key {i}/{len(GEMINI_API_KEYS)} for [{rubric}]")
                result = await call_gemini(api_key, prompt, model)
                log.info(f"✅ Content generated via {model} key {i} for [{rubric}]")
                return result
            except RuntimeError as e:
                if "CRITICAL" in str(e):
                    # Невалідний ключ — пропускаємо тільки цей ключ, не зупиняємо все
                    log.error(f"❌ Gemini {model} key {i} invalid/critical: {e} — skipping this key")
                    continue
                log.warning(f"⚠️ {model} key {i} failed for [{rubric}]: {e} — trying next")
            except Exception as e:
                log.error(f"❌ Unexpected error {model} key {i} for [{rubric}]: {e}")

    log.error(f"❌ All Gemini models+keys failed for [{rubric}] — SKIPPING POST")
    raise RuntimeError(f"All Gemini models failed for [{rubric}]")

# ──────────────────────────────────────────────
# HTML ШАБЛОНИ — GLASSMORPHISM
# ──────────────────────────────────────────────
def html_base(photo_b64: str, content_blocks: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ width: 1080px; height: 1920px; overflow: hidden; font-family: 'Montserrat', sans-serif; position: relative; }}
.bg {{ position: absolute; top: 0; left: 0; width: 1080px; height: 1920px;
  background-image: url('data:image/jpeg;base64,{photo_b64}');
  background-size: cover; background-position: center; z-index: 0; }}
.bg-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: linear-gradient(to bottom, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.15) 40%, rgba(0,0,0,0.55) 100%);
  z-index: 1; }}
.content {{ position: absolute; top: 0; left: 0; width: 1080px; height: 1920px;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  padding: 80px 64px 200px 64px; gap: 44px; z-index: 5; }}
.glass-block {{ width: 100%; background: rgba(12, 12, 18, 0.58);
  backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
  border-radius: 32px; padding: 52px 60px;
  border: 1px solid rgba(201,168,76,0.25); box-shadow: 0 8px 32px rgba(0,0,0,0.30); }}
.brand {{ font-size: 28px; font-weight: 300; letter-spacing: 2px;
  color: rgba(245,245,247,0.75); text-shadow: 0 2px 8px rgba(0,0,0,0.85); }}
.brand .eng {{ color: #c9a84c; }}
.gold-line {{ width: 120px; height: 2px;
  background: linear-gradient(90deg, transparent, #c9a84c, transparent); margin: 24px 0; }}
.bottom-bar {{ position: absolute; bottom: 0; left: 0; right: 0; z-index: 6;
  padding: 36px 70px 44px 70px;
  background: rgba(0,0,0,0.35); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="bg-overlay"></div>
<div class="content">{content_blocks}</div>
</body>
</html>"""


def build_daily_phrase(data: dict, photo_b64: str) -> str:
    phrase = data.get("phrase_en", "")
    ex_en  = data.get("example_en", "")
    ex_ua  = data.get("example_ua", "")
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    blocks = f"""
  <div class="glass-block" style="height:380px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:flex-start; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:28px; font-weight:600; letter-spacing:3px; color:#c9a84c;
                text-transform:uppercase; margin-bottom:28px; {ts}">
      Daily Phrase
    </div>
    <div style="font-size:clamp(46px,5vw,64px); font-weight:700; color:#ffffff;
                {ts} line-height:1.25; flex:1; display:flex; align-items:center;">
      {phrase}
    </div>
  </div>
  <div class="glass-block" style="height:550px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:clamp(42px,4.5vw,58px); font-weight:600; color:#ffffff;
                {ts} line-height:1.3; margin-bottom:4px;">
      {ex_en}
    </div>
    <div class="gold-line"></div>
    <div style="font-size:clamp(40px,4.3vw,56px); font-weight:300; color:rgba(245,245,247,0.85);
                {ts2} line-height:1.3; flex:1; display:flex; align-items:center;">
      {ex_ua}
    </div>
    <div style="text-align:right; margin-top:20px;">
      <span class="brand">Improve Your <span class="eng">English</span></span>
    </div>
  </div>"""
    return html_base(photo_b64, blocks)


def build_situation_phrases(data: dict, photo_b64: str, category: dict) -> str:
    phrases    = data.get("phrases", [])
    topic_name = category.get("name", "")
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    block_height, font_en, font_ua = 280, 50, 44
    log.info(f"📐 Situation: fixed height={block_height}px")

    # Тема над блоками без розмиття
    topic_header = f"""
  <div style="width:100%; text-align:left; padding:0 8px; margin-bottom:4px;">
    <div style="font-size:68px; font-weight:600; color:#c9a84c;
                letter-spacing:2px; {ts} line-height:1.1;">
      {topic_name}
    </div>
  </div>"""

    blocks = topic_header
    for p in phrases[:5]:
        en = p.get("en", "")
        ua = p.get("ua", "")
        blocks += f"""
  <div class="glass-block" style="height:{block_height}px; padding:24px 52px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:{font_en}px; font-weight:700; color:#ffffff;
                {ts} line-height:1.25; margin-bottom:12px;">{en}</div>
    <div style="font-size:{font_ua}px; font-weight:300; color:rgba(245,245,247,0.82);
                {ts2} line-height:1.25;">{ua}</div>
  </div>"""

    html = html_base(photo_b64, blocks)
    html = html.replace("gap: 44px;", "gap: 20px;", 1)

    # Бренд внизу з blur
    bottom = '''<div class="bottom-bar" style="text-align:right; background:transparent; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);">
      <span class="brand">Improve Your <span class="eng">English</span></span>
    </div>'''
    html = html.replace("</body>", bottom + "</body>")
    return html


def build_quote_motivation(data: dict, photo_b64: str) -> str:
    quote_en = data.get("quote_en", "")
    quote_ua = data.get("quote_ua", "")
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    blocks = f"""
  <div class="glass-block" style="height:520px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:flex-start; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:28px; font-weight:600; letter-spacing:3px; color:#c9a84c;
                text-transform:uppercase; margin-bottom:28px; {ts}">
      Motivation
    </div>
    <div style="font-size:clamp(46px,5vw,64px); font-weight:700; color:#ffffff;
                {ts} line-height:1.3; flex:1; display:flex; align-items:center;">
      {quote_en}
    </div>
    <div class="gold-line" style="margin:20px 0 0 0;"></div>
  </div>
  <div class="glass-block" style="height:520px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:clamp(46px,5vw,64px); font-weight:300; color:rgba(245,245,247,0.88);
                {ts2} line-height:1.3; flex:1; display:flex; align-items:center;">
      {quote_ua}
    </div>
    <div style="text-align:right; margin-top:20px;">
      <span class="brand">Improve Your <span class="eng">English</span></span>
    </div>
  </div>"""
    return html_base(photo_b64, blocks)

# ──────────────────────────────────────────────
# PLAYWRIGHT — HTML → PNG
# ──────────────────────────────────────────────
async def render_card(html: str) -> bytes:
    log.info("🎨 Starting Playwright render...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        try:
            page = await browser.new_page(viewport={"width": 1080, "height": 1920})
            await page.set_content(html, wait_until="networkidle")
            await asyncio.sleep(1.0)
            png_bytes = await page.screenshot(type="png", full_page=False)
        finally:
            await browser.close()

    size_mb = len(png_bytes) / 1024 / 1024
    log.info(f"📸 PNG rendered: {size_mb:.2f} MB")
    if size_mb > 10:
        log.warning(f"⚠️ PNG too large ({size_mb:.2f} MB) — Telegram may reject")
    return png_bytes


# ──────────────────────────────────────────────
# TELEGRAM — ПУБЛІКАЦІЯ
# ──────────────────────────────────────────────
async def send_photo_to_telegram(png_bytes: bytes, rubric: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files={"photo": (f"{rubric}.png", png_bytes, "image/png")},
                )
                log.info(f"📤 Telegram sendPhoto attempt {attempt}: status={resp.status_code}")
                if resp.status_code == 200:
                    log.info(f"✅ Telegram: photo sent [{rubric}]")
                    return True
                else:
                    log.error(f"❌ Telegram sendPhoto failed: {resp.status_code} — {resp.text[:300]}")
        except Exception as e:
            log.error(f"❌ Telegram sendPhoto exception attempt {attempt}: {e}")
        if attempt < 3:
            await asyncio.sleep(5 * attempt)
    log.error(f"❌ Telegram: photo send failed for [{rubric}] after 3 attempts")
    return False


async def send_quiz_to_telegram(data: dict, rubric: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"

    # Gemini Flash Lite іноді повертає список замість об'єкта
    if isinstance(data, list):
        log.warning(f"⚠️ Quiz data is list, taking first element for [{rubric}]")
        data = data[0] if data else {}

    question   = data.get("question", "")
    options    = data.get("options", [])
    correct    = data.get("correct_index", 0)
    explanation = data.get("explanation_ua", "")

    if not question or not options or len(options) < 2:
        log.error(f"❌ Quiz data invalid for [{rubric}]: question='{question}' options={options}")
        return False

    payload = {
        "chat_id":         TELEGRAM_CHAT_ID,
        "question":        question,
        "options":         json.dumps(options),
        "type":            "quiz",
        "correct_option_id": int(correct),
        "is_anonymous":    True,
        "explanation":     explanation if explanation else "",
    }

    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, data=payload)
                log.info(f"📤 Telegram sendPoll attempt {attempt}: status={resp.status_code}")
                if resp.status_code == 200:
                    log.info(f"✅ Telegram: quiz sent [{rubric}]")
                    return True
                else:
                    log.error(f"❌ Telegram sendPoll failed: {resp.status_code} — {resp.text[:300]}")
        except Exception as e:
            log.error(f"❌ Telegram sendPoll exception attempt {attempt}: {e}")
        if attempt < 3:
            await asyncio.sleep(5 * attempt)

    log.error(f"❌ Telegram: quiz send failed for [{rubric}] after 3 attempts")
    return False

# ──────────────────────────────────────────────
# ГОЛОВНІ ФУНКЦІЇ ПУБЛІКАЦІЇ
# ──────────────────────────────────────────────
async def publish_image_card(rubric: str, redis_client: UpstashRedis):
    history_mgr = HistoryManager(redis_client)
    start_time  = time.time()
    log.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(f"🚀 START image [{rubric}] at {datetime.now().strftime('%H:%M:%S')}")

    if not await history_mgr.acquire_lock(rubric):
        log.warning(f"⚠️ Lock exists for [{rubric}] — skipping (already running)")
        return

    try:
        # 1. Визначаємо фото query та extra дані
        extra = {}
        photo_query = ""
        category = None

        if rubric == "daily_phrase":
            photo_query = get_photo_query_for_daily_phrase()
            daily_topic = await get_daily_phrase_topic(history_mgr)
            extra = {"topic_name": daily_topic["name"], "topic_desc": daily_topic["desc"]}

        elif rubric == "situation_phrases":
            holiday = get_today_holiday()
            if holiday:
                log.info(f"🎉 Holiday [{holiday['name']}] — using holiday situation")
                extra = {
                    "situation_name": holiday["situation_name"],
                    "situation_description": holiday["situation_description"],
                }
                photo_query = get_photo_query_for_situation({})
                category = {
                    "name": holiday["situation_name"],
                    "emoji": "",
                }
                # Індекс не рухаємо в день свята
            else:
                idx = await history_mgr.get_situation_index()
                category = SITUATION_CATEGORIES[idx]
                extra = {
                    "situation_name": category["name"],
                    "situation_description": category["description"],
                }
                photo_query = get_photo_query_for_situation(category)
                await history_mgr.advance_situation_index()

        elif rubric == "quote_motivation":
            photo_query = get_photo_query_for_quote()

        log.info(f"🔍 Photo query for [{rubric}]: '{photo_query}'")

        # 2. Завантажуємо фото
        use_topics = rubric != "situation_phrases"
        photo_url = await fetch_photo(photo_query, use_topics=use_topics)
        if not photo_url:
            log.error(f"❌ No photo available for [{rubric}] — SKIPPING POST")
            return

        photo_bytes = await download_photo(photo_url)
        if not photo_bytes:
            log.error(f"❌ Photo download failed for [{rubric}] — SKIPPING POST")
            return

        import base64
        photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")
        log.info(f"✅ Photo ready for [{rubric}]: {len(photo_bytes)//1024}KB")

        # 3. Генеруємо контент
        history = await history_mgr.get_used(rubric)
        log.info(f"🤖 Generating content for [{rubric}]...")
        data = await generate_content(rubric, history, extra)
        log.info(f"✅ Content: {json.dumps(data, ensure_ascii=False)[:200]}")

        # Якщо Gemini повернув кращий photo_query — оновлюємо фото
        ai_photo_query = data.get("photo_query", "").strip()
        if ai_photo_query and ai_photo_query != photo_query:
            log.info(f"🎨 AI photo query: '{ai_photo_query}' — fetching better photo")
            better_url = await fetch_photo(ai_photo_query, use_topics=use_topics)
            if better_url:
                better_bytes = await download_photo(better_url)
                if better_bytes:
                    photo_b64 = base64.b64encode(better_bytes).decode("utf-8")
                    log.info(f"✅ Better photo loaded: {len(better_bytes)//1024}KB")

        # 4. Будуємо HTML
        if rubric == "daily_phrase":
            html = build_daily_phrase(data, photo_b64)
        elif rubric == "situation_phrases":
            cat = category or SITUATION_CATEGORIES[0]
            html = build_situation_phrases(data, photo_b64, cat)
        elif rubric == "quote_motivation":
            html = build_quote_motivation(data, photo_b64)
        else:
            log.error(f"❌ Unknown image rubric: {rubric}")
            return

        # 5. Рендеримо PNG
        log.info(f"🎨 Rendering PNG for [{rubric}]...")
        png_bytes = await render_card(html)

        # 6. Публікуємо
        log.info(f"📤 Sending to Telegram [{rubric}]...")
        success = await send_photo_to_telegram(png_bytes, rubric)

        # 7. Зберігаємо в історію
        if success:
            history_key = json.dumps(data, ensure_ascii=False)[:100]
            await history_mgr.add_used(rubric, history_key)

        elapsed = time.time() - start_time
        log.info(f"⏱️ [{rubric}] completed in {elapsed:.1f}s | success={success}")

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR in [{rubric}]: {e}", exc_info=True)
    finally:
        await history_mgr.release_lock(rubric)


async def publish_quiz(rubric: str, redis_client: UpstashRedis):
    history_mgr = HistoryManager(redis_client)
    start_time  = time.time()
    log.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(f"🚀 START quiz [{rubric}] at {datetime.now().strftime('%H:%M:%S')}")

    if not await history_mgr.acquire_lock(rubric):
        log.warning(f"⚠️ Lock exists for [{rubric}] — skipping")
        return

    try:
        # 1. Генеруємо контент
        history = await history_mgr.get_used(rubric)
        log.info(f"🤖 Generating quiz for [{rubric}]...")
        data = await generate_content(rubric, history)
        log.info(f"✅ Quiz data: {json.dumps(data, ensure_ascii=False)[:200]}")

        # 2. Публікуємо
        success = await send_quiz_to_telegram(data, rubric)

        # 3. Зберігаємо в історію
        if success:
            history_key = data.get("question", "")[:100]
            await history_mgr.add_used(rubric, history_key)

        elapsed = time.time() - start_time
        log.info(f"⏱️ [{rubric}] completed in {elapsed:.1f}s | success={success}")

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR in quiz [{rubric}]: {e}", exc_info=True)
    finally:
        await history_mgr.release_lock(rubric)


async def publish_card(rubric: str, redis_client: UpstashRedis):
    if rubric in QUIZ_RUBRICS:
        await publish_quiz(rubric, redis_client)
    else:
        await publish_image_card(rubric, redis_client)


# ──────────────────────────────────────────────
# HTTP СЕРВЕР — health check + /test/{rubric}
# ──────────────────────────────────────────────
_redis_client_global: "UpstashRedis | None" = None

VALID_RUBRICS = list(SCHEDULE.values())

RUBRIC_HELP = "\n".join(f"  /test/{r}" for r in VALID_RUBRICS)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.rstrip("/")

        # GET / — health check
        if path == "" or path == "/":
            self.send_response(200)
            self.end_headers()
            body = "English A2 Bot v2.0 - OK\n\nTest endpoints:\n" + RUBRIC_HELP + "\n"
            self.wfile.write(body.encode("utf-8"))
            return

        # GET /test/{rubric}
        if path.startswith("/test/"):
            rubric = path[6:]  # все після /test/
            if rubric not in VALID_RUBRICS:
                self.send_response(400)
                self.end_headers()
                msg = f"Unknown rubric: '{rubric}'\nValid rubrics:\n{RUBRIC_HELP}\n"
                self.wfile.write(msg.encode())
                log.warning(f"⚠️ /test/ called with unknown rubric: '{rubric}'")
                return

            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"⏳ Triggering [{rubric}]... check Telegram!\n".encode())
            log.info(f"🧪 Manual test triggered via HTTP: [{rubric}]")

            # Запускаємо publish в asyncio event loop
            if _redis_client_global is not None:
                asyncio.run_coroutine_threadsafe(
                    publish_card(rubric, _redis_client_global),
                    _loop_global,
                )
            else:
                log.error("❌ Redis client not initialized yet — cannot trigger test")
            return

        # Все інше — 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found\n")

    def log_message(self, format, *args):
        pass  # вимикаємо стандартні HTTP логи


_loop_global: asyncio.AbstractEventLoop | None = None


def start_health_server():
    port   = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    log.info(f"🌐 HTTP server running on port {port}")
    log.info(f"🧪 Test endpoints available: /test/{{rubric}}")
    server.serve_forever()


# ──────────────────────────────────────────────
# ПЛАНУВАЛЬНИК
# ──────────────────────────────────────────────
async def scheduler(redis_client: UpstashRedis):
    log.info("⏰ Scheduler started")
    published_today: set = set()

    while True:
        now  = datetime.now()
        hour = now.hour

        if hour == 0 and now.minute == 0:
            published_today.clear()
            log.info("🔄 Reset published_today for new day")

        if hour in SCHEDULE and hour not in published_today:
            rubric = SCHEDULE[hour]
            published_today.add(hour)
            log.info(f"⏰ Triggering [{rubric}] at {now.strftime('%H:%M')}")
            asyncio.create_task(publish_card(rubric, redis_client))

        await asyncio.sleep(60)


# ──────────────────────────────────────────────
# ТОЧКА ВХОДУ
# ──────────────────────────────────────────────
async def main():
    global _redis_client_global, _loop_global

    log.info("🤖 English A2 Bot v2.0 starting...")
    log.info(f"📋 Schedule: {SCHEDULE}")
    log.info(f"🖼️ Image rubrics: {IMAGE_RUBRICS}")
    log.info(f"📝 Quiz rubrics: {QUIZ_RUBRICS}")
    log.info(f"🧪 Test endpoints: /test/<rubric> — valid: {VALID_RUBRICS}")

    redis_client = UpstashRedis()
    try:
        result = await redis_client.ping()
        log.info(f"✅ Redis connected: ping={result}")
    except Exception as e:
        log.error(f"❌ Redis connection failed: {e}")
        raise

    # Зберігаємо глобально для HTTP handler
    _redis_client_global = redis_client
    _loop_global = asyncio.get_event_loop()

    thread = threading.Thread(target=start_health_server, daemon=True)
    thread.start()
    log.info("🌐 Health server thread started")

    await scheduler(redis_client)


if __name__ == "__main__":
    # Chromium встановлено під час Docker білду
    log.info("🚀 Starting bot (Chromium pre-installed in Docker image)")
    asyncio.run(main())
