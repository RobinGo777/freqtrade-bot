"""
English A2 Telegram Bot — v2.0
Glassmorphism дизайн + Native Telegram Quiz + Розумна система фонів
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo
from io import BytesIO
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from playwright.async_api import async_playwright

# ──────────────────────────────────────────────
# КОНФІГУРАЦІЯ / ВАЛІДАЦІЯ
# ──────────────────────────────────────────────
def _get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

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
TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
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
ELEVENLABS_API_KEY   = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID  = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
# Google Cloud TTS: шлях до JSON service account (у контейнері — secret mount)
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

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
# РОЗКЛАД (усі тригери — локальний час Europe/Kyiv)
# ──────────────────────────────────────────────
TZ = ZoneInfo("Europe/Kyiv")

# Пн–Пт: 10:00 — «великий» пост; 11:30, 13:00, 14:30, 16:00 — квізи (фіксований порядок щодня)
QUIZ_SLOTS: list[tuple[tuple[int, int], str]] = [
    ((11, 30), "grammar_quiz"),
    ((13, 0), "vocabulary_quiz"),
    ((14, 30), "confusing_words_quiz"),
    ((16, 0), "prepositions_quiz"),
]

# Пн–Пт о 10:00 Europe/Kyiv. weekday: 0=Пн … 4=Пт (datetime.weekday())
WEEKDAY_10_00: dict[int, str] = {
    0: "quote_motivation",   # Пн
    1: "vocabulary_15",      # Вт 10:00 — 15 слів + IPA (кремова картка)
    2: "daily_phrase",       # Ср 10:00 — фраза дня + приклад (en) + переклад (ua); build_daily_phrase + Unsplash
    3: "situation_phrases",  # Чт 10:00 — 5 фраз (en/ua) для життєвої ситуації; ротація SITUATION_CATEGORIES або тема свята; build_situation_phrases + Unsplash
    4: "photo_relax",        # Пт 10:00 — чисте фото PNG; 4 речення en A2 у caption; Unsplash→Pexels→Pixabay
}

# Сб / Нд о 16:00 — окремі рубрики (контент TBD)
WEEKEND_16_00: dict[int, str] = {
    5: "interesting_cities",
    6: "travel_video",
}

# Нд 16:00 — travel_video: сток-відео + Gemini + TTS + FFmpeg (9:16), без UNESCO
TRAVEL_VIDEO_LANDMARK_CATEGORIES: list[str] = [
    "Natural wonders",
    "Historic & cultural sites",
    "Urban & modern icons",
    "Cultural & symbolic places",
    "Tourist & resort destinations",
    "Unique or extreme places",
]
TRAVEL_VIDEO_MAIN_MAX_SEC = 58.0
TRAVEL_VIDEO_BRAND_SEC = 2.5
TRAVEL_VIDEO_PIPELINE_ATTEMPTS = 3
# Pexels часто дає 4K — декод + буфери дають OOM на малих контейнерах; беремо ≤ цієї довгої сторони, якщо є
TRAVEL_VIDEO_PEXELS_MAX_LONG_EDGE = 1920
TRAVEL_VIDEO_NARRATION_WORDS_MAX = 160

INTERESTING_CITIES_SENTENCES_MIN = 3
INTERESTING_CITIES_SENTENCES_MAX = 5

PLACEHOLDER_RUBRICS = frozenset()


def get_rubric_for_datetime(now: datetime) -> str | None:
    """Повертає рубрику для моменту `now` (має бути aware у TZ) або None."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    else:
        now = now.astimezone(TZ)

    wd = now.weekday()
    h, m = now.hour, now.minute

    if wd <= 4:
        if (h, m) == (10, 0):
            return WEEKDAY_10_00[wd]
        for (hh, mm), rubric in QUIZ_SLOTS:
            if (h, m) == (hh, mm):
                return rubric
        return None

    if wd in WEEKEND_16_00 and (h, m) == (16, 0):
        return WEEKEND_16_00[wd]
    return None


QUIZ_RUBRICS = {"grammar_quiz", "vocabulary_quiz", "confusing_words_quiz", "prepositions_quiz"}
IMAGE_RUBRICS = {
    "daily_phrase",
    "situation_phrases",
    "quote_motivation",
    "vocabulary_15",
    "photo_relax",
    "interesting_cities",
}

# photo_relax: ротація тем пошуку фото + контекст для Gemini (без дощу/вечора як орієнтирів)
PHOTO_RELAX_THEMES: list[dict] = [
    {"id": "sea", "label": "the sea or coastline (water, horizon, open or calm sea)", "photo_query": "ocean sea coast nature landscape"},
    {"id": "forest", "label": "a forest or woodland (trees, path, green light)", "photo_query": "forest trees nature path sunlight"},
    {"id": "mountains", "label": "mountains or hills (peaks, ridges, distance)", "photo_query": "mountain peaks landscape nature alpine"},
    {"id": "lake", "label": "a lake (still water, shore, reflections)", "photo_query": "lake mountains reflection nature landscape"},
    {"id": "waterfall", "label": "a waterfall (water, rocks, mist)", "photo_query": "waterfall nature forest rocks"},
]

# Стиль тексту для п’ятничного photo_relax (ротація окремо від теми фото)
PHOTO_RELAX_VOICE_STYLES: list[dict] = [
    {
        "id": "romantic_poetic",
        "instruction": (
            "Voice: romantic / gently poetic. Use simple imagery or a soft metaphor; keep sentences easy to read. "
            "Sound intimate and unhurried, never grand or theatrical."
        ),
    },
    {
        "id": "minimal_modern",
        "instruction": (
            "Voice: minimal / modern. Four short sentences, very lean wording, stylish and a little reflective. "
            "No filler; each line should feel clean and quiet."
        ),
    },
    {
        "id": "instagram_lifestyle",
        "instruction": (
            "Voice: Instagram / lifestyle. Warm and close, as if you are sharing a real moment with friends. "
            "Natural, conversational, a bit upbeat about slowing down."
        ),
    },
    {
        "id": "calm_therapeutic",
        "instruction": (
            "Voice: calm / grounding. Soft reminder to breathe, slow down, and feel a little lighter after the week. "
            "Gentle, reassuring, like a kind note to yourself — not clinical."
        ),
    },
    {
        "id": "escape_travel",
        "instruction": (
            "Voice: small escape / wanderlust. Light wish to pause routine, change the view, or feel a bit of freedom. "
            "Hopeful and light, not a travel brochure."
        ),
    },
]

# ──────────────────────────────────────────────
# SITUATION PHRASES — 18 КАТЕГОРІЙ (РОТАЦІЯ)
# ──────────────────────────────────────────────
SITUATION_CATEGORIES = [
    # Travel & Transport
    {"name": "At the Airport", "emoji": "✈️", "photo_query": "airplane sky clouds", "description": "Check-in, boarding pass, gate, security, luggage"},
    {"name": "On the Plane", "emoji": "🛫", "photo_query": "airplane window sky clouds cinematic", "description": "Aisle seat, turbulence, flight attendant, requests on board"},
    {"name": "Train Station", "emoji": "🚂", "photo_query": "train tracks sunset landscape", "description": "Platform, timetable, one-way, return ticket"},
    {"name": "Public Transport", "emoji": "🚌", "photo_query": "city street evening golden bokeh", "description": "Bus stop, fare, transfer, next stop"},
    {"name": "Taxi & Uber", "emoji": "🚕", "photo_query": "city night lights bokeh", "description": "Drop me off, take me to, estimated time, tip"},
    {"name": "Car Rental", "emoji": "🚗", "photo_query": "open road landscape", "description": "Insurance, fill up, mileage, deposit"},
    {"name": "Asking for Directions", "emoji": "🗺️", "photo_query": "city street golden hour bokeh", "description": "Turn left, go straight, crossroads, landmark"},
    {"name": "Border Control", "emoji": "🛂", "photo_query": "airport cinematic", "description": "Passport, visa, customs, declaration"},
    {"name": "Booking a Trip Online", "emoji": "💻", "photo_query": "travel planning", "description": "Refundable, availability, confirmation, cancel"},
    {"name": "Travel Problems", "emoji": "⚠️", "photo_query": "airport dark cinematic", "description": "Delayed flight, lost luggage, compensation, claim"},
    {"name": "Road Trip", "emoji": "🛣️", "photo_query": "open road sunset cinematic", "description": "Route, pit stop, gas station, scenic route"},
    # Accommodation
    {"name": "Hotel Check-in", "emoji": "🏨", "photo_query": "hotel exterior cinematic", "description": "Reservation, room key, checkout time, receipt"},
    {"name": "Hotel Problems", "emoji": "🔧", "photo_query": "hotel corridor cinematic", "description": "Complain, room upgrade, noisy, maintenance"},
    {"name": "Renting an Apartment", "emoji": "🏠", "photo_query": "apartment building exterior", "description": "Lease, deposit, landlord, utilities"},
    # Food & Dining
    {"name": "Making a Reservation", "emoji": "📞", "photo_query": "restaurant terrace evening lights bokeh", "description": "Book a table, for two, outdoor seating, occasion"},
    {"name": "Ordering Food", "emoji": "🍽️", "photo_query": "restaurant evening warm bokeh", "description": "I'll have, medium rare, vegan, gluten-free, allergy"},
    {"name": "At the Cafe", "emoji": "☕", "photo_query": "cafe evening bokeh warm", "description": "Flat white, oat milk, to go, pastry, decaf"},
    {"name": "The Bill & Tipping", "emoji": "💳", "photo_query": "restaurant evening golden bokeh cinematic", "description": "Split the bill, tip, service charge, receipt"},
    {"name": "Supermarket Shopping", "emoji": "🛒", "photo_query": "market outdoor golden", "description": "Aisle, best before, organic, on sale, checkout"},
    # Work & Career
    {"name": "Job Interview", "emoji": "💼", "photo_query": "city skyline business", "description": "Strengths, weaknesses, experience, salary expectations"},
    {"name": "Office Life", "emoji": "🏢", "photo_query": "city office building", "description": "Colleagues, printer, break room, overtime"},
    {"name": "Business Meeting", "emoji": "📊", "photo_query": "city business district", "description": "Agenda, minutes, action points, follow up"},
    {"name": "Remote Work", "emoji": "🖥️", "photo_query": "nature peaceful morning", "description": "Home office, video call, mute, screen share"},
    {"name": "Asking for a Day Off", "emoji": "📅", "photo_query": "nature park morning peaceful", "description": "Annual leave, sick day, approve, cover"},
    # Health
    {"name": "At the Doctor", "emoji": "🏥", "photo_query": "nature green healing morning peaceful", "description": "Symptoms, diagnosis, prescription, appointment"},
    {"name": "At the Pharmacy", "emoji": "💊", "photo_query": "green nature bokeh peaceful", "description": "Prescription, dosage, side effects, over the counter"},
    {"name": "Describing Symptoms", "emoji": "🤒", "photo_query": "nature peaceful green morning cinematic", "description": "Sore throat, headache, fever, dizzy"},
    {"name": "Emergency Services", "emoji": "🚨", "photo_query": "city night dark cinematic", "description": "Call 999/112, ambulance, fire brigade, police"},
    # Shopping
    {"name": "Clothes Shopping", "emoji": "👗", "photo_query": "shopping street evening golden bokeh", "description": "Size, fitting room, try on, exchange, refund"},
    {"name": "Returns & Refunds", "emoji": "🔄", "photo_query": "city street evening", "description": "Receipt, exchange, faulty, guarantee"},
    {"name": "At the Market", "emoji": "🛍️", "photo_query": "outdoor market golden cinematic", "description": "Fresh produce, bargain, vendor, cash only"},
    # Social
    {"name": "Meeting Someone New", "emoji": "🤝", "photo_query": "friends outdoor golden hour", "description": "Nice to meet you, what do you do, where are you from"},
    {"name": "Small Talk", "emoji": "💬", "photo_query": "park golden hour bokeh cinematic", "description": "Weather, weekend plans, local area, how was your day"},
    {"name": "Making Plans", "emoji": "📆", "photo_query": "city park golden bokeh", "description": "Are you free, shall we, how about, let's meet at"},
    {"name": "Invitations", "emoji": "🎉", "photo_query": "celebration bokeh night colorful", "description": "You're invited, I'd love to come, unfortunately I can't"},
    {"name": "Apologies", "emoji": "🙏", "photo_query": "peaceful nature morning", "description": "I'm so sorry, my mistake, I didn't mean to"},
    # Education
    {"name": "In the Classroom", "emoji": "📚", "photo_query": "autumn park golden cinematic", "description": "Could you repeat, I don't understand, homework, exam"},
    {"name": "Talking to a Teacher", "emoji": "🎓", "photo_query": "autumn leaves golden cinematic", "description": "Could you explain, I need help with, deadline, grade"},
    {"name": "Study Abroad", "emoji": "🌍", "photo_query": "autumn park path golden", "description": "Enrol, campus, lecture, assignment, student visa"},
    # Nature
    {"name": "Talking About Weather", "emoji": "🌤️", "photo_query": "dramatic sky clouds landscape", "description": "Forecast, temperature, humid, sunny spells, showers"},
    {"name": "Outdoor Activities", "emoji": "🏔️", "photo_query": "mountain forest cinematic", "description": "Hiking trail, picnic, campfire, wildlife, scenic view"},
    # Personal
    {"name": "Describing Yourself", "emoji": "😊", "photo_query": "nature peaceful morning", "description": "Personality, hobbies, background, hometown"},
    {"name": "Talking About Family", "emoji": "👨‍👩‍👧", "photo_query": "nature golden bokeh cinematic", "description": "Siblings, extended family, childhood, upbringing"},
    # Finance
    {"name": "At the Bank", "emoji": "🏦", "photo_query": "city financial district", "description": "Open account, transfer, exchange rate, ATM"},
    {"name": "Paying & Splitting Bills", "emoji": "💰", "photo_query": "city street evening bokeh", "description": "Split, go Dutch, contactless, cash back"},
    # Digital
    {"name": "Online Shopping", "emoji": "🛒", "photo_query": "technology bokeh abstract night", "description": "Add to cart, checkout, track order, return policy"},
    {"name": "Tech Problems", "emoji": "💻", "photo_query": "city night lights bokeh abstract", "description": "Frozen screen, update, reboot, password reset"},
    # Character
    {"name": "Expressing Feelings", "emoji": "💭", "photo_query": "nature peaceful cinematic", "description": "I feel overwhelmed, I'm excited, I'm a bit nervous"},
    {"name": "Giving Compliments", "emoji": "⭐", "photo_query": "golden hour nature bokeh", "description": "That looks great on you, well done, I'm impressed"},
]

ATLAS_TOPIC_NAMES = [
    "At the Airport", "On the Plane", "Train Station", "Public Transport", "Taxi & Uber",
    "Car Rental", "Asking for Directions", "Border Control", "Booking a Trip Online",
    "Travel Problems", "Road Trip", "Hotel Check-in/out", "Hotel Problems",
    "Renting an Apartment", "Housework", "Home Appliances", "Describing Your Home",
    "Neighbours & Community", "Making a Reservation", "Ordering Food", "At the Cafe",
    "The Bill & Tipping", "Cooking & Recipes", "Supermarket Shopping", "Food Vocabulary",
    "Food Trends", "Food Idioms", "Job Interview", "Office Life", "Meetings",
    "Emails & Calls", "Sick Leave", "Freelance & Remote Work", "CV & Cover Letter",
    "Networking", "Work Phrasal Verbs", "At the Doctor", "At the Pharmacy",
    "Healthy Lifestyle", "Mental Health", "At the Dentist", "Body Parts & Injuries",
    "Gym & Fitness", "Health Idioms", "Clothes Shopping", "Beauty Salon",
    "Banking & Money", "At the Post Office", "Electronics & Tech", "Online Shopping",
    "Bargaining & Sales", "Small Talk", "Hobbies", "Cinema & Theatre",
    "Parties & Invitations", "Compliments", "Apologies & Conflicts", "Sports & Games",
    "Music & Concerts", "Books & Reading", "Language Learning", "In the Classroom",
    "Exams & Tests", "University Life", "IELTS & Cambridge", "Online Learning",
    "Study Tips & Strategies", "Weather Forecast", "Animals & Pets", "Environment",
    "City vs Countryside", "Natural Disasters", "Seasons & Nature Walks", "Appearance",
    "Character", "Family", "Feelings", "Dating", "Friendship",
    "Cultural Differences", "Leaking Tap", "Power Cut", "Locksmith",
    "Furniture Assembly", "Interior Design", "Moving House", "Budgeting", "Taxes",
    "Investment", "Scams & Safety", "Insurance", "Renting vs Buying", "Social Media",
    "Online Dating", "Cyberbullying", "Artificial Intelligence", "Streaming Services",
    "Tech Vocabulary", "Podcasts & YouTube", "Digital Etiquette", "Positive Traits",
    "Negative Traits", "Body Language", "Expressing Anger", "Ambitions",
    "Growth Mindset", "Habits & Routines", "Inspirational Quotes",
]


def _slug_to_description(name: str) -> str:
    return f"Useful A2 phrases and real-life mini-dialogues for topic: {name}"


def _sync_situation_categories_with_atlas(existing: list[dict], atlas_names: list[str]) -> list[dict]:
    # Зберігаємо наявні категорії (з їх photo_query/emoji/description), але доводимо список до 1:1 з atlas.
    by_name = {str(item.get("name", "")).strip(): item for item in existing if item.get("name")}
    merged = []
    for name in atlas_names:
        if name in by_name:
            merged.append(by_name[name])
            continue
        merged.append(
            {
                "name": name,
                "emoji": "🧩",
                "photo_query": "cinematic realistic scene soft light",
                "description": _slug_to_description(name),
            }
        )
    return merged


SITUATION_CATEGORIES = _sync_situation_categories_with_atlas(SITUATION_CATEGORIES, ATLAS_TOPIC_NAMES)

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

ATLAS_SYNC_TOPICS = [
    {"name": "Hotel Check-in/out", "desc": "Check-in and check-out flow, reception communication"},
    {"name": "Housework", "desc": "Cleaning, laundry, chores and home routine phrases"},
    {"name": "Home Appliances", "desc": "Vocabulary for household devices and maintenance"},
    {"name": "Describing Your Home", "desc": "Describe rooms, furniture, layout, and comfort"},
    {"name": "Neighbours & Community", "desc": "Neighborhood communication and community life"},
    {"name": "Cooking & Recipes", "desc": "Ingredients, cooking actions, and recipe language"},
    {"name": "Meetings", "desc": "Meeting structure, agenda, and participation language"},
    {"name": "Emails & Calls", "desc": "Formal and semi-formal communication for work"},
    {"name": "Sick Leave", "desc": "Requesting leave, reporting illness, and updates"},
    {"name": "Freelance & Remote Work", "desc": "Remote routines, clients, and deadlines"},
    {"name": "CV & Cover Letter", "desc": "Job application documents and key phrases"},
    {"name": "Networking", "desc": "Professional introductions and relationship building"},
    {"name": "Work Phrasal Verbs", "desc": "High-frequency phrasal verbs in work context"},
    {"name": "Healthy Lifestyle", "desc": "Wellness, habits, sleep, movement, and nutrition"},
    {"name": "Mental Health", "desc": "Stress, emotions, recovery, and self-support language"},
    {"name": "At the Dentist", "desc": "Dental appointments, symptoms, and treatment"},
    {"name": "Body Parts & Injuries", "desc": "Body vocabulary, pain description, and injuries"},
    {"name": "Gym & Fitness", "desc": "Workout language, routines, and fitness goals"},
    {"name": "Health Idioms", "desc": "Common idioms used in health and wellness context"},
    {"name": "Beauty Salon", "desc": "Hair, beauty services, and appointment phrases"},
    {"name": "Banking & Money", "desc": "Bank operations, cards, transfers, and statements"},
    {"name": "At the Post Office", "desc": "Parcels, delivery, tracking, and postal services"},
    {"name": "Electronics & Tech", "desc": "Devices, specs, setup, and troubleshooting basics"},
    {"name": "Bargaining & Sales", "desc": "Discounts, offers, negotiation, and price talk"},
    {"name": "Hobbies", "desc": "Interests, leisure activities, and preference language"},
    {"name": "Cinema & Theatre", "desc": "Movies, seats, showtimes, and review language"},
    {"name": "Parties & Invitations", "desc": "Inviting, accepting, declining, and planning events"},
    {"name": "Compliments", "desc": "Polite praise and positive feedback in conversation"},
    {"name": "Apologies & Conflicts", "desc": "Apologizing, clarifying, and resolving tension"},
    {"name": "Sports & Games", "desc": "Sports vocabulary, rules, and match discussion"},
    {"name": "Music & Concerts", "desc": "Genres, performances, and live event language"},
    {"name": "Language Learning", "desc": "Learning methods, progress, and language goals"},
    {"name": "Exams & Tests", "desc": "Preparation, test-taking, and results language"},
    {"name": "University Life", "desc": "Campus, classes, assignments, and student life"},
    {"name": "IELTS & Cambridge", "desc": "Exam prep vocabulary and task-specific phrases"},
    {"name": "Online Learning", "desc": "Platforms, self-study flow, and course language"},
    {"name": "Study Tips & Strategies", "desc": "Memory methods, planning, and discipline"},
    {"name": "Weather Forecast", "desc": "Forecast terms and seasonal weather descriptions"},
    {"name": "Animals & Pets", "desc": "Pet care, behavior, and vet communication"},
    {"name": "Environment", "desc": "Sustainability, ecology, and climate language"},
    {"name": "City vs Countryside", "desc": "Compare urban and rural lifestyles"},
    {"name": "Natural Disasters", "desc": "Emergency and disaster-related vocabulary"},
    {"name": "Seasons & Nature Walks", "desc": "Nature descriptions and seasonal observation"},
    {"name": "Appearance", "desc": "Looks, style, and physical description language"},
    {"name": "Character", "desc": "Personality traits and behavior descriptions"},
    {"name": "Family", "desc": "Family roles, relationships, and traditions"},
    {"name": "Feelings", "desc": "Emotion vocabulary and self-expression"},
    {"name": "Dating", "desc": "Social-romantic communication and etiquette"},
    {"name": "Friendship", "desc": "Supportive communication in friendships"},
    {"name": "Cultural Differences", "desc": "Cross-cultural communication and etiquette"},
    {"name": "Leaking Tap", "desc": "Home repair and plumber communication"},
    {"name": "Power Cut", "desc": "Power outage vocabulary and problem reporting"},
    {"name": "Locksmith", "desc": "Lost keys, lock issues, and urgent requests"},
    {"name": "Furniture Assembly", "desc": "Assembly instructions and household setup"},
    {"name": "Interior Design", "desc": "Home style, furniture, and color descriptions"},
    {"name": "Moving House", "desc": "Packing, moving logistics, and settling in"},
    {"name": "Budgeting", "desc": "Budget planning, expenses, and saving habits"},
    {"name": "Taxes", "desc": "Basic tax-related terms and official communication"},
    {"name": "Investment", "desc": "Basic investing vocabulary and risk discussion"},
    {"name": "Scams & Safety", "desc": "Fraud prevention and digital safety language"},
    {"name": "Insurance", "desc": "Policies, claims, and coverage communication"},
    {"name": "Renting vs Buying", "desc": "Housing choice discussion and finance terms"},
    {"name": "Social Media", "desc": "Posting, engagement, and online interactions"},
    {"name": "Online Dating", "desc": "Profiles, messaging, and boundaries online"},
    {"name": "Cyberbullying", "desc": "Online safety, reporting, and supportive responses"},
    {"name": "Artificial Intelligence", "desc": "AI tools, prompts, and practical usage terms"},
    {"name": "Streaming Services", "desc": "Subscriptions, content, and viewing habits"},
    {"name": "Tech Vocabulary", "desc": "Core modern digital and software terminology"},
    {"name": "Podcasts & YouTube", "desc": "Listening and video-learning related language"},
    {"name": "Digital Etiquette", "desc": "Polite online behavior and communication norms"},
    {"name": "Positive Traits", "desc": "Describing strengths and positive character features"},
    {"name": "Expressing Anger", "desc": "Assertive language for frustration and boundaries"},
    {"name": "Ambitions", "desc": "Long-term goals and motivational communication"},
    {"name": "Inspirational Quotes", "desc": "Quote language, interpretation, and reflection"},
]


def _merge_topics_unique(base: list, extra: list) -> list:
    seen = set()
    merged = []
    for item in base + extra:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


# Автосинхронізація з atlas: розширюємо покриття тем без дублювань.
DAILY_PHRASE_TOPICS = _merge_topics_unique(DAILY_PHRASE_TOPICS, ATLAS_SYNC_TOPICS)


# ──────────────────────────────────────────────
# СВЯТА
# ──────────────────────────────────────────────
HOLIDAYS = {
    (1, 1):   {"name": "New Year", "emoji": "🎆", "photo_query": "fireworks night celebration colorful", "situation_name": "New Year", "situation_description": "New Year wishes, resolutions, countdown phrases"},
    (2, 14):  {"name": "Valentine's Day", "emoji": "💝", "photo_query": "valentines day hearts romantic bokeh", "situation_name": "Valentine's Day", "situation_description": "Romantic phrases, love expressions, Valentine's Day"},
    (3, 8):   {"name": "Women's Day", "emoji": "🌸", "photo_query": "spring flowers bokeh", "situation_name": "Women's Day", "situation_description": "Appreciation phrases, congratulations, Women's Day"},
    (4, 1):   {"name": "April Fool's Day", "emoji": "🃏", "photo_query": "playful colorful bokeh fun cinematic", "situation_name": "April Fool's Day", "situation_description": "Jokes, pranks, funny expressions in English"},
    (4, 20):  {"name": "Easter", "emoji": "🐣", "photo_query": "easter spring nature soft light", "situation_name": "Easter", "situation_description": "Easter traditions, greetings, family celebration"},
    (4, 22):  {"name": "Earth Day", "emoji": "🌍", "photo_query": "nature green forest cinematic", "situation_name": "Earth Day", "situation_description": "Environment phrases, recycling, saving the planet"},
    (5, 12):  {"name": "Mother's Day", "emoji": "💐", "photo_query": "flowers spring warm natural light", "situation_name": "Mother's Day", "situation_description": "Family gratitude, appreciation, kind wishes for mothers"},
    (6, 15):  {"name": "Father's Day", "emoji": "👔", "photo_query": "family outdoors warm natural light", "situation_name": "Father's Day", "situation_description": "Respect, gratitude, and wishes for fathers"},
    (6, 21):  {"name": "International Yoga Day", "emoji": "🧘", "photo_query": "calm green nature mindfulness", "situation_name": "International Yoga Day", "situation_description": "Mindfulness, wellness, and healthy routine phrases"},
    (7, 31):  {"name": "International Friendship Day", "emoji": "🤝", "photo_query": "friends outdoors soft golden light", "situation_name": "International Friendship Day", "situation_description": "Friendship, support, and appreciation phrases"},
    (9, 1):   {"name": "Back to School", "emoji": "🎒", "photo_query": "school campus autumn soft light", "situation_name": "Back to School", "situation_description": "Study motivation, classroom language, and school routines"},
    (10, 31): {"name": "Halloween", "emoji": "🎃", "photo_query": "halloween pumpkin night", "situation_name": "Halloween", "situation_description": "Halloween phrases, costumes, trick or treat"},
    (11, 28): {"name": "Thanksgiving", "emoji": "🦃", "photo_query": "cozy dinner table autumn warm light", "situation_name": "Thanksgiving", "situation_description": "Gratitude, family dinner, and thankful expressions"},
    (12, 25): {"name": "Christmas", "emoji": "🎄", "photo_query": "christmas lights bokeh night snow", "situation_name": "Christmas", "situation_description": "Christmas greetings, gift phrases, festive expressions"},
    (12, 31): {"name": "New Year's Eve", "emoji": "🥂", "photo_query": "fireworks celebration night colorful", "situation_name": "New Year's Eve", "situation_description": "Countdown phrases, toasts, New Year wishes"},
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

DAILY_PHRASE_SEASON_NATURE_PHOTOS = {
    "spring": [
        "green spring forest path soft fog cinematic",
        "fresh spring meadow wildflowers soft morning light",
        "river in spring forest lush greenery moody",
    ],
    "summer": [
        "deep green summer forest trail cinematic",
        "mountain lake summer nature calm overcast",
        "green valley summer clouds atmospheric",
    ],
    "autumn": [
        "autumn forest path golden leaves misty mood",
        "mountain forest autumn fog cinematic",
        "rainy autumn woodland moody nature",
    ],
    "winter": [
        "winter pine forest soft snow overcast cinematic",
        "frozen lake winter mountains moody atmosphere",
        "snowy forest trail calm winter nature",
    ],
}

DAILY_PHRASE_MONTH_NATURE_PHOTOS = {
    1:  ["snowy pine forest blue winter haze", "frozen river valley winter overcast"],
    2:  ["misty winter forest soft grey light", "late winter mountain forest cinematic"],
    3:  ["early spring buds forest path fresh green", "spring rain forest moss cinematic"],
    4:  ["spring meadow flowers soft clouds nature", "fresh green park trees after rain"],
    5:  ["lush green forest canopy soft light", "mountain valley spring greenery atmospheric"],
    6:  ["summer forest shadows calm natural light", "green hills and lake early summer"],
    7:  ["deep summer pine forest moody cinematic", "mountain lake reflections summer clouds"],
    8:  ["late summer green valley haze cinematic", "forest lake calm end of summer mood"],
    9:  ["early autumn forest green-gold transition", "misty september woodland trail"],
    10: ["autumn foliage forest cinematic overcast", "foggy october forest path moody"],
    11: ["late autumn bare forest mist grey mood", "rainy november woodland cinematic"],
    12: ["first snow pine forest winter calm", "december mountain forest overcast atmosphere"],
}

DAILY_PHRASE_MONTH_STAGE_HINTS = {
    "early": ["fresh", "new month energy", "soft morning"],
    "mid": ["steady", "balanced calm", "natural rhythm"],
    "late": ["reflective", "deep focus", "quiet mood"],
}

SITUATION_SEASON_STYLE = {
    "spring": [
        "soft spring atmosphere",
        "fresh green tones",
        "gentle natural light",
    ],
    "summer": [
        "clear summer mood",
        "warm balanced light",
        "clean cinematic contrast",
    ],
    "autumn": [
        "cozy autumn mood",
        "golden-brown tones",
        "soft misty atmosphere",
    ],
    "winter": [
        "cool winter mood",
        "deep calm contrast",
        "soft overcast light",
    ],
}

SITUATION_MONTH_STAGE_STYLE = {
    "early": ["fresh start", "light dynamic feel"],
    "mid": ["stable rhythm", "balanced composition"],
    "late": ["reflective mood", "calm deeper tone"],
}

# Теми з атласу без власного photo_query отримують generic — він часто дає сірі «чорно-білі» урбан-стоки.
SITUATION_GENERIC_PHOTO_QUERY = "cinematic realistic scene soft light"

# Базові запити для тем, де generic виглядає однаково (офіс, листи, зустрічі).
SITUATION_TOPIC_PHOTO_BASE_OVERRIDES: dict[str, str] = {
    "Emails & Calls": "warm desk lamp laptop screen morning office coffee cozy natural light colorful",
    "Meetings": "modern conference room natural daylight indoor plants collaboration warm wood tones colorful",
    "Office Life": "bright office interior warm natural window light plants laptop colorful not grey",
    "Job Interview": "professional office lobby warm natural light modern interior soft color",
    "Freelance & Remote Work": "cozy home office warm desk lamp plants natural light colorful",
    "Networking": "coffee shop meeting warm bokeh people blurred background colorful candid",
    "CV & Cover Letter": "minimal desk notebook pen warm wood natural light soft color",
    "Sick Leave": "peaceful bedroom window soft morning light cozy warm tones",
    "Work Phrasal Verbs": "casual office teamwork warm light collaborative colorful",
    "Digital Etiquette": "smartphone laptop desk warm evening light cozy colorful screen glow",
    "Online Communication": "video call laptop warm ring light home office cozy natural color",
}

# Ротація настрою кольору (Unsplash краще знаходить насичені кадри, ніж «minimal»).
SITUATION_WARM_COLOR_MOODS = [
    "warm golden amber tones vibrant color",
    "soft teal and amber cinematic color grading cozy",
    "sunlit interior warm natural palette rich color",
    "evening city lights warm bokeh colorful not monochrome",
    "morning soft fog pastel sky warm highlights",
    "cozy indoor plants and wood warm earthy color",
    "late afternoon honey light soft contrast colorful",
]

# Суфікс без «minimal» — він тягнув до плоских сірих фонів.
SITUATION_PHOTO_QUERY_SUFFIX = (
    "cinematic soft light copy space soft bokeh warm natural colors rich tones vibrant not monochrome no text"
)

QUOTE_PHOTOS = [
    "green mountain lake overcast cinematic",
    "waterfall rocks green forest cinematic",
    "wild river through pine forest moody cinematic",
    "rocky sea coastline dramatic nature",
    "deep green canyon river aerial cinematic",
    "forest lake reflections dark clouds cinematic",
    "ocean cliffs waves dramatic moody nature",
    "alpine lake shoreline pine trees cinematic",
]

QUOTE_THEMES = [
    {
        "name": "Stoicism & Resilience",
        "description": "Focus on what we can control, strength in difficulty, obstacles into opportunities.",
    },
    {
        "name": "Deep Action & Discipline",
        "description": "Act today, action over overthinking, daily consistency and compounding habits.",
    },
    {
        "name": "Self-Actualization",
        "description": "Authenticity, courage to be yourself, develop unique talents, ignore social noise.",
    },
    {
        "name": "Mindfulness & Presence",
        "description": "Living in the present, gratitude, mental hygiene, less anxiety about future.",
    },
    {
        "name": "Intellectual Growth",
        "description": "Love of learning, curiosity, language as a tool to understand the world.",
    },
    {
        "name": "Contribution & Impact",
        "description": "Helping others, personal growth that improves the world around you.",
    },
]

WEAK_QUOTE_PATTERNS = (
    "just do it",
    "dream big",
    "stay strong",
    "never give up",
    "keep going",
    "believe in yourself",
    "be positive",
    "you can do it",
)

PREPOSITIONS_SUBTYPES = [
    {
        "name": "Time Prepositions",
        "description": "in/on/at for time + during/for/since/until/by/within",
        "examples": "in 2026, on Monday, at 7 PM, during the lesson, since 2020",
    },
    {
        "name": "Place & Position Prepositions",
        "description": "in/on/at for location + under/over/between/among/next to/opposite/behind/in front of",
        "examples": "in the box, on the table, at the station, between two buildings",
    },
    {
        "name": "Movement & Direction Prepositions",
        "description": "to/towards/into/out of/across/through/past/along",
        "examples": "go to school, walk across the street, run through the park",
    },
    {
        "name": "Dependent Prepositions (Adj + Prep)",
        "description": "interested in, proud of, famous for, good at, angry with, different from",
        "examples": "She is interested in art.",
    },
    {
        "name": "Dependent Prepositions (Verb + Prep)",
        "description": "wait for, listen to, depend on, agree with, belong to, laugh at, think about",
        "examples": "I’m waiting for the bus.",
    },
    {
        "name": "Fixed Prepositional Phrases",
        "description": "by mistake, on foot, in a hurry, at last, on holiday, by chance, in advance",
        "examples": "We arrived at last.",
    },
]

PREPOSITIONS_EXERCISE_FORMATS = [
    "multiple_choice",
    "sentence_transformation",
    "error_correction",
    "contextual_gap_fill",
]

CONFUSING_WORDS_SUBTYPES = [
    {
        "name": "Homophones",
        "description": "same/very similar sound, different spelling/meaning",
        "examples": "their/there/they're, its/it's, your/you're, weather/whether, passed/past",
    },
    {
        "name": "Look-alikes",
        "description": "similar spelling, different meaning/part of speech",
        "examples": "accept/except, affect/effect, advice/advise, quiet/quite, loose/lose",
    },
    {
        "name": "False Friends",
        "description": "words that look familiar to Ukrainian speakers but mean something else",
        "examples": "actual/actually, fabric, magazine, sympathetic",
    },
    {
        "name": "Semantic Confusion",
        "description": "near-synonyms or commonly confused usage pairs",
        "examples": "make/do, say/tell/speak/talk, lend/borrow, rob/steal, remember/remind",
    },
]

CONFUSING_WORDS_FORMATS = [
    "multiple_choice",
    "error_correction",
    "definition_match",
    "odd_one_out",
]

GRAMMAR_SUBTYPES = [
    {
        "name": "Verb Tenses",
        "description": "present/past/future systems and time links",
        "focus": "Present Simple vs Continuous, Past Simple vs Continuous, Present Perfect, Future forms, Past Perfect, Present Perfect Continuous",
    },
    {
        "name": "Modals",
        "description": "ability, permission, obligation, advice, possibility",
        "focus": "can/could/be able to, must/have to/needn't, should/ought to/had better, may/might/could",
    },
    {
        "name": "Sentence Structure",
        "description": "complex syntax and clause control",
        "focus": "Passive Voice, Conditionals (0/1/2), Reported Speech, Relative Clauses, Direct vs Indirect Questions",
    },
    {
        "name": "Verb Patterns",
        "description": "gerund vs infinitive patterns",
        "focus": "verb + -ing, verb + to + infinitive, gerund after prepositions",
    },
    {
        "name": "Nouns, Articles, Pronouns",
        "description": "countability, article choice, pronoun forms",
        "focus": "countable/uncountable, a/an/the/zero article, quantifiers, possessive/reflexive/relative pronouns",
    },
    {
        "name": "Adjectives & Adverbs",
        "description": "description quality and modifier control",
        "focus": "comparatives/superlatives, -ed vs -ing adjectives, adjective order, adverbs of frequency/manner/degree",
    },
    {
        "name": "Prepositions in Grammar Context",
        "description": "time/place/movement/dependent prepositions in grammar frames",
        "focus": "in/on/at, across/through/past/towards, depend on/wait for/good at",
    },
]

GRAMMAR_EXERCISE_FORMATS = [
    "multiple_choice",
    "error_correction",
    "sentence_transformation",
    "form_selection",
]

GRAMMAR_SENTENCE_TYPES = [
    "affirmative",
    "negative",
    "question",
]

QUIZ_VALIDATION_MAX_ATTEMPTS = 3
QUIZ_SIGNATURE_HISTORY_LIMIT = 60
QUIZ_SIGNATURE_CHECK_WINDOW = 40
PHOTO_URL_HISTORY_LIMIT = 50
PHOTO_URL_CHECK_WINDOW = 35
PHOTO_URL_REFETCH_ATTEMPTS = 4

# Додаємо до запиту при refetch, щоб Unsplash не повертав той самий «переможець»
DAILY_PHOTO_REFETCH_TAGS = [
    "soft light variation",
    "different angle nature",
    "alternate framing landscape",
    "misty atmospheric depth",
    "golden hour variant",
    "wide scenic mood",
    "peaceful morning haze",
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


async def get_daily_phrase_photo_query(history_mgr) -> str:
    """Повертає сезонний природний фон daily_phrase: сезон + місяць + дата."""
    holiday = get_today_holiday()
    if holiday:
        log.info(f"🎉 Holiday detected: {holiday['name']}")
        return holiday["photo_query"]

    today = date.today()
    month = today.month
    day = today.day
    season = get_season(month)
    if day <= 10:
        stage = "early"
    elif day <= 20:
        stage = "mid"
    else:
        stage = "late"

    seasonal_pool = DAILY_PHRASE_SEASON_NATURE_PHOTOS.get(season, [])
    monthly_pool = DAILY_PHRASE_MONTH_NATURE_PHOTOS.get(month, [])
    stage_hints = DAILY_PHRASE_MONTH_STAGE_HINTS.get(stage, [])
    base_pool = seasonal_pool + monthly_pool
    if stage_hints:
        # Додаємо підказку стадії місяця до частини запитів для м'якого date-aware ефекту.
        staged_pool = [f"{q} {random.choice(stage_hints)}" for q in random.sample(base_pool, k=min(3, len(base_pool)))]
    else:
        staged_pool = []
    candidates = base_pool + staged_pool
    if not candidates:
        candidates = ["green nature landscape cinematic moody"]

    key = "used:daily_phrase_photo_queries_v2"
    try:
        used_raw = await history_mgr.r.lrange(key, 0, -1)
        used_set = set(used_raw or [])
        available = [q for q in candidates if q not in used_set]
        if not available:
            log.info("🔄 Daily phrase nature photo pool exhausted — resetting rotation")
            await history_mgr.r.delete(key)
            available = candidates

        query = random.choice(available)
        await history_mgr.r.lpush(key, query)
        await history_mgr.r.ltrim(key, 0, max(20, len(candidates)) - 1)
        log.info(
            f"🎨 Daily phrase nature photo query: '{query}' | season={season} month={month} day={day} stage={stage}"
        )
        return query
    except Exception as e:
        fallback = random.choice(candidates)
        log.error(f"❌ get_daily_phrase_photo_query error: {e} — fallback '{fallback}'")
        return fallback


def _infer_situation_base_from_topic_name(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("email", "call", "phone", "message")):
        return "warm desk workspace laptop soft light cozy natural color"
    if any(k in n for k in ("meeting", "conference", "agenda")):
        return "conference room daylight warm natural indoor plants color"
    if any(k in n for k in ("office", "work", "cv", "cover letter", "network", "freelance", "remote")):
        return "modern office warm natural daylight interior color not grey"
    if any(k in n for k in ("bank", "money", "post office", "tax", "budget", "insurance")):
        return "urban street golden hour warm light bokeh colorful city"
    if any(k in n for k in ("doctor", "dentist", "pharmacy", "health", "hospital", "gym")):
        return "calm green nature healing soft morning warm light color"
    if any(k in n for k in ("shop", "market", "clothes", "beauty", "sale")):
        return "shopping district warm evening lights bokeh colorful"
    if any(k in n for k in ("travel", "airport", "hotel", "train", "taxi")):
        return "travel journey warm golden hour landscape soft color cinematic"
    if any(k in n for k in ("home", "house", "apartment", "furniture", "tap", "lock")):
        return "cozy home interior warm natural light soft color interior"
    if any(k in n for k in ("weather", "nature", "animal", "environment")):
        return "dramatic sky landscape warm natural colors golden hour outdoor"
    return None


def resolve_situation_base_photo_query(category: dict) -> str:
    raw = (category.get("photo_query") or "").strip()
    name = str(category.get("name") or "").strip()
    if raw and raw != SITUATION_GENERIC_PHOTO_QUERY:
        return raw
    if name in SITUATION_TOPIC_PHOTO_BASE_OVERRIDES:
        return SITUATION_TOPIC_PHOTO_BASE_OVERRIDES[name]
    inferred = _infer_situation_base_from_topic_name(name) if name else None
    if inferred:
        return inferred
    return raw or "cozy everyday life warm natural color soft light cinematic"


async def get_situation_photo_query(history_mgr, category: dict) -> str:
    holiday = get_today_holiday()
    if holiday:
        log.info(f"🎉 Holiday situation photo: {holiday['name']}")
        return holiday["photo_query"] + " celebration warm natural colors"

    today = date.today()
    season = get_season(today.month)
    if today.day <= 10:
        stage = "early"
    elif today.day <= 20:
        stage = "mid"
    else:
        stage = "late"

    base = resolve_situation_base_photo_query(category)
    season_style = random.choice(SITUATION_SEASON_STYLE.get(season, ["cinematic soft light"]))
    stage_style = random.choice(SITUATION_MONTH_STAGE_STYLE.get(stage, ["balanced composition"]))

    def compose(mood: str) -> str:
        return f"{base} {season_style} {stage_style} {mood} {SITUATION_PHOTO_QUERY_SUFFIX}"

    moods = list(SITUATION_WARM_COLOR_MOODS)
    random.shuffle(moods)
    candidates = [compose(m) for m in moods]

    key = "used:situation_photo_queries_v2"
    try:
        used_raw = await history_mgr.r.lrange(key, 0, -1)
        used_set = set(used_raw or [])
        available = [q for q in candidates if q not in used_set]
        if not available:
            log.info("🔄 Situation photo mood pool exhausted — resetting rotation")
            await history_mgr.r.delete(key)
            available = candidates

        query = random.choice(available)
        await history_mgr.r.lpush(key, query)
        await history_mgr.r.ltrim(key, 0, 39)
        log.info(
            f"🎨 Situation photo query: '{query}' | season={season} month={today.month} day={today.day} stage={stage}"
        )
        return query
    except Exception as e:
        fallback = compose(random.choice(SITUATION_WARM_COLOR_MOODS))
        log.error(f"❌ get_situation_photo_query error: {e} — fallback '{fallback}'")
        return fallback


def get_photo_query_for_quote() -> str:
    return random.choice(QUOTE_PHOTOS)


async def get_quote_photo_query(history_mgr) -> str:
    """Повертає фон для quote_motivation без швидких повторів."""
    key = "used:quote_photo_queries"
    try:
        used_raw = await history_mgr.r.lrange(key, 0, -1)
        used_set = set(used_raw or [])
        available = [q for q in QUOTE_PHOTOS if q not in used_set]
        if not available:
            log.info("🔄 Quote photo pool exhausted — resetting rotation")
            await history_mgr.r.delete(key)
            available = QUOTE_PHOTOS

        query = random.choice(available)
        await history_mgr.r.lpush(key, query)
        await history_mgr.r.ltrim(key, 0, len(QUOTE_PHOTOS) - 1)
        log.info(f"🎨 Quote rotated photo query: {query}")
        return query
    except Exception as e:
        fallback = random.choice(QUOTE_PHOTOS)
        log.error(f"❌ get_quote_photo_query error: {e} — fallback '{fallback}'")
        return fallback


async def pick_vocabulary_15_theme(history_mgr: "HistoryManager") -> dict:
    """
    Вівторок + дата з HOLIDAYS → святкова лексика.
    Інакше випадково: тема з DAILY_PHRASE_TOPICS (атлас) або «вільна» категорія від Gemini.
    Список використаних назв тем — Redis used:vocabulary_15_themes (поповнюється після успішного поста).
    """
    today = datetime.now(TZ).date()
    wd = today.weekday()
    hol = HOLIDAYS.get((today.month, today.day))

    if wd == 1 and hol is not None:
        return {
            "theme_mode": "holiday",
            "theme_title": f"{hol['name']} — English vocabulary",
            "theme_scope": (
                f"{hol['situation_description']} "
                "Generate vocabulary useful for this occasion (mixed A2–B1)."
            ),
        }

    try:
        used_raw = await history_mgr.r.lrange("used:vocabulary_15_themes", 0, -1)
        used_set = {str(x) for x in (used_raw or []) if x}
    except Exception as e:
        log.error(f"❌ pick_vocabulary_15_theme redis: {e}")
        used_set = set()

    if random.random() < 0.5:
        candidates = [t for t in DAILY_PHRASE_TOPICS if t["name"] not in used_set]
        if not candidates:
            try:
                await history_mgr.r.delete("used:vocabulary_15_themes")
            except Exception:
                pass
            candidates = list(DAILY_PHRASE_TOPICS)
        t = random.choice(candidates)
        return {
            "theme_mode": "atlas",
            "theme_title": t["name"],
            "theme_scope": t["desc"],
        }

    return {
        "theme_mode": "gemini_freeform",
        "theme_title": "",
        "theme_scope": (
            "Invent ONE clear English title for this post (3–12 words) WITHOUT the numeral 15 in the title — "
            "the list is already numbered on the card. Examples: "
            "'Action verbs for everyday life', 'Adjectives to describe people', "
            "'Phrasal verbs about travel'. Then list exactly 15 items matching that title."
        ),
    }


async def pick_photo_relax_theme(history_mgr: "HistoryManager") -> dict:
    """Ротація тем природи + стилю тексту для п’ятниці photo_relax (індекси після успішного поста)."""
    idx = await history_mgr.get_photo_relax_theme_index()
    t = PHOTO_RELAX_THEMES[idx]
    vidx = await history_mgr.get_photo_relax_voice_style_index()
    vs = PHOTO_RELAX_VOICE_STYLES[vidx]
    log.info(
        f"🌿 photo_relax theme idx={idx} id={t['id']} query='{t['photo_query']}' | "
        f"voice_style={vs['id']}"
    )
    return {
        "visual_theme": t["label"],
        "photo_query": t["photo_query"],
        "theme_id": t["id"],
        "voice_style_id": vs["id"],
        "voice_style_instruction": vs["instruction"],
    }


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
        self.url   = _get_required_env("UPSTASH_REDIS_REST_URL").rstrip("/")
        self.token = _get_required_env("UPSTASH_REDIS_REST_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def _cmd(self, *args):
        payload = [str(a) for a in args]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # JSON payload надійніше за URL-параметри для спецсимволів.
                resp = await client.post(self.url, headers=self.headers, json=payload)
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

    async def get_recent_signatures(self, rubric: str, limit: int = QUIZ_SIGNATURE_CHECK_WINDOW) -> list:
        key = f"used:signatures:{rubric}"
        try:
            items = await self.r.lrange(key, 0, limit - 1)
            return [item if isinstance(item, str) else str(item) for item in (items or [])]
        except Exception as e:
            log.error(f"❌ Redis get_recent_signatures error for [{rubric}]: {e}")
            return []

    async def add_signature(self, rubric: str, signature: str, max_items: int = QUIZ_SIGNATURE_HISTORY_LIMIT):
        key = f"used:signatures:{rubric}"
        try:
            await self.r.lpush(key, signature)
            await self.r.ltrim(key, 0, max_items - 1)
        except Exception as e:
            log.error(f"❌ Redis add_signature error for [{rubric}]: {e}")

    async def get_recent_photo_urls(self, rubric: str, limit: int = PHOTO_URL_CHECK_WINDOW) -> list:
        key = f"used:photo_urls:{rubric}"
        try:
            items = await self.r.lrange(key, 0, limit - 1)
            return [item if isinstance(item, str) else str(item) for item in (items or [])]
        except Exception as e:
            log.error(f"❌ Redis get_recent_photo_urls error for [{rubric}]: {e}")
            return []

    async def add_photo_url(self, rubric: str, url: str, max_items: int = PHOTO_URL_HISTORY_LIMIT):
        key = f"used:photo_urls:{rubric}"
        try:
            await self.r.lpush(key, url)
            await self.r.ltrim(key, 0, max_items - 1)
        except Exception as e:
            log.error(f"❌ Redis add_photo_url error for [{rubric}]: {e}")

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

    async def get_quote_theme_index(self) -> int:
        try:
            val = await self.r.get("quote:theme_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(QUOTE_THEMES)
            log.info(f"🧭 Quote theme index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_quote_theme_index error: {e} — using index 0")
            return 0

    async def advance_quote_theme_index(self):
        try:
            current = await self.get_quote_theme_index()
            next_idx = (current + 1) % len(QUOTE_THEMES)
            await self.r.set("quote:theme_rotation_index", str(next_idx))
            log.info(f"🧭 Quote theme index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_quote_theme_index error: {e}")

    async def get_photo_relax_theme_index(self) -> int:
        try:
            val = await self.r.get("photo_relax:theme_rotation_index")
            idx = int(val) if val is not None else 0
            return idx % len(PHOTO_RELAX_THEMES)
        except Exception as e:
            log.error(f"❌ Redis get_photo_relax_theme_index error: {e} — using index 0")
            return 0

    async def advance_photo_relax_theme_index(self):
        try:
            current = await self.get_photo_relax_theme_index()
            next_idx = (current + 1) % len(PHOTO_RELAX_THEMES)
            await self.r.set("photo_relax:theme_rotation_index", str(next_idx))
            log.info(f"🌿 photo_relax theme index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_photo_relax_theme_index error: {e}")

    async def get_photo_relax_voice_style_index(self) -> int:
        try:
            val = await self.r.get("photo_relax:voice_style_index")
            idx = int(val) if val is not None else 0
            return idx % len(PHOTO_RELAX_VOICE_STYLES)
        except Exception as e:
            log.error(f"❌ Redis get_photo_relax_voice_style_index error: {e} — using index 0")
            return 0

    async def advance_photo_relax_voice_style_index(self):
        try:
            current = await self.get_photo_relax_voice_style_index()
            next_idx = (current + 1) % len(PHOTO_RELAX_VOICE_STYLES)
            await self.r.set("photo_relax:voice_style_index", str(next_idx))
            log.info(f"🌿 photo_relax voice style index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_photo_relax_voice_style_index error: {e}")

    async def get_travel_video_banned_places(self) -> list[str]:
        try:
            items = await self.r.lrange("used:travel_video_places", 0, 199)
            return [str(x) for x in (items or []) if x]
        except Exception as e:
            log.error(f"❌ Redis get_travel_video_banned_places error: {e}")
            return []

    async def add_travel_video_place(self, place_key: str):
        try:
            await self.r.lpush("used:travel_video_places", place_key)
            await self.r.ltrim("used:travel_video_places", 0, 299)
        except Exception as e:
            log.error(f"❌ Redis add_travel_video_place error: {e}")

    async def get_interesting_cities_banned(self) -> list[str]:
        try:
            items = await self.r.lrange("used:interesting_cities_places", 0, 99)
            return [str(x) for x in (items or []) if x]
        except Exception as e:
            log.error(f"❌ Redis get_interesting_cities_banned error: {e}")
            return []

    async def add_interesting_city_place(self, place_key: str):
        try:
            await self.r.lpush("used:interesting_cities_places", place_key)
            await self.r.ltrim("used:interesting_cities_places", 0, 199)
        except Exception as e:
            log.error(f"❌ Redis add_interesting_city_place error: {e}")

    async def get_prepositions_subtype_index(self) -> int:
        try:
            val = await self.r.get("prepositions:subtype_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(PREPOSITIONS_SUBTYPES)
            log.info(f"📚 Prepositions subtype index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_prepositions_subtype_index error: {e} — using index 0")
            return 0

    async def advance_prepositions_subtype_index(self):
        try:
            current = await self.get_prepositions_subtype_index()
            next_idx = (current + 1) % len(PREPOSITIONS_SUBTYPES)
            await self.r.set("prepositions:subtype_rotation_index", str(next_idx))
            log.info(f"📚 Prepositions subtype index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_prepositions_subtype_index error: {e}")

    async def get_prepositions_format_index(self) -> int:
        try:
            val = await self.r.get("prepositions:format_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(PREPOSITIONS_EXERCISE_FORMATS)
            log.info(f"🧩 Prepositions format index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_prepositions_format_index error: {e} — using index 0")
            return 0

    async def advance_prepositions_format_index(self):
        try:
            current = await self.get_prepositions_format_index()
            next_idx = (current + 1) % len(PREPOSITIONS_EXERCISE_FORMATS)
            await self.r.set("prepositions:format_rotation_index", str(next_idx))
            log.info(f"🧩 Prepositions format index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_prepositions_format_index error: {e}")

    async def get_confusing_words_subtype_index(self) -> int:
        try:
            val = await self.r.get("confusing_words:subtype_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(CONFUSING_WORDS_SUBTYPES)
            log.info(f"🧠 ConfusingWords subtype index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_confusing_words_subtype_index error: {e} — using index 0")
            return 0

    async def advance_confusing_words_subtype_index(self):
        try:
            current = await self.get_confusing_words_subtype_index()
            next_idx = (current + 1) % len(CONFUSING_WORDS_SUBTYPES)
            await self.r.set("confusing_words:subtype_rotation_index", str(next_idx))
            log.info(f"🧠 ConfusingWords subtype index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_confusing_words_subtype_index error: {e}")

    async def get_confusing_words_format_index(self) -> int:
        try:
            val = await self.r.get("confusing_words:format_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(CONFUSING_WORDS_FORMATS)
            log.info(f"🧠 ConfusingWords format index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_confusing_words_format_index error: {e} — using index 0")
            return 0

    async def advance_confusing_words_format_index(self):
        try:
            current = await self.get_confusing_words_format_index()
            next_idx = (current + 1) % len(CONFUSING_WORDS_FORMATS)
            await self.r.set("confusing_words:format_rotation_index", str(next_idx))
            log.info(f"🧠 ConfusingWords format index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_confusing_words_format_index error: {e}")

    async def get_vocabulary_topic_index(self) -> int:
        try:
            val = await self.r.get("vocabulary:topic_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(SITUATION_CATEGORIES)
            log.info(f"📘 Vocabulary topic index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_vocabulary_topic_index error: {e} — using index 0")
            return 0

    async def advance_vocabulary_topic_index(self):
        try:
            current = await self.get_vocabulary_topic_index()
            next_idx = (current + 1) % len(SITUATION_CATEGORIES)
            await self.r.set("vocabulary:topic_rotation_index", str(next_idx))
            log.info(f"📘 Vocabulary topic index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_vocabulary_topic_index error: {e}")

    async def get_grammar_subtype_index(self) -> int:
        try:
            val = await self.r.get("grammar:subtype_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(GRAMMAR_SUBTYPES)
            log.info(f"🧱 Grammar subtype index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_grammar_subtype_index error: {e} — using index 0")
            return 0

    async def advance_grammar_subtype_index(self):
        try:
            current = await self.get_grammar_subtype_index()
            next_idx = (current + 1) % len(GRAMMAR_SUBTYPES)
            await self.r.set("grammar:subtype_rotation_index", str(next_idx))
            log.info(f"🧱 Grammar subtype index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_grammar_subtype_index error: {e}")

    async def get_grammar_format_index(self) -> int:
        try:
            val = await self.r.get("grammar:format_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(GRAMMAR_EXERCISE_FORMATS)
            log.info(f"🧱 Grammar format index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_grammar_format_index error: {e} — using index 0")
            return 0

    async def advance_grammar_format_index(self):
        try:
            current = await self.get_grammar_format_index()
            next_idx = (current + 1) % len(GRAMMAR_EXERCISE_FORMATS)
            await self.r.set("grammar:format_rotation_index", str(next_idx))
            log.info(f"🧱 Grammar format index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_grammar_format_index error: {e}")

    async def get_grammar_sentence_type_index(self) -> int:
        try:
            val = await self.r.get("grammar:sentence_type_rotation_index")
            idx = int(val) if val is not None else 0
            idx = idx % len(GRAMMAR_SENTENCE_TYPES)
            log.info(f"🧱 Grammar sentence type index: {idx}")
            return idx
        except Exception as e:
            log.error(f"❌ Redis get_grammar_sentence_type_index error: {e} — using index 0")
            return 0

    async def advance_grammar_sentence_type_index(self):
        try:
            current = await self.get_grammar_sentence_type_index()
            next_idx = (current + 1) % len(GRAMMAR_SENTENCE_TYPES)
            await self.r.set("grammar:sentence_type_rotation_index", str(next_idx))
            log.info(f"🧱 Grammar sentence type index advanced: {current} → {next_idx}")
        except Exception as e:
            log.error(f"❌ Redis advance_grammar_sentence_type_index error: {e}")

# ──────────────────────────────────────────────
# ФОТО API
# ──────────────────────────────────────────────
async def fetch_photo_unsplash(query: str, use_topics: bool = True, pick_random: bool = False) -> str | None:
    if not UNSPLASH_ACCESS_KEY:
        log.warning("⚠️ UNSPLASH_ACCESS_KEY not set")
        return None
    try:
        # pick_random: різні кадри з одного пошуку (анти-повтор); інакше — топ за лайками
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
            log.info(f"📷 Unsplash response: status={resp.status_code} query='{query}' pick_random={pick_random}")
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("results", [])
                if not photos:
                    log.warning(f"⚠️ Unsplash: no results for query '{query}'")
                    return None
                pool = photos[:10] if pick_random else photos[:5]
                if pick_random:
                    best = random.choice(pool)
                else:
                    best = max(pool, key=lambda p: p.get("likes", 0))
                photo_url = best["urls"]["regular"]
                log.info(f"✅ Unsplash photo: likes={best.get('likes',0)} url={photo_url[:60]}")
                return photo_url
            else:
                log.warning(f"⚠️ Unsplash failed: {resp.status_code} — {resp.text[:200]}")
                return None
    except Exception as e:
        log.error(f"❌ Unsplash exception: {e}")
        return None


async def fetch_photo_pexels(query: str, pick_random: bool = False) -> str | None:
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
                    pool = photos[:10] if pick_random else photos[:5]
                    photo = random.choice(pool)
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


async def fetch_photo_pixabay(query: str, pick_random: bool = False) -> str | None:
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
                    pool = hits[:10] if pick_random else hits[:5]
                    photo = random.choice(pool)
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


async def fetch_photo(query: str, use_topics: bool = True, pick_random: bool = False) -> str | None:
    """Пробує Unsplash → Pexels → Pixabay. pick_random — варіативність з топу результатів (анти-повтор)."""
    log.info(f"🔍 Fetching photo for query: '{query}' use_topics={use_topics} pick_random={pick_random}")

    # Спроба 1: Unsplash
    for attempt in range(1, 4):
        photo_url = await fetch_photo_unsplash(query, use_topics=use_topics, pick_random=pick_random)
        if photo_url:
            return photo_url
        log.warning(f"⚠️ Unsplash attempt {attempt}/3 failed")
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)

    # Спроба 2: Pexels
    for attempt in range(1, 4):
        photo_url = await fetch_photo_pexels(query, pick_random=pick_random)
        if photo_url:
            return photo_url
        log.warning(f"⚠️ Pexels attempt {attempt}/3 failed")
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)

    # Спроба 3: Pixabay
    for attempt in range(1, 4):
        photo_url = await fetch_photo_pixabay(query, pick_random=pick_random)
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
# TRAVEL VIDEO — сток, FFmpeg, TTS
# ──────────────────────────────────────────────
def _travel_video_place_key(landmark: str, country: str) -> str:
    return _normalize_text(f"{landmark.strip()}|{country.strip()}")


def ffprobe_duration_seconds(path: str) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return 0.0
        return float(r.stdout.strip())
    except Exception as e:
        log.warning(f"⚠️ ffprobe duration: {e}")
        return 0.0


def _run_ffmpeg(args: list[str]) -> bool:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.error(f"❌ ffmpeg failed: {r.stderr[:800]}")
            return False
        return True
    except Exception as e:
        log.error(f"❌ ffmpeg exception: {e}")
        return False


async def download_url_to_file(url: str, dest_path: str) -> bool:
    """Стрімінг у файл без повного буфера в RAM (важливо для OOM на великих MP4)."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("GET", url, follow_redirects=True) as resp:
                if resp.status_code != 200:
                    log.warning(f"⚠️ download HTTP {resp.status_code}")
                    return False
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
        ok = os.path.getsize(dest_path) > 0
        if not ok:
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return ok
    except Exception as e:
        log.error(f"❌ download_url_to_file: {e}")
        try:
            if os.path.isfile(dest_path):
                os.remove(dest_path)
        except OSError:
            pass
        return False


def _pick_pexels_video_url(videos: list) -> str | None:
    """Вертикаль ≥720p; уникаємо 4K як джерела (OOM): пріоритет long edge ≤ TRAVEL_VIDEO_PEXELS_MAX_LONG_EDGE."""
    candidates: list[tuple[int, int, int, str]] = []  # area, h, w, link
    for v in videos:
        for vf in v.get("video_files") or []:
            w = int(vf.get("width") or 0)
            h = int(vf.get("height") or 0)
            link = vf.get("link")
            if not link or w < 1 or h < 1:
                continue
            if h < w:
                continue
            if h < 720:
                continue
            candidates.append((w * h, h, w, link))

    if not candidates:
        return None

    cap = TRAVEL_VIDEO_PEXELS_MAX_LONG_EDGE
    capped = [c for c in candidates if c[1] <= cap]
    pool = capped if capped else candidates
    if capped:
        pool.sort(key=lambda t: t[1], reverse=True)
        return pool[0][3]
    pool.sort(key=lambda t: t[0])
    return pool[0][3]


async def fetch_pexels_videos(query: str) -> list[str]:
    if not PEXELS_API_KEY:
        return []
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "orientation": "portrait", "per_page": 15}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                log.warning(f"⚠️ Pexels videos HTTP {resp.status_code}")
                return []
            data = resp.json()
            out: list[str] = []
            for v in data.get("videos") or []:
                u = _pick_pexels_video_url([v])
                if u:
                    out.append(u)
            return out
    except Exception as e:
        log.error(f"❌ fetch_pexels_videos: {e}")
        return []


async def fetch_pixabay_videos(query: str) -> list[str]:
    if not PIXABAY_API_KEY:
        return []
    url = "https://pixabay.com/api/videos/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "per_page": 15,
        "safesearch": "true",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                log.warning(f"⚠️ Pixabay videos HTTP {resp.status_code}")
                return []
            data = resp.json()
            out: list[str] = []
            for hit in data.get("hits") or []:
                vids = hit.get("videos") or {}
                # medium first — large часто зайво важкий для RAM/OOM у контейнері
                for key in ("medium", "small", "large", "tiny"):
                    block = vids.get(key)
                    if isinstance(block, dict) and block.get("url"):
                        out.append(block["url"])
                        break
            return out
    except Exception as e:
        log.error(f"❌ fetch_pixabay_videos: {e}")
        return []


async def fetch_pixabay_music_url() -> str | None:
    if not PIXABAY_API_KEY:
        return None
    url = "https://pixabay.com/api/audio/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": "calm ambient instrumental",
        "per_page": 10,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            for hit in data.get("hits") or []:
                au = hit.get("audioURL") or hit.get("previewURL") or hit.get("audio")
                if au:
                    return str(au)
    except Exception as e:
        log.warning(f"⚠️ fetch_pixabay_music_url: {e}")
    return None


def normalize_clip_to_vertical_9_16(src: str, dst: str, max_sec: float) -> bool:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    args = [
        "ffmpeg",
        "-y",
        "-threads",
        "2",
        "-i",
        src,
        "-vf",
        vf,
        "-t",
        str(max_sec),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "26",
        "-an",
        "-movflags",
        "+faststart",
        dst,
    ]
    return _run_ffmpeg(args)


def concat_videos_ffmpeg(paths: list[str], out_path: str, max_total_sec: float) -> bool:
    if not paths:
        return False
    if len(paths) == 1:
        d = ffprobe_duration_seconds(paths[0])
        if d <= max_total_sec:
            shutil.copy(paths[0], out_path)
            return True
        return _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                paths[0],
                "-t",
                str(max_total_sec),
                "-c",
                "copy",
                out_path,
            ]
        )
    lst = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        for p in paths:
            lst.write(f"file '{os.path.abspath(p)}'\n")
        lst.close()
        tmp_concat = out_path + ".concat.mp4"
        ok = _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                lst.name,
                "-c",
                "copy",
                tmp_concat,
            ]
        )
        if not ok:
            return False
        d = ffprobe_duration_seconds(tmp_concat)
        if d > max_total_sec:
            ok2 = _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_concat,
                    "-t",
                    str(max_total_sec),
                    "-c",
                    "copy",
                    out_path,
                ]
            )
            try:
                os.remove(tmp_concat)
            except OSError:
                pass
            return ok2
        shutil.move(tmp_concat, out_path)
        return True
    finally:
        try:
            os.remove(lst.name)
        except OSError:
            pass


async def elevenlabs_tts_to_mp3(text: str, out_path: str) -> bool:
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        return False
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {"text": text, "model_id": "eleven_multilingual_v2"}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                log.warning(f"⚠️ ElevenLabs HTTP {resp.status_code}: {resp.text[:200]}")
                return False
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        log.warning(f"⚠️ ElevenLabs TTS: {e}")
        return False


def google_tts_to_mp3(text: str, out_path: str) -> bool:
    if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.isfile(GOOGLE_APPLICATION_CREDENTIALS):
        log.warning("⚠️ Google TTS: GOOGLE_APPLICATION_CREDENTIALS path missing or not a file")
        return False
    try:
        from google.cloud import texttospeech
    except ImportError:
        log.error("❌ google-cloud-texttospeech not installed")
        return False
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            name="en-GB-Wavenet-B",
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
        )
        resp = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        with open(out_path, "wb") as f:
            f.write(resp.audio_content)
        return True
    except Exception as e:
        log.warning(f"⚠️ Google TTS: {e}")
        return False


def mix_voice_and_music(voice_mp3: str, music_path: str | None, out_path: str) -> bool:
    if not music_path or not os.path.isfile(music_path):
        shutil.copy(voice_mp3, out_path)
        return True
    # Тиха музика; amix duration=first = довжина голосу
    args = [
        "ffmpeg",
        "-y",
        "-i",
        voice_mp3,
        "-i",
        music_path,
        "-filter_complex",
        "[1:a]volume=0.12[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[aout]",
        "-map",
        "[aout]",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        out_path,
    ]
    return _run_ffmpeg(args)


def mux_video_audio_pad(video_path: str, audio_path: str, out_path: str) -> bool:
    vd = ffprobe_duration_seconds(video_path)
    ad = ffprobe_duration_seconds(audio_path)
    if vd <= 0 or ad <= 0:
        return False
    pad = max(0.0, vd - ad)
    if pad <= 0.01:
        return _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                out_path,
            ]
        )
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-filter_complex",
            f"[1:a]apad=pad_dur={pad}[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ]
    )


def final_encode_for_telegram(src_path: str, dst_path: str) -> bool:
    """Агресивний бітрейт, H.264 + AAC, 9:16."""
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-threads",
            "2",
            "-i",
            src_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "28",
            "-maxrate",
            "2M",
            "-bufsize",
            "4M",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            dst_path,
        ]
    )


def concat_two_videos(v1: str, v2: str, out_path: str) -> bool:
    lst = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        lst.write(f"file '{os.path.abspath(v1)}'\nfile '{os.path.abspath(v2)}'\n")
        lst.close()
        ok = _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                lst.name,
                "-c",
                "copy",
                out_path,
            ]
        )
        if ok:
            return True
        log.warning("⚠️ concat copy failed — re-encoding")
        return _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                lst.name,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "26",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                out_path,
            ]
        )
    finally:
        try:
            os.remove(lst.name)
        except OSError:
            pass


BRANDING_ENDCARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,600&family=Montserrat:wght@300;500;600&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: 1080px;
    height: 1920px;
    background: #0a0a0a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Montserrat', sans-serif;
  }}
  .card {{
    background: #1c1c1e;
    border-radius: 28px;
    padding: 72px 56px;
    text-align: center;
    margin: 0 48px;
    position: relative;
    overflow: hidden;
  }}
  .card::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
  }}
  .wordmark {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 56px;
    font-weight: 300;
    letter-spacing: 6px;
    color: #f5f5f7;
    line-height: 1.15;
  }}
  .wordmark span {{ color: #c9a84c; font-style: italic; }}
  .divider {{
    width: 120px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 28px auto;
  }}
  .tagline {{
    font-size: 11px;
    letter-spacing: 8px;
    text-transform: uppercase;
    color: #8e8e93;
    margin-top: 8px;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="wordmark">Improve<br>Your <span>English</span></div>
    <div class="divider"></div>
    <div class="tagline">Learn • Grow • Succeed</div>
  </div>
</body>
</html>"""


async def render_branding_png_bytes() -> bytes:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1080, "height": 1920})
        await page.set_content(BRANDING_ENDCARD_HTML, wait_until="domcontentloaded")
        await asyncio.sleep(1.2)
        png = await page.screenshot(type="png")
        await browser.close()
        return png


async def branding_clip_to_mp4(png_path: str, out_mp4: str, duration_sec: float) -> bool:
    """Статичний кадр → коротке відео з тихим AAC (для concat з основним роликом)."""
    args = [
        "ffmpeg",
        "-y",
        "-threads",
        "2",
        "-loop",
        "1",
        "-i",
        png_path,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-shortest",
        "-movflags",
        "+faststart",
        out_mp4,
    ]
    return _run_ffmpeg(args)


async def send_video_to_telegram(video_path: str, rubric: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(video_path, "rb") as f:
                    data = {"chat_id": TELEGRAM_CHAT_ID}
                    resp = await client.post(
                        url,
                        data=data,
                        files={"video": (f"{rubric}.mp4", f, "video/mp4")},
                    )
                log.info(f"📤 Telegram sendVideo attempt {attempt}: status={resp.status_code}")
                if resp.status_code == 200:
                    log.info(f"✅ Telegram: video sent [{rubric}]")
                    return True
                log.error(f"❌ sendVideo failed: {resp.status_code} — {resp.text[:400]}")
        except Exception as e:
            log.error(f"❌ sendVideo exception attempt {attempt}: {e}")
        if attempt < 3:
            await asyncio.sleep(5 * attempt)
    return False


def validate_travel_video_landmark_bundle(data: dict, banned: set[str]) -> tuple[bool, str]:
    lm = str(data.get("landmark_name", "")).strip()
    country = str(data.get("country", "")).strip()
    cat = str(data.get("category", "")).strip()
    stock_q = str(data.get("stock_query", "")).strip()
    narr = str(data.get("narration", "")).strip()
    if not lm or not country:
        return False, "empty landmark or country"
    if cat not in TRAVEL_VIDEO_LANDMARK_CATEGORIES:
        return False, f"bad category: {cat}"
    if not stock_q:
        return False, "empty stock_query"
    if not narr:
        return False, "empty narration"
    words = narr.split()
    if len(words) < 40 or len(words) > TRAVEL_VIDEO_NARRATION_WORDS_MAX:
        return False, f"narration word count {len(words)}"
    pk = _travel_video_place_key(lm, country)
    if pk in banned:
        return False, "place banned"
    cy = re.compile(r"[\u0400-\u04FF]")
    if cy.search(narr) or cy.search(lm) or cy.search(country):
        return False, "Cyrillic in fields"
    return True, "ok"


async def generate_travel_video_landmark_bundle(
    history: list,
    banned_places: list[str],
    max_attempts: int = 3,
) -> dict:
    extra = {"banned_places": banned_places}
    banned_norm = {_normalize_text(x) for x in banned_places if x}
    hist = list(history)
    last_reason = "unknown"
    for attempt in range(1, max_attempts + 1):
        data = await generate_content("travel_video_landmark", hist, extra)
        if isinstance(data, list):
            data = data[0] if data else {}
        ok, reason = validate_travel_video_landmark_bundle(data, banned_norm)
        if ok:
            if attempt > 1:
                log.info(f"✅ travel_video_landmark validated on retry #{attempt}")
            return data
        last_reason = reason
        log.warning(
            f"⚠️ travel_video_landmark reject attempt {attempt}/{max_attempts}: {reason}"
        )
        hist = hist + [f"reject:{reason}"]
    raise RuntimeError(f"travel_video_landmark failed: {last_reason}")


async def build_travel_video_main_from_stock(
    stock_query: str,
    landmark: str,
    country: str,
    tmpdir: str,
) -> str | None:
    """Повертає шлях до нормалізованого відео ≤ TRAVEL_VIDEO_MAIN_MAX_SEC."""
    urls: list[str] = []
    urls.extend(await fetch_pexels_videos(stock_query))
    if not urls:
        urls.extend(await fetch_pixabay_videos(stock_query))
    if not urls:
        q2 = f"{landmark} {country} landmark vertical"
        urls.extend(await fetch_pexels_videos(q2))
        urls.extend(await fetch_pixabay_videos(q2))
    if not urls:
        log.warning("⚠️ No stock video URLs")
        return None

    norm_paths: list[str] = []
    total = 0.0
    for i, u in enumerate(urls):
        if total >= TRAVEL_VIDEO_MAIN_MAX_SEC:
            break
        raw_path = os.path.join(tmpdir, f"raw_{i}.mp4")
        npath = os.path.join(tmpdir, f"norm_{i}.mp4")
        if not await download_url_to_file(u, raw_path):
            continue
        if not normalize_clip_to_vertical_9_16(
            raw_path, npath, TRAVEL_VIDEO_MAIN_MAX_SEC
        ):
            continue
        d = ffprobe_duration_seconds(npath)
        if d <= 0:
            continue
        norm_paths.append(npath)
        total += d
        if total >= TRAVEL_VIDEO_MAIN_MAX_SEC - 0.5:
            break

    if not norm_paths:
        return None

    out = os.path.join(tmpdir, "main_segment.mp4")
    if not concat_videos_ffmpeg(norm_paths, out, TRAVEL_VIDEO_MAIN_MAX_SEC):
        return None
    return out


async def publish_travel_video(rubric: str, redis_client: UpstashRedis):
    history_mgr = HistoryManager(redis_client)
    start_time = time.time()
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(f"🎬 START travel_video [{rubric}] at {datetime.now().strftime('%H:%M:%S')}")

    if not await history_mgr.acquire_lock(rubric):
        log.warning(f"⚠️ Lock exists for [{rubric}] — skipping")
        return

    try:
        banned = await history_mgr.get_travel_video_banned_places()
        history = await history_mgr.get_used(rubric)

        success = False
        for cycle in range(1, TRAVEL_VIDEO_PIPELINE_ATTEMPTS + 1):
            tmpdir = tempfile.mkdtemp(prefix="travel_vid_")
            try:
                log.info(f"🔄 travel_video pipeline cycle {cycle}/{TRAVEL_VIDEO_PIPELINE_ATTEMPTS}")
                try:
                    bundle = await generate_travel_video_landmark_bundle(
                        history, banned, max_attempts=3
                    )
                except RuntimeError as e:
                    log.warning(f"⚠️ [NF] travel_video Gemini bundle failed cycle {cycle}: {e}")
                    continue

                lm = str(bundle.get("landmark_name", "")).strip()
                country = str(bundle.get("country", "")).strip()
                stock_q = str(bundle.get("stock_query", "")).strip()
                narr = str(bundle.get("narration", "")).strip()

                main_seg = await build_travel_video_main_from_stock(
                    stock_q, lm, country, tmpdir
                )
                if not main_seg:
                    log.warning(f"⚠️ [NF] travel_video no stock video cycle {cycle}")
                    continue

                voice_mp3 = os.path.join(tmpdir, "voice.mp3")
                tts_ok = await elevenlabs_tts_to_mp3(narr, voice_mp3)
                if not tts_ok:
                    tts_ok = google_tts_to_mp3(narr, voice_mp3)
                if not tts_ok:
                    log.warning(f"⚠️ [NF] travel_video TTS failed cycle {cycle}")
                    continue

                music_path = None
                mu = await fetch_pixabay_music_url()
                if mu:
                    music_path = os.path.join(tmpdir, "music.mp3")
                    if not await download_url_to_file(mu, music_path):
                        music_path = None

                mixed_mp3 = os.path.join(tmpdir, "mixed.mp3")
                if not mix_voice_and_music(voice_mp3, music_path, mixed_mp3):
                    continue

                muxed = os.path.join(tmpdir, "main_with_audio.mp4")
                if not mux_video_audio_pad(main_seg, mixed_mp3, muxed):
                    log.warning(f"⚠️ [NF] travel_video mux failed cycle {cycle}")
                    continue

                png_bytes = await render_branding_png_bytes()
                brand_png = os.path.join(tmpdir, "brand.png")
                with open(brand_png, "wb") as f:
                    f.write(png_bytes)
                brand_mp4 = os.path.join(tmpdir, "brand.mp4")
                if not await branding_clip_to_mp4(
                    brand_png, brand_mp4, TRAVEL_VIDEO_BRAND_SEC
                ):
                    log.warning(f"⚠️ [NF] travel_video brand clip failed cycle {cycle}")
                    continue

                pre_final = os.path.join(tmpdir, "pre_final.mp4")
                if not concat_two_videos(muxed, brand_mp4, pre_final):
                    log.warning(f"⚠️ [NF] travel_video final concat failed cycle {cycle}")
                    continue

                final_path = os.path.join(tmpdir, "final_telegram.mp4")
                if not final_encode_for_telegram(pre_final, final_path):
                    log.warning(f"⚠️ [NF] travel_video final encode failed cycle {cycle}")
                    continue

                sz_mb = os.path.getsize(final_path) / (1024 * 1024)
                log.info(f"📦 travel_video final size: {sz_mb:.2f} MB")
                if sz_mb > 48:
                    log.warning("⚠️ [NF] travel_video file large for Telegram — trying anyway")

                success = await send_video_to_telegram(final_path, rubric)
                if success:
                    pk = _travel_video_place_key(lm, country)
                    if pk:
                        await history_mgr.add_travel_video_place(pk)
                    await history_mgr.add_used(
                        rubric, json.dumps(bundle, ensure_ascii=False)[:120]
                    )
                    log.info(
                        f"📣 [NF] travel_video published | {lm} | {country} | cycle={cycle}"
                    )
                    break
            finally:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception as e:
                    log.warning(f"⚠️ tmpdir cleanup: {e}")

        if not success:
            log.error(
                f"❌ [NF] travel_video SKIPPED after {TRAVEL_VIDEO_PIPELINE_ATTEMPTS} cycles — check logs above"
            )

        elapsed = time.time() - start_time
        log.info(f"⏱️ [{rubric}] completed in {elapsed:.1f}s | success={success}")

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR in travel_video: {e}", exc_info=True)
    finally:
        await history_mgr.release_lock(rubric)


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
    cefr_band = extra.get("cefr_band", "A2-B1")

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
  "example_ua": "Ukrainian translation of the example sentence (max 140 characters)"
}}
Rules:
- Phrase must relate to the given topic
- Minimum 5 words — avoid very short phrases like "See you" or "Thank you"
- Simple A2 vocabulary, natural everyday conversation
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
  ]
}}
Rules:
- Practical A2 level phrases for the situation
- Each phrase must be different and useful
{LANGUAGE_CENSOR}"""

    if rubric == "quote_motivation":
        theme_name = extra.get("quote_theme_name", "Stoicism & Resilience")
        theme_description = extra.get("quote_theme_description", "Focus on what we can control and keep moving.")
        return f"""System role:
You are a philosophical coach and mentor. Your task is to create deep motivational texts that combine stoicism, modern psychology, and strategic thinking.

Task:
Find a short motivational or wise quote for A2 level English students.
Selected theme for this post (mandatory):
- {theme_name}: {theme_description}
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "quote_en": "the quote in English (minimum 6 words, maximum 12 words, simple A2 vocabulary)",
  "quote_ua": "Ukrainian translation (natural, not word-for-word)"
}}
Rules:
- Minimum 6 words — NEVER generate quotes like 'Just do it', 'Dream big', 'Stay strong' (too short)
- Maximum 12 words
- Simple A2 vocabulary, memorable
- Good examples: 'The expert in anything was once a beginner.' (9 words)
- Self-Correction: Make sure the phrase is at least 6 words long
- Tone of voice:
  - Intellectual depth: avoid toxic positivity and empty motivation; prefer reflective, meaningful truths
  - Aesthetic brevity: every word must carry weight; no filler, no redundancy
  - Ambivalence: combine disciplined action with deep self-respect and inner calm
  - Bilingual quality: quote_en is the original in English; quote_ua is an artistic, natural Ukrainian rendering (not literal translation)
- The quote MUST match the selected theme for this post
- Prefer practical, grounded ideas over generic slogans
- Avoid cliches, toxic positivity, and vague lines without concrete meaning
- Include ONLY one quote (not a list), and keep it original in wording
{LANGUAGE_CENSOR}"""

    if rubric == "grammar_quiz":
        subtype = extra.get("grammar_subtype", "Verb Tenses")
        subtype_desc = extra.get("grammar_subtype_desc", "core grammar control for A2-B1")
        subtype_focus = extra.get("grammar_subtype_focus", "present and past contrasts")
        exercise_format = extra.get("grammar_format", "multiple_choice")
        sentence_type = extra.get("grammar_sentence_type", "affirmative")
        return f"""You are an English teacher. Create a grammar quiz question for A2-B1 level students.
{history_note}
Target CEFR level: {cefr_band}.
Selected grammar subtype (MANDATORY): {subtype}
Subtype scope: {subtype_desc}
Subtype focus points: {subtype_focus}
Exercise format for this item (MANDATORY): {exercise_format}
Preferred sentence type for this item: {sentence_type}
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
- Telegram sendPoll supports ONE question only, so output exactly one question item
- Keep difficulty in A2-B1 range: clear context + one key grammar decision
- The question MUST match the selected subtype
- Rotate these grammar families over time:
  1) Verb Tenses
  2) Modals
  3) Sentence Structure (passive, conditionals, reported speech, relative clauses, indirect questions)
  4) Verb Patterns (gerund/infinitive)
  5) Nouns/Articles/Pronouns
  6) Adjectives/Adverbs
  7) Prepositions in grammar context
- Exercise format behavior:
  - multiple_choice: standard one-gap grammar choice
  - error_correction: sentence has a grammar mistake, options are corrected variants
  - sentence_transformation: complete transformed sentence with correct grammar form
  - form_selection: choose correct verb/modal/article form by context
- If selected subtype is Verb Tenses:
  - rotate sentence forms, not only affirmative
  - include negative and question patterns regularly
  - respect selected sentence type: affirmative / negative / question
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "vocabulary_quiz":
        topic_name = extra.get("vocab_topic_name", "Daily Life")
        topic_desc = extra.get("vocab_topic_desc", "everyday real-life contexts")
        return f"""You are an English teacher. Create a vocabulary quiz question for A2-B1 level students.
{history_note}
Target CEFR level: {cefr_band}.
Selected vocabulary theme (MANDATORY): {topic_name}
Theme scope: {topic_desc}
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
- Test everyday A2-B1 vocabulary in context sentences
- The sentence MUST match the selected vocabulary theme
- Use practical real-life context instead of abstract examples
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "confusing_words_quiz":
        subtype = extra.get("confusing_subtype", "Semantic Confusion")
        subtype_desc = extra.get("confusing_subtype_desc", "commonly confused words by context")
        subtype_examples = extra.get("confusing_subtype_examples", "make/do, lend/borrow")
        exercise_format = extra.get("confusing_format", "multiple_choice")
        return f"""You are an English teacher. Create a quiz about commonly confused English words for A2-B1 level students.
{history_note}
Target CEFR level: {cefr_band}.
Selected subtype (MANDATORY): {subtype}
Subtype scope: {subtype_desc}
Subtype examples: {subtype_examples}
Exercise format for this item (MANDATORY): {exercise_format}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Sentence with a blank ___ testing confusing words (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation of the difference (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- Always provide exactly 4 options
- correct_index is 0-based and must point to the only correct option
- Telegram sendPoll supports ONE question only, so output exactly one question item (no series)
- Rotate confusion families over time:
  1) Homophones
  2) Look-alikes
  3) False Friends
  4) Semantic Confusion
- Keep complexity in A2-B1 range:
  - A2: everyday high-frequency words in clear contexts
  - B1: nuanced usage (make/do, say/tell/speak/talk, lend/borrow, rob/steal...)
- Exercise format behavior:
  - multiple_choice: standard single gap
  - error_correction: question contains wrong usage; options provide corrections
  - definition_match: short definition in question; options are candidate words
  - odd_one_out: pick the item that does NOT fit semantic/usage pattern
- Avoid repeating the same pair and near-identical stem from recent history
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "prepositions_quiz":
        subtype = extra.get("prepositions_subtype", "Time Prepositions")
        subtype_desc = extra.get("prepositions_subtype_desc", "in/on/at and common related prepositions")
        subtype_examples = extra.get("prepositions_subtype_examples", "in May, on Monday, at 5 PM")
        exercise_format = extra.get("exercise_format", "multiple_choice")
        return f"""You are an English teacher. Create a prepositions quiz question for A2-B1 level students.
{history_note}
Target CEFR level: {cefr_band}.
Selected subtype (MANDATORY): {subtype}
Subtype scope: {subtype_desc}
Subtype examples: {subtype_examples}
Exercise format for this item (MANDATORY): {exercise_format}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "question": "Sentence with a missing preposition ___ (max 100 chars)",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation_ua": "Ukrainian explanation why this preposition is correct (max 150 chars, natural Ukrainian)"
}}
Rules:
- Question and options in English only
- Always provide exactly 4 options
- correct_index is 0-based and must point to the only correct option
- Keep one clear target gap (only one blank ___)
- Rotate difficulty inside A2-B1: basic use + natural collocations
- Never repeat very similar stems from recent history
- Test prepositions across these families over time:
  1) Time (in/on/at + during/for/since/until/by/within)
  2) Place/Position (in/on/at + under/over/between/among/next to/opposite/behind/in front of)
  3) Movement (to/towards/into/out of/across/through/past/along)
  4) Dependent prepositions (adjective + preposition, verb + preposition)
  5) Fixed prepositional phrases (by mistake, on foot, in a hurry, at last...)
- Exercise format behavior:
  - multiple_choice: classic single sentence gap fill
  - sentence_transformation: ask to complete transformed sentence with one preposition
  - error_correction: show wrong sentence in question, options are candidate corrections/prepositions
  - contextual_gap_fill: short 1-2 sentence context, but only ONE target gap for Telegram quiz
- Explanation in Ukrainian
{LANGUAGE_CENSOR}"""

    if rubric == "vocabulary_15":
        mode = extra.get("theme_mode", "atlas")
        title = str(extra.get("theme_title", "")).strip()
        scope = str(extra.get("theme_scope", "")).strip()
        cefr = extra.get("cefr_band", "A2-B1 mixed")
        title_line = f'Fixed theme title (use EXACTLY this string for "theme_title"): "{title}"\n' if title else ""
        mode_rules = ""
        if mode == "gemini_freeform":
            mode_rules = (
                "You MUST invent theme_title yourself (English, 3–12 words) and build 15 words that match it.\n"
                "theme_title MUST NOT contain the numeral 15 or phrases like '15 words' — the card already shows 15 numbered items.\n"
            )
        else:
            mode_rules = (
                "theme_title in JSON must match the fixed title above (character-for-character).\n"
            )
        return f"""You are an English teacher. Create a vocabulary list post for Ukrainian students.
{title_line}Theme mode: {mode}
Theme scope: {scope}
Target CEFR level: {cefr} — use mixed A2–B1 vocabulary (clear, high-frequency items; short phrases allowed if they are standard chunks).
{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "theme_title": "English title for the card (see rules above)",
  "theme_title_ua": "optional Ukrainian gloss for logs (not shown on the card)",
  "words": [
    {{"word": "English word or short chunk", "ipa": "/IPA in slashes using International Phonetic Alphabet/", "ua": "natural Ukrainian translation or gloss"}},
    ... exactly 15 objects total ...
  ]
}}
Rules:
- "theme_title_ua": optional; card heading shows only "theme_title" in English
- "theme_title" must be a thematic heading only — do NOT put the numeral 15 in it (no '15 words', '15 verbs', etc.)
- Exactly 15 items in "words"
- Each "ipa" MUST be IPA inside slashes, e.g. /ˈwɜːrd/ — American or British is OK but be consistent within the list
- Single words preferred; two-word items only if idiomatic (e.g. phrasal verbs)
- Ukrainian must sound natural (not literal calques); follow Ukrainian language quality rules below
- Do not repeat words from the "avoid" list in spirit: vary lemmas and senses
{mode_rules}
{LANGUAGE_CENSOR}"""

    if rubric == "photo_relax":
        visual = str(extra.get("visual_theme", "nature landscape")).strip()
        tid = str(extra.get("theme_id", "")).strip()
        vstyle = str(extra.get("voice_style_instruction", "")).strip()
        if not vstyle:
            vstyle = PHOTO_RELAX_VOICE_STYLES[0]["instruction"]
        return f"""You write short English captions for a nature photo — like a real person, not a textbook.
Context: this post goes out on a Friday for people who need to breathe out after the week. The image is ONLY nature / landscape matching: {visual} (theme id: {tid}).

Writing style for THIS post (follow closely):
{vstyle}

Overall mood to capture: relief after the week, quiet, a little dreamy, the feeling that the weekend is near — inner "exhale", space to rest or imagine a small getaway. Warm, alive, lightly poetic sometimes, but NO pomposity, NO clichés like "this place will take your breath away" or "hidden gem".

{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "sentences": [
    "First sentence in English.",
    "Second sentence in English.",
    "Third sentence in English.",
    "Fourth sentence in English."
  ]
}}
Rules:
- English ONLY (no Ukrainian, no Russian).
- Exactly 4 separate strings in "sentences" — four full sentences total.
- Clear, natural English (roughly A2–B1): mostly simple grammar; you may use a slightly richer word here if it fits the tone — still easy for learners.
- Match the nature scene above; do NOT centre the text on rain, night, cities, streets, or crowds of people.
- You may use "you" or stay neutral — your choice.
- Do not repeat the same opening hook as in the recent history hints above.
- No hashtags; no emojis (prefer none).
- No bullet lists inside a sentence; each array item is one complete sentence."""

    if rubric == "interesting_cities":
        banned = extra.get("banned_cities") or []
        banned_note = ""
        if banned:
            banned_note = (
                f"\nDo NOT choose any of these city+country pairs again "
                f"(same idea, even if spelling varies): {banned[-40:]}\n"
            )
        return f"""You are a travel blogger and an English teacher. You write lively, emotional posts in English for Telegram and Instagram — like a real person, not a guidebook or Wikipedia.
Your style:
— conversational, warm, a little intimate; sometimes light humour
— no pomposity; avoid empty clichés like "this city will enchant you", "hidden gem", "must-visit destination", "breathtaking" without a real detail
— as if you're texting a friend, not selling a tour
— no dry facts, years, or dates; no encyclopaedic tone

Task: write ONE short post about ONE place — a city, a small town, or a capital. The whole caption body must be **only {INTERESTING_CITIES_SENTENCES_MIN} to {INTERESTING_CITIES_SENTENCES_MAX} sentences** (the "sentences" array — nothing else).

Content focus (no separate "hook" line, no dramatic opener): start straight into what feels real about the place — what makes it special, one or two concrete spots or small moments, maybe a brief personal aside. **Do NOT** add a punchy advertising-style hook at the start. **Do NOT** end with a question to the reader, a "what do you think?", or a tagged-on CTA — the last sentence should simply land (feeling or image), not ask something.

{banned_note}{history_note}
Return ONLY valid JSON, no markdown, no extra text:
{{
  "city_name": "English name of the place",
  "country": "English name of the country",
  "photo_query": "compact English keywords for stock photo search (place + country + one visual cue; avoid keyword spam)",
  "sentences": [
    "{INTERESTING_CITIES_SENTENCES_MIN} to {INTERESTING_CITIES_SENTENCES_MAX} strings only; each string is ONE complete sentence.",
    "..."
  ]
}}
Rules:
- The "sentences" array MUST have length between {INTERESTING_CITIES_SENTENCES_MIN} and {INTERESTING_CITIES_SENTENCES_MAX} inclusive.
- English ONLY in all fields (no Ukrainian, no Russian). No emoji, no flag symbols.
- Natural spoken English (B1-ish): clear for learners; avoid rare jargon.
- No numbered list markers in the text (no "1.", "2." at the start of a sentence).
- Pick ONE interesting place anywhere in the world — vary continents over time. Be specific.
- "photo_query": keywords for a strong photo of THAT place (Unsplash/Pexels/Pixabay).
- Do not repeat places from the banned list above."""

    if rubric == "travel_video_landmark":
        banned = extra.get("banned_places") or []
        banned_note = ""
        if banned:
            banned_note = (
                f"\nDo NOT choose any landmark+country pair from this list "
                f"(same place even if spelling varies): {banned[-60:]}\n"
            )
        cats = ", ".join(TRAVEL_VIDEO_LANDMARK_CATEGORIES)
        return f"""You are an English teacher. Pick ONE famous real-world landmark (a building, bridge, mountain, waterfall, temple, etc.) — NOT a whole city, NOT a vague region.
{banned_note}{history_note}
Allowed categories (pick exactly one for "category" — must match one string exactly):
{cats}
Do NOT use UNESCO as a category label. Do not focus the text on UNESCO listing; just describe the place for learners.
Return ONLY valid JSON, no markdown, no extra text:
{{
  "landmark_name": "English name of the landmark",
  "country": "English name of the country",
  "category": "one of the allowed category strings above (exact match)",
  "stock_query": "English keywords for stock VIDEO search (landmark + country + vertical/portrait; no quotes)",
  "narration": "Full English voiceover script for A2 learners: about 5–8 short sentences, 70–{TRAVEL_VIDEO_NARRATION_WORDS_MAX} words total, present simple mostly, simple vocabulary, British English style, no lists, no stage directions, no quotes, no emojis."
}}
Rules:
- English ONLY in all fields.
- "narration" is spoken English only — no Ukrainian, no Russian.
- Be factually plausible; do not invent dangerous or offensive content.
- Vary continents and landmark types over time when possible."""

    raise ValueError(f"Unknown rubric: {rubric}")

# ──────────────────────────────────────────────
# GEMINI / GROQ — ГЕНЕРАЦІЯ КОНТЕНТУ
# ──────────────────────────────────────────────
CRITICAL_ERRORS = {
    "INVALID_ARGUMENT", "API_KEY_INVALID", "PERMISSION_DENIED",
    "invalid_api_key", "authentication_failed", "account_suspended",
}

QUIZ_FALLBACK_BANK = {
    "grammar_quiz": {
        "question": "If it ___ tomorrow, we'll stay home.",
        "options": ["rains", "rained", "is raining", "will rain"],
        "correct_index": 0,
        "explanation_ua": "💡 У First Conditional після if вживаємо Present Simple: If it rains...",
    },
    "vocabulary_quiz": {
        "question": "At the airport, show your ___ at check-in.",
        "options": ["boarding pass", "recipe", "blanket", "ticket office"],
        "correct_index": 0,
        "explanation_ua": "💡 Boarding pass — це посадковий талон, який показують на реєстрації.",
    },
    "confusing_words_quiz": {
        "question": "Can I ___ your pen for a minute?",
        "options": ["borrow", "lend", "bring", "carry"],
        "correct_index": 0,
        "explanation_ua": "💡 Borrow = брати в когось, lend = давати комусь.",
    },
    "prepositions_quiz": {
        "question": "She has lived here ___ 2021.",
        "options": ["since", "for", "during", "until"],
        "correct_index": 0,
        "explanation_ua": "💡 Since вживаємо з початковою точкою в часі: since 2021.",
    },
}


def is_critical_error(error_text: str) -> bool:
    for ce in CRITICAL_ERRORS:
        if ce.lower() in error_text.lower():
            return True
    return False


def _normalize_quote_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9а-щьюяєіїґ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9а-щьюяєіїґ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _ipa_square_brackets(ipa: str) -> str:
    """IPA для відображення у квадратних дужках як у EnglishCardsGenerator."""
    s = str(ipa).strip()
    if len(s) >= 2 and s.startswith("/") and s.endswith("/"):
        s = s[1:-1].strip()
    return s


def get_cefr_band_for_today() -> str:
    # Пн-Вт A2, Ср-Чт A2+, Пт-Сб B1, Нд mixed review
    weekday = datetime.now(TZ).weekday()  # Mon=0 ... Sun=6
    if weekday in (0, 1):
        return "A2"
    if weekday in (2, 3):
        return "A2+"
    if weekday in (4, 5):
        return "B1"
    return "A2-B1 mixed review"


def build_quiz_signature(rubric: str, data: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    q = _normalize_text(str(data.get("question", "")))
    signature_parts = [rubric, q]
    for key in (
        "grammar_subtype",
        "grammar_format",
        "grammar_sentence_type",
        "vocab_topic_name",
        "confusing_subtype",
        "confusing_format",
        "prepositions_subtype",
        "exercise_format",
    ):
        if key in extra:
            signature_parts.append(_normalize_text(str(extra[key])))
    return "|".join(signature_parts)


def validate_quiz_payload(rubric: str, data: dict, history: list, recent_signatures: list, extra: dict | None = None) -> tuple[bool, str]:
    extra = extra or {}
    question = str(data.get("question", "")).strip()
    options = data.get("options", [])
    correct = data.get("correct_index", -1)
    explanation = str(data.get("explanation_ua", "")).strip()

    if not question:
        return False, "empty question"
    if len(question) > 300:
        return False, "question too long"
    if not isinstance(options, list) or len(options) != 4:
        return False, "options must be exactly 4"
    clean_options = [str(o).strip() for o in options]
    if any(not o for o in clean_options):
        return False, "empty option"
    if len(set(o.lower() for o in clean_options)) < 4:
        return False, "duplicate options"
    if not isinstance(correct, int) or correct < 0 or correct > 3:
        return False, "invalid correct_index"
    if len(explanation) < 8:
        return False, "explanation too short"

    q_norm = _normalize_text(question)
    if len(q_norm.split()) < 4:
        return False, "question too short"
    hist_norm = [_normalize_text(h) for h in history[-30:]]
    if any(q_norm and (q_norm in h or h in q_norm) for h in hist_norm if h):
        return False, "question too similar to history"

    signature = build_quiz_signature(rubric, data, extra)
    recent_sig_norm = {_normalize_text(s) for s in recent_signatures}
    if _normalize_text(signature) in recent_sig_norm:
        return False, "signature repeated recently"

    if "___" not in question:
        if rubric in {"grammar_quiz", "vocabulary_quiz", "prepositions_quiz"}:
            return False, "missing blank marker ___"

    return True, "ok"


def validate_quote_motivation(data: dict, used_history: list) -> tuple[bool, str]:
    quote_en = str(data.get("quote_en", "")).strip()
    quote_ua = str(data.get("quote_ua", "")).strip()

    if not quote_en or not quote_ua:
        return False, "missing required fields"

    words = [w for w in quote_en.split() if w.strip()]
    if len(words) < 6 or len(words) > 12:
        return False, f"word count out of range ({len(words)})"

    quote_norm = _normalize_quote_text(quote_en)
    if any(pattern in quote_norm for pattern in WEAK_QUOTE_PATTERNS):
        return False, "contains weak cliche phrase"

    if len(quote_ua) < 12:
        return False, "ukrainian translation too short"
    if quote_en.lower() == quote_ua.lower():
        return False, "translation equals english text"

    recent_norm = [_normalize_quote_text(item) for item in used_history[-25:]]
    if quote_norm and any(quote_norm in h or h in quote_norm for h in recent_norm if h):
        return False, "too similar to recent history"

    return True, "ok"


def build_vocabulary_15_word_signature(words: list) -> str:
    parts = []
    for item in words:
        if not isinstance(item, dict):
            continue
        w = _normalize_text(str(item.get("word", "")))
        if w:
            parts.append(w)
    return "|".join(sorted(parts))


def validate_vocabulary_15(
    data: dict,
    used_history: list,
    recent_signatures: list,
    extra: dict | None = None,
) -> tuple[bool, str]:
    extra = extra or {}
    mode = extra.get("theme_mode", "atlas")
    title = str(data.get("theme_title", "")).strip()
    words = data.get("words")

    if not title:
        return False, "empty theme_title"
    t_ua = str(data.get("theme_title_ua", "")).strip()
    if len(t_ua) > 120:
        return False, "theme_title_ua too long"
    if mode in ("atlas", "holiday"):
        exp = str(extra.get("theme_title", "")).strip()
        if exp and _normalize_text(exp) != _normalize_text(title):
            return False, "theme_title does not match expected"

    if not isinstance(words, list) or len(words) != 15:
        return False, "words must be a list of exactly 15 items"

    seen = set()
    for i, item in enumerate(words):
        if not isinstance(item, dict):
            return False, f"word #{i+1} invalid type"
        w = str(item.get("word", "")).strip()
        ipa = str(item.get("ipa", "")).strip()
        ua = str(item.get("ua", "")).strip()
        if not w or not ua:
            return False, f"word #{i+1} empty word or ua"
        if len(w) > 48:
            return False, f"word #{i+1} too long"
        if ipa.count("/") < 2:
            return False, f"word #{i+1} ipa must be IPA in slashes"
        if len(ua) > 120:
            return False, f"word #{i+1} ua too long"
        key = _normalize_text(w)
        if not key:
            return False, f"word #{i+1} empty after normalize"
        if key in seen:
            return False, "duplicate words"
        seen.add(key)

    sig = build_vocabulary_15_word_signature(words)
    if not sig:
        return False, "empty signature"
    recent_norm = {_normalize_text(s) for s in recent_signatures}
    if _normalize_text(sig) in recent_norm:
        return False, "word set repeated recently"

    return True, "ok"


async def generate_vocabulary_15_content(
    history: list,
    extra: dict | None,
    recent_signatures: list,
    max_attempts: int = 3,
) -> dict:
    extra = extra or {}
    last_reason = "unknown"
    for attempt in range(1, max_attempts + 1):
        data = await generate_content("vocabulary_15", history, extra)
        ok, reason = validate_vocabulary_15(data, history, recent_signatures, extra)
        if ok:
            if attempt > 1:
                log.info(f"✅ vocabulary_15 validated on retry #{attempt}")
            return data
        last_reason = reason
        preview = build_vocabulary_15_word_signature(data.get("words", []))[:120]
        log.warning(f"⚠️ vocabulary_15 rejected (attempt {attempt}/{max_attempts}): {reason} | sig='{preview}'")
        history = history + [preview or "reject"]

    raise RuntimeError(f"vocabulary_15 failed validation after {max_attempts} attempts: {last_reason}")


async def call_gemini(
    api_key: str,
    prompt: str,
    model: str = "gemini-2.0-flash-lite",
    max_output_tokens: int | None = None,
) -> dict:
    """Викликає Gemini модель з конкретним API ключем."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    mt = max_output_tokens if max_output_tokens is not None else 1000
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": mt},
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

    if rubric == "vocabulary_15":
        max_tok = 3200
    elif rubric == "photo_relax":
        max_tok = 1000
    elif rubric == "interesting_cities":
        max_tok = 1200
    elif rubric == "travel_video_landmark":
        max_tok = 2200
    else:
        max_tok = 1000
    for model in GEMINI_MODELS:
        for i, api_key in enumerate(GEMINI_API_KEYS, 1):
            try:
                log.info(f"🔑 Trying {model} key {i}/{len(GEMINI_API_KEYS)} for [{rubric}]")
                result = await call_gemini(api_key, prompt, model, max_output_tokens=max_tok)
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


async def generate_quote_motivation_content(history: list, extra: dict = None, max_attempts: int = 3) -> dict:
    theme_name = (extra or {}).get("quote_theme_name", "unknown")
    log.info(f"📣 [NF] quote_motivation generation started | theme='{theme_name}' | max_attempts={max_attempts}")
    for attempt in range(1, max_attempts + 1):
        data = await generate_content("quote_motivation", history, extra)
        is_valid, reason = validate_quote_motivation(data, history)
        if is_valid:
            if attempt > 1:
                log.info(f"✅ quote_motivation validated on retry #{attempt}")
            log.info(f"📣 [NF] quote_motivation accepted | theme='{theme_name}' | attempt={attempt}")
            return data

        rejected = str(data.get("quote_en", "")).strip()[:120]
        log.warning(f"⚠️ quote_motivation rejected (attempt {attempt}/{max_attempts}): {reason} | quote='{rejected}'")
        log.warning(f"📣 [NF] quote_motivation rejected | theme='{theme_name}' | attempt={attempt} | reason='{reason}'")
        history = history + [rejected]

    raise RuntimeError(f"quote_motivation failed validation after {max_attempts} attempts")


async def generate_quiz_content_with_validation(
    rubric: str,
    history: list,
    extra: dict | None = None,
    recent_signatures: list | None = None,
    max_attempts: int = 3,
) -> tuple[dict, int, str]:
    recent_signatures = recent_signatures or []
    last_reason = "unknown"
    for attempt in range(1, max_attempts + 1):
        data = await generate_content(rubric, history, extra or {})
        ok, reason = validate_quiz_payload(rubric, data, history, recent_signatures, extra or {})
        if ok:
            return data, attempt, "ok"
        last_reason = reason
        log.warning(f"⚠️ {rubric} rejected (attempt {attempt}/{max_attempts}): {reason}")
        history = history + [str(data.get("question", "")).strip()[:120]]

    fallback = QUIZ_FALLBACK_BANK.get(rubric, {})
    if fallback:
        log.warning(f"🛟 {rubric} fallback used after {max_attempts} failed attempts: {last_reason}")
        return fallback, max_attempts + 1, f"fallback:{last_reason}"
    raise RuntimeError(f"{rubric} failed validation after {max_attempts} attempts: {last_reason}")

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
.brand {{ font-size: 36px; font-weight: 300; letter-spacing: 2px;
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


def _normalize_photo_relax_sentences(data: dict) -> list[str] | None:
    raw = data.get("sentences")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    out = [str(x).strip() for x in raw]
    if any(not x for x in out):
        return None
    return out


def build_photo_relax_signature(sentences: list[str]) -> str:
    return _normalize_text(" ".join(sentences))[:240]


def validate_photo_relax(
    data: dict,
    used_history: list,
    recent_signatures: list,
    extra: dict | None = None,
) -> tuple[bool, str]:
    _ = used_history
    _ = extra
    sents = _normalize_photo_relax_sentences(data)
    if not sents:
        return False, "need exactly 4 non-empty sentences in sentences[]"
    cy = re.compile(r"[\u0400-\u04FF]")
    for i, s in enumerate(sents):
        if len(s) > 300:
            return False, f"sentence {i+1} too long"
        if cy.search(s):
            return False, "Cyrillic not allowed"
    sig = build_photo_relax_signature(sents)
    recent_norm = {_normalize_text(x) for x in recent_signatures if x}
    if sig in recent_norm:
        return False, "text too similar to recent post"
    data["sentences"] = sents
    return True, "ok"


async def generate_photo_relax_content(
    history: list,
    extra: dict | None,
    recent_signatures: list,
    max_attempts: int = 3,
) -> dict:
    extra = extra or {}
    last_reason = "unknown"
    hist = list(history)
    for attempt in range(1, max_attempts + 1):
        data = await generate_content("photo_relax", hist, extra)
        ok, reason = validate_photo_relax(data, hist, recent_signatures, extra)
        if ok:
            if attempt > 1:
                log.info(f"✅ photo_relax validated on retry #{attempt}")
            return data
        last_reason = reason
        preview = " ".join(str(x) for x in (data.get("sentences") or []))[:100]
        log.warning(f"⚠️ photo_relax rejected (attempt {attempt}/{max_attempts}): {reason} | {preview}")
        hist = hist + [preview or "reject"]
    raise RuntimeError(f"photo_relax failed validation after {max_attempts} attempts: {last_reason}")


def _has_emoji_or_flag(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x1F1E6 <= o <= 0x1F1FF:
            return True
    return False


def _interesting_cities_place_key(city: str, country: str) -> str:
    return _normalize_text(f"{city.strip()}|{country.strip()}")


def _normalize_interesting_cities_sentences(data: dict) -> list[str] | None:
    raw = data.get("sentences")
    if not isinstance(raw, list):
        return None
    out = [str(x).strip() for x in raw]
    out = [x for x in out if x]
    n = len(out)
    if n < INTERESTING_CITIES_SENTENCES_MIN or n > INTERESTING_CITIES_SENTENCES_MAX:
        return None
    return out


def build_interesting_cities_signature(data: dict) -> str:
    city = str(data.get("city_name", "")).strip()
    country = str(data.get("country", "")).strip()
    sents = _normalize_interesting_cities_sentences(data)
    body = " ".join(sents) if sents else ""
    return _normalize_text(f"{city}|{country}|{body}")[:400]


def validate_interesting_cities(
    data: dict,
    used_history: list,
    recent_signatures: list,
    extra: dict | None = None,
) -> tuple[bool, str]:
    _ = used_history
    extra = extra or {}
    city = str(data.get("city_name", "")).strip()
    country = str(data.get("country", "")).strip()
    if not city or not country:
        return False, "empty city_name or country"
    pq = str(data.get("photo_query", "")).strip()
    if not pq:
        pq = f"{city} {country} city travel landscape architecture"
    data["photo_query"] = pq
    sents = _normalize_interesting_cities_sentences(data)
    if not sents:
        return (
            False,
            f"sentences[] must have {INTERESTING_CITIES_SENTENCES_MIN}–{INTERESTING_CITIES_SENTENCES_MAX} non-empty items",
        )
    cy = re.compile(r"[\u0400-\u04FF]")
    for i, s in enumerate(sents):
        if len(s) > 220:
            return False, f"sentence {i+1} too long"
        if cy.search(s) or _has_emoji_or_flag(s):
            return False, f"sentence {i+1}: Cyrillic or emoji not allowed"
    if cy.search(city) or cy.search(country) or _has_emoji_or_flag(city) or _has_emoji_or_flag(country):
        return False, "city/country must be Latin letters only, no emoji"

    pk = _interesting_cities_place_key(city, country)
    banned = {_normalize_text(x) for x in (extra.get("banned_cities") or []) if x}
    if pk in banned:
        return False, "city already used (banned list)"

    sig = build_interesting_cities_signature(
        {"city_name": city, "country": country, "sentences": sents}
    )
    recent_norm = {_normalize_text(x) for x in recent_signatures if x}
    if sig in recent_norm:
        return False, "duplicate recent signature"

    data["city_name"] = city
    data["country"] = country
    data["photo_query"] = pq
    data["sentences"] = sents
    return True, "ok"


async def generate_interesting_cities_content(
    history: list,
    extra: dict | None,
    recent_signatures: list,
    max_attempts: int = 3,
) -> dict:
    extra = extra or {}
    last_reason = "unknown"
    hist = list(history)
    for attempt in range(1, max_attempts + 1):
        data = await generate_content("interesting_cities", hist, extra)
        if isinstance(data, list):
            data = data[0] if data else {}
        ok, reason = validate_interesting_cities(data, hist, recent_signatures, extra)
        if ok:
            if attempt > 1:
                log.info(f"✅ interesting_cities validated on retry #{attempt}")
            return data
        last_reason = reason
        preview = f"{data.get('city_name')}|{data.get('country')}"
        log.warning(
            f"⚠️ interesting_cities rejected (attempt {attempt}/{max_attempts}): {reason} | {preview}"
        )
        hist = hist + [preview or "reject"]
    raise RuntimeError(
        f"interesting_cities failed validation after {max_attempts} attempts: {last_reason}"
    )


def build_interesting_cities(data: dict, photo_b64: str) -> str:
    _ = data  # copy: Telegram caption only
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: 1080px;
    height: 1920px;
    overflow: hidden;
    background: #111;
  }}
  .ic-photo-full {{
    width: 1080px;
    height: 1920px;
    background-image: url('data:image/jpeg;base64,{photo_b64}');
    background-size: cover;
    background-position: center;
  }}
</style>
</head>
<body>
  <div class="ic-photo-full" role="img" aria-label="City photo"></div>
</body>
</html>"""


def build_photo_relax(data: dict, photo_b64: str) -> str:
    _ = data  # sentences: Telegram caption only
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: 1080px;
    height: 1920px;
    overflow: hidden;
    background: #111;
  }}
  .pr-photo-full {{
    width: 1080px;
    height: 1920px;
    background-image: url('data:image/jpeg;base64,{photo_b64}');
    background-size: cover;
    background-position: center;
  }}
</style>
</head>
<body>
  <div class="pr-photo-full" role="img" aria-label="Nature photo"></div>
</body>
</html>"""


def build_daily_phrase(data: dict, photo_b64: str) -> str:
    phrase = _safe_html(data.get("phrase_en", ""))
    ex_en  = _safe_html(data.get("example_en", ""))
    ex_ua  = _safe_html(data.get("example_ua", ""))
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    blocks = f"""
  <div class="glass-block" style="height:380px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:flex-start; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:36px; font-weight:600; letter-spacing:3px; color:#c9a84c;
                text-transform:uppercase; margin-bottom:28px; {ts}">
      Daily Phrase
    </div>
    <div style="font-size:clamp(46px,5vw,64px); font-weight:700; color:#fbf8f5;
                {ts} line-height:1.25; flex:1; display:flex; align-items:center;">
      {phrase}
    </div>
  </div>
  <div class="glass-block" style="height:550px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:clamp(42px,4.5vw,58px); font-weight:600; color:#fbf8f5;
                {ts} line-height:1.3; margin-bottom:4px;">
      {ex_en}
    </div>
    <div class="gold-line"></div>
    <div style="font-size:clamp(40px,4.3vw,56px); font-weight:300; color:rgba(251,248,245,0.85);
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
    topic_name = _safe_html(category.get("name", ""))
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    block_height, font_en, font_ua = 280, 50, 44
    log.info(f"📐 Situation: fixed height={block_height}px")

    # Тема над блоками без розмиття
    topic_header = f"""
  <div style="width:100%; text-align:left; padding:0 8px; margin-bottom:4px;">
    <div style="font-size:48px; font-weight:600; color:#c9a84c;
                letter-spacing:2px; {ts} line-height:1.1;">
      {topic_name}
    </div>
  </div>"""

    blocks = topic_header
    for p in phrases[:5]:
        en = _safe_html(p.get("en", ""))
        ua = _safe_html(p.get("ua", ""))
        blocks += f"""
  <div class="glass-block" style="height:{block_height}px; padding:24px 52px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;
       background: rgba(8,10,16,0.74); border: 1px solid rgba(201,168,76,0.28);
       backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);">
    <div style="font-size:{font_en}px; font-weight:700; color:#ffffff;
                {ts} line-height:1.25; margin-bottom:12px; color:#fbf8f5;">{en}</div>
    <div style="font-size:{font_ua}px; font-weight:300; color:rgba(251,248,245,0.84);
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
    quote_en = _safe_html(data.get("quote_en", ""))
    quote_ua = _safe_html(data.get("quote_ua", ""))
    ts  = "text-shadow: 0 2px 8px rgba(0,0,0,0.85), 0 1px 3px rgba(0,0,0,0.95);"
    ts2 = "text-shadow: 0 2px 6px rgba(0,0,0,0.75), 0 1px 3px rgba(0,0,0,0.85);"

    blocks = f"""
  <div class="glass-block" style="height:560px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:flex-start; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:36px; font-weight:600; letter-spacing:3px; color:#c9a84c;
                text-transform:uppercase; margin-bottom:28px; {ts}">
      Motivation
    </div>
    <div style="font-size:clamp(46px,5vw,64px); font-weight:700; color:#fbf8f5;
                {ts} line-height:1.3; flex:1; display:flex; align-items:center;">
      {quote_en}
    </div>
    <div class="gold-line" style="margin:20px 0 0 0;"></div>
  </div>
  <div class="glass-block" style="height:560px; padding:48px 60px; display:flex;
       flex-direction:column; justify-content:center; overflow:hidden; box-sizing:border-box;">
    <div style="font-size:clamp(46px,5vw,64px); font-weight:300; color:rgba(251,248,245,0.88);
                {ts2} line-height:1.3; flex:1; display:flex; align-items:center;">
      {quote_ua}
    </div>
    <div style="text-align:right; margin-top:20px;">
      <span class="brand">Improve Your <span class="eng">English</span></span>
    </div>
  </div>"""
    return html_base(photo_b64, blocks)


def vocabulary_15_heading_display(title: str) -> str:
    """Заголовок без «15» — номери вже в списку."""
    s = (title or "").strip()
    if not s:
        return s
    s = re.sub(r"^15\s*[-–—:.]\s*", "", s)
    s = re.sub(r"^15\s+", "", s)
    s = re.sub(r"\s*[-–—]\s*15\s+words?\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_vocabulary_15(data: dict) -> str:
    """
    Кремова картка в стилі EnglishCardsGenerator.html + авто-підгін шрифту (JS),
    як fitTextInBox у генераторі: довгі слова/рядки зменшують базовий font-size, поки блок вміщується.
    """
    raw_title = str(data.get("theme_title", "Vocabulary")).strip()
    displayed = vocabulary_15_heading_display(raw_title)
    title_en = _safe_html(displayed if displayed else (raw_title or "Vocabulary"))
    title_block = f'<p class="v15-heading"><strong>{title_en}</strong></p>'

    words = data.get("words", [])[:15]
    items: list[str] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        w = _safe_html(str(item.get("word", "")).strip())
        ipa_raw = str(item.get("ipa", "")).strip()
        ipa_inner = _safe_html(_ipa_square_brackets(ipa_raw))
        ua = _safe_html(str(item.get("ua", "")).strip())
        items.append(
            f'<li class="v15-li">{w} <span class="v15-ipa">[{ipa_inner}]</span> – {ua}</li>'
        )
    list_html = "\n".join(items)

    return f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,500&family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --ink: #2a2520;
    --muted: #6b635c;
    --gold-warm: #c9a84c;
    --stone: #a69f98;
    --cream: linear-gradient(160deg, #e8e2d9 0%, #f0ede9 45%, #d6d2ce 100%);
  }}
  body {{
    width: 1080px;
    height: 1920px;
    overflow: hidden;
    font-family: 'Cormorant Garamond', serif;
    color: var(--ink);
    background: #e8e2d9;
  }}
  .v15-card {{
    position: relative;
    width: 1080px;
    height: 1920px;
    background: var(--cream);
    overflow: hidden;
  }}
  .v15-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    opacity: 0.5;
    background-image: url('https://www.transparenttextures.com/patterns/cream-pixels.png');
  }}
  .v15-card::after {{
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    opacity: 0.2;
    background-image:
      radial-gradient(circle, var(--stone) 1px, transparent 1px),
      radial-gradient(circle, var(--gold-warm) 1px, transparent 1px);
    background-size: 40px 40px, 65px 65px;
    background-position: 0 0, 20px 20px;
  }}
  .v15-inner {{
    position: relative;
    z-index: 2;
    height: 100%;
    min-height: 1920px;
    box-sizing: border-box;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    align-content: stretch;
    padding: 8% 7% 4.5%;
  }}
  .v15-brand {{
    display: flex;
    justify-content: flex-end;
    align-items: baseline;
    flex: 0 0 auto;
    width: 100%;
    margin-bottom: 2rem;
    font-size: 12px;
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    letter-spacing: 0.125em;
    color: #0a0a0a;
  }}
  .v15-brand-stack {{
    display: flex;
    flex-wrap: wrap;
    flex-direction: row;
    justify-content: flex-end;
    align-items: baseline;
    gap: 0.25em;
    line-height: 1.1;
    max-width: 100%;
  }}
  .v15-brand-gold {{
    font-style: italic;
    font-weight: 300;
    color: var(--gold-warm);
  }}
  .v15-main {{
    min-height: 0;
    min-width: 0;
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 6px 0 10px;
    text-align: left;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .v15-fit-root {{
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    width: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    box-sizing: border-box;
  }}
  .v15-fit-content {{
    flex: 0 0 auto;
    width: 100%;
    min-width: 0;
    font-size: 32px;
    box-sizing: border-box;
  }}
  .v15-heading {{
    font-family: 'Cormorant Garamond', serif;
    font-weight: 700;
    font-size: 1em;
    line-height: 1.42;
    color: var(--ink);
    margin-bottom: 0.35em;
    word-break: break-word;
  }}
  .v15-heading strong {{
    font-weight: 700;
  }}
  .v15-rule {{
    display: block;
    height: 2px;
    margin: 0.65em 0 0.85em;
    border: 0;
    background: rgba(42, 37, 32, 0.42);
  }}
  .v15-ol {{
    list-style: decimal;
    list-style-position: inside;
    padding-left: 0;
    margin: 0;
    width: 100%;
    max-width: 100%;
    font-size: 1em;
    font-weight: 600;
    line-height: 1.42;
    word-break: break-word;
    overflow-wrap: anywhere;
  }}
  .v15-li {{
    margin: 0.12em 0;
    padding-left: 0.15em;
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .v15-ipa {{
    font-weight: 400;
    color: var(--ink);
    opacity: 0.92;
  }}
  .v15-tagline {{
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.556em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    width: max-content;
    max-width: 100%;
    margin-left: auto;
    margin-right: auto;
    margin-top: 2rem;
    padding: 0.15rem 12px 0;
    line-height: 1.4;
    flex-shrink: 0;
  }}
</style>
</head>
<body>
<div class="v15-card">
  <div class="v15-inner">
    <div class="v15-brand">
      <div class="v15-brand-stack">
        <span>Improve Your</span><span class="v15-brand-gold">English</span>
      </div>
    </div>
    <div class="v15-main">
      <div class="v15-fit-root">
        <div class="v15-fit-content">
        {title_block}
        <hr class="v15-rule" />
        <ol class="v15-ol">
          {list_html}
        </ol>
        </div>
      </div>
    </div>
    <div class="v15-tagline">Learn • Grow • Succeed</div>
  </div>
</div>
</body>
</html>"""


# vocabulary_15: як EnglishCardsGenerator — бренд/теглайн через fitFontToMaxWidth; список — .v15-fit-content
# (не .v15-fit-root). Верхня межа шрифту списку не фіксована 56px — інакше лишається порожнє місце по вертикалі.
VOCABULARY_15_FIT_JS = """
() => {
  var BRAND_TAGLINE_WIDTH_FRAC = 0.5;
  var BRAND_FONT_MAX_FRAC = 0.28;
  var TAGLINE_FONT_MAX_FRAC = 0.2;

  function fitFontToMaxWidth(sizeEl, measureEl, targetW, minPx, maxPx) {
    if (!sizeEl || !measureEl || targetW <= 0) {
      return;
    }
    var lo = minPx;
    var hi = Math.max(minPx, maxPx);
    var i = 0;
    for (i = 0; i < 32; i++) {
      var mid = (lo + hi) / 2;
      sizeEl.style.fontSize = mid + "px";
      void measureEl.offsetWidth;
      var sw = measureEl.scrollWidth;
      if (sw <= targetW) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    sizeEl.style.fontSize = lo + "px";
  }

  function fitsBox(box, w, h) {
    var pad = 8;
    return box.scrollWidth <= w + 2 && box.scrollHeight <= h - pad;
  }

  function fitOnce() {
    var card = document.querySelector(".v15-card");
    var main = document.querySelector(".v15-main");
    var box = document.querySelector(".v15-fit-content");
    var inner = document.querySelector(".v15-inner");
    var brand = document.querySelector(".v15-brand");
    var stack = document.querySelector(".v15-brand-stack");
    var tag = document.querySelector(".v15-tagline");
    if (!main || !box || !card) {
      return { ok: false, reason: "missing_nodes" };
    }
    var cardW = Math.floor(card.clientWidth || 1080);
    var targetHalf = Math.floor(cardW * BRAND_TAGLINE_WIDTH_FRAC);
    fitFontToMaxWidth(brand, stack, targetHalf, 8, cardW * BRAND_FONT_MAX_FRAC);
    if (tag) {
      fitFontToMaxWidth(tag, tag, targetHalf, 6, cardW * TAGLINE_FONT_MAX_FRAC);
    }
    void inner && inner.offsetHeight;
    void main.offsetHeight;

    var mr = main.getBoundingClientRect();
    var w = Math.floor(mr.width);
    var h = Math.floor(mr.height);
    if (h < 350 && inner) {
      var ir = inner.getBoundingClientRect();
      var bh = brand ? brand.getBoundingClientRect().height : 0;
      var th = tag ? tag.getBoundingClientRect().height : 0;
      h = Math.max(500, Math.floor(ir.height - bh - th - 48));
    }
    if (w < 200 || h < 350) {
      return { ok: false, reason: "bad_box", w: w, h: h };
    }
    var MIN_PX = Math.max(18, Math.floor(cardW * 0.014));
    var MAX_PX = Math.min(
      Math.min(cardW * 0.095, (card.clientHeight || 1920) * 0.11),
      Math.floor(h * 0.072),
      Math.floor(w * 0.12)
    );
    if (MAX_PX <= MIN_PX) {
      MAX_PX = MIN_PX + 20;
    }
    var lo = MIN_PX;
    var hi = MAX_PX;
    var j = 0;
    for (j = 0; j < 52; j++) {
      var mid = (lo + hi) / 2;
      box.style.fontSize = mid + "px";
      void box.offsetHeight;
      if (fitsBox(box, w, h)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    box.style.fontSize = lo + "px";
    void box.offsetHeight;
    if (!fitsBox(box, w, h)) {
      var s = lo;
      for (var k = 0; k < 48; k++) {
        s -= 0.75;
        if (s < 14) {
          break;
        }
        box.style.fontSize = s + "px";
        void box.offsetHeight;
        if (fitsBox(box, w, h)) {
          lo = s;
          break;
        }
      }
    }
    return {
      ok: true,
      w: w,
      h: h,
      fontPx: Math.round(lo * 100) / 100,
      scrollH: box.scrollHeight,
      scrollW: box.scrollWidth,
      mainClientH: main.clientHeight,
      brandPx: brand ? Math.round(parseFloat(getComputedStyle(brand).fontSize) * 100) / 100 : null,
      tagPx: tag ? Math.round(parseFloat(getComputedStyle(tag).fontSize) * 100) / 100 : null,
    };
  }
  var fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  return fontsReady.then(function () {
    return new Promise(function (resolve) {
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          resolve(fitOnce());
        });
      });
    });
  });
}
"""


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
            if "v15-fit-content" in html:
                try:
                    r1 = await page.evaluate(VOCABULARY_15_FIT_JS)
                    log.info(f"📐 vocabulary_15 fit [1]: {r1}")
                    await asyncio.sleep(0.05)
                    r2 = await page.evaluate(VOCABULARY_15_FIT_JS)
                    log.info(f"📐 vocabulary_15 fit [2]: {r2}")
                except Exception as e:
                    log.warning(f"⚠️ vocabulary_15 font-fit evaluate: {e}")
                await asyncio.sleep(0.5)
            else:
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
# TELEGRAM — ПУБЛІКАЦІ
# ──────────────────────────────────────────────
TELEGRAM_CAPTION_MAX = 1024


def clip_telegram_caption(text: str) -> tuple[str, bool]:
    text = (text or "").strip()
    if len(text) <= TELEGRAM_CAPTION_MAX:
        return text, False
    return text[: TELEGRAM_CAPTION_MAX - 1].rstrip() + "…", True


def build_photo_relax_caption(data: dict) -> str:
    raw = data.get("sentences")
    if not isinstance(raw, list):
        return ""
    lines = [str(x).strip() for x in raw[:4] if str(x).strip()]
    return "\n\n".join(lines)


def build_interesting_cities_caption(data: dict) -> str:
    city = str(data.get("city_name", "")).strip()
    country = str(data.get("country", "")).strip()
    sents = _normalize_interesting_cities_sentences(data)
    if not sents:
        return f"{city}, {country}".strip(", ")
    body = "\n\n".join(sents)
    return f"{city}, {country}\n\n{body}"


async def send_photo_to_telegram(
    png_bytes: bytes, rubric: str, caption: str | None = None
) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    cap = (caption or "").strip()
    form_data: dict = {"chat_id": TELEGRAM_CHAT_ID}
    if cap:
        form_data["caption"] = cap
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    data=form_data,
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

    if not isinstance(question, str):
        log.error(f"❌ Quiz question has invalid type for [{rubric}]")
        return False
    question = question.strip()

    if not isinstance(options, list):
        log.error(f"❌ Quiz options invalid type for [{rubric}]: {type(options)}")
        return False
    options = [str(opt).strip() for opt in options if str(opt).strip()]

    if not isinstance(correct, int):
        log.error(f"❌ Quiz correct_index invalid type for [{rubric}]: {type(correct)}")
        return False

    if not isinstance(explanation, str):
        explanation = str(explanation)
    explanation = explanation.strip()

    if not question or len(question) > 300:
        log.error(f"❌ Quiz question invalid length for [{rubric}]: len={len(question)}")
        return False
    if len(options) != 4:
        log.error(f"❌ Quiz options count must be 4 for [{rubric}], got={len(options)}")
        return False
    if any(len(opt) > 100 for opt in options):
        log.error(f"❌ Quiz option too long for [{rubric}]")
        return False
    if correct < 0 or correct >= len(options):
        log.error(f"❌ Quiz correct_index out of range for [{rubric}]: {correct}")
        return False
    if len(explanation) > 200:
        explanation = explanation[:200]
    if explanation and not explanation.lstrip().startswith("💡"):
        explanation = f"💡 {explanation}"

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
            photo_query = await get_daily_phrase_photo_query(history_mgr)
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
                photo_query = await get_situation_photo_query(history_mgr, {})
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
                photo_query = await get_situation_photo_query(history_mgr, category)
                await history_mgr.advance_situation_index()

        elif rubric == "quote_motivation":
            theme_idx = await history_mgr.get_quote_theme_index()
            theme = QUOTE_THEMES[theme_idx]
            extra = {
                "quote_theme_name": theme["name"],
                "quote_theme_description": theme["description"],
            }
            log.info(f"🧭 quote_motivation selected theme: {theme['name']}")
            log.info(f"📣 [NF] quote_motivation theme selected | index={theme_idx} | name='{theme['name']}'")
            photo_query = await get_quote_photo_query(history_mgr)

        elif rubric == "vocabulary_15":
            theme_pick = await pick_vocabulary_15_theme(history_mgr)
            extra = {**theme_pick, "cefr_band": "A2-B1 mixed"}
            log.info(
                f"📚 vocabulary_15 theme_mode={extra.get('theme_mode')} | "
                f"title='{extra.get('theme_title', '')[:80]}'"
            )

        elif rubric == "photo_relax":
            extra = await pick_photo_relax_theme(history_mgr)
            photo_query = extra["photo_query"]
            log.info(
                f"🌿 photo_relax visual_theme={extra.get('theme_id')} "
                f"voice_style={extra.get('voice_style_id')} query='{photo_query}'"
            )

        elif rubric == "interesting_cities":
            extra = {}

        photo_url = None
        photo_b64 = None
        ic_data = None

        if rubric == "interesting_cities":
            history_ic = await history_mgr.get_used(rubric)
            recent_ic_sig = await history_mgr.get_recent_signatures(
                rubric, limit=QUIZ_SIGNATURE_CHECK_WINDOW
            )
            banned = await history_mgr.get_interesting_cities_banned()
            extra["banned_cities"] = banned
            log.info(f"🏙️ interesting_cities banned places in Redis: {len(banned)}")
            ic_data = await generate_interesting_cities_content(
                history_ic, extra, recent_ic_sig, max_attempts=3
            )
            photo_query = str(ic_data.get("photo_query", "")).strip()
            if not photo_query:
                photo_query = (
                    f"{ic_data.get('city_name', '')} {ic_data.get('country', '')} city travel landscape"
                )
            log.info(f"🔍 Photo query for [interesting_cities]: '{photo_query}'")
            use_topics_ic = True
            photo_url = await fetch_photo(
                photo_query, use_topics=use_topics_ic, pick_random=True
            )
            recent_photo_urls = await history_mgr.get_recent_photo_urls(rubric, limit=PHOTO_URL_CHECK_WINDOW)
            for _ in range(PHOTO_URL_REFETCH_ATTEMPTS):
                if photo_url and photo_url in set(recent_photo_urls):
                    log.warning(f"⚠️ Repeated photo URL detected for [{rubric}] — refetching")
                    refetch_q = (
                        f"{photo_query} cityscape landmark "
                        f"{random.choice(['street', 'skyline', 'old town', 'waterfront', 'architecture'])}"
                    )
                    photo_url = await fetch_photo(
                        refetch_q, use_topics=use_topics_ic, pick_random=True
                    )
                else:
                    break
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

        elif rubric == "vocabulary_15":
            log.info("🎨 vocabulary_15: cream card (EnglishCardsGenerator style, no Unsplash photo)")
        else:
            log.info(f"🔍 Photo query for [{rubric}]: '{photo_query}'")

            # 2. Завантажуємо фото (situation: use_topics=True — кураторські колекції дають менше «сірого» урбану)
            use_topics = rubric != "photo_relax"
            # daily_phrase: один і той самий запит часто дає той самий «переможець» Unsplash — тягнемо випадковий з топу
            pick_first = rubric == "daily_phrase"
            pick_rand = pick_first or rubric == "photo_relax"
            photo_url = await fetch_photo(photo_query, use_topics=use_topics, pick_random=pick_rand)
            recent_photo_urls = await history_mgr.get_recent_photo_urls(rubric, limit=PHOTO_URL_CHECK_WINDOW)
            for _ in range(PHOTO_URL_REFETCH_ATTEMPTS):
                if photo_url and photo_url in set(recent_photo_urls):
                    log.warning(f"⚠️ Repeated photo URL detected for [{rubric}] — refetching")
                    if rubric == "situation_phrases":
                        refetch_q = f"{photo_query} alternate framing {random.choice(SITUATION_WARM_COLOR_MOODS)}"
                    elif rubric == "daily_phrase":
                        refetch_q = f"{photo_query} {random.choice(DAILY_PHOTO_REFETCH_TAGS)}"
                    elif rubric == "quote_motivation":
                        refetch_q = f"{photo_query} different mood variation natural"
                    elif rubric == "photo_relax":
                        refetch_q = f"{photo_query} scenic landscape nature {random.choice(['peaceful', 'serene', 'wild', 'pristine'])}"
                    else:
                        refetch_q = photo_query
                    photo_url = await fetch_photo(refetch_q, use_topics=use_topics, pick_random=True)
                else:
                    break
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

        # 3. Генеруємо контент (interesting_cities уже згенеровано перед фото)
        if ic_data is not None:
            data = ic_data
            log.info(f"🤖 Content pre-generated for [interesting_cities]")
        else:
            history = await history_mgr.get_used(rubric)
            log.info(f"🤖 Generating content for [{rubric}]...")
            if rubric == "quote_motivation":
                data = await generate_quote_motivation_content(history, extra, max_attempts=3)
            elif rubric == "vocabulary_15":
                recent_vocab_sig = await history_mgr.get_recent_signatures(
                    "vocabulary_15", limit=QUIZ_SIGNATURE_CHECK_WINDOW
                )
                data = await generate_vocabulary_15_content(
                    history, extra, recent_vocab_sig, max_attempts=3
                )
            elif rubric == "photo_relax":
                recent_pr = await history_mgr.get_recent_signatures(
                    "photo_relax", limit=QUIZ_SIGNATURE_CHECK_WINDOW
                )
                data = await generate_photo_relax_content(
                    history, extra, recent_pr, max_attempts=3
                )
            else:
                data = await generate_content(rubric, history, extra)
        log.info(f"✅ Content: {json.dumps(data, ensure_ascii=False)[:200]}")

        # 4. Будуємо HTML (фон з фото або кремова картка vocabulary_15)
        if rubric == "daily_phrase":
            html = build_daily_phrase(data, photo_b64)
        elif rubric == "situation_phrases":
            cat = category or SITUATION_CATEGORIES[0]
            html = build_situation_phrases(data, photo_b64, cat)
        elif rubric == "quote_motivation":
            html = build_quote_motivation(data, photo_b64)
        elif rubric == "vocabulary_15":
            html = build_vocabulary_15(data)
        elif rubric == "photo_relax":
            html = build_photo_relax(data, photo_b64)
        elif rubric == "interesting_cities":
            html = build_interesting_cities(data, photo_b64)
        else:
            log.error(f"❌ Unknown image rubric: {rubric}")
            return

        # 5. Рендеримо PNG
        log.info(f"🎨 Rendering PNG for [{rubric}]...")
        png_bytes = await render_card(html)

        # 6. Публікуємо (caption під медіа — photo_relax та interesting_cities)
        log.info(f"📤 Sending to Telegram [{rubric}]...")
        caption_raw = None
        if rubric == "photo_relax":
            caption_raw = build_photo_relax_caption(data)
        elif rubric == "interesting_cities":
            caption_raw = build_interesting_cities_caption(data)
        caption_out, clipped = (
            clip_telegram_caption(caption_raw) if caption_raw else ("", False)
        )
        if clipped:
            log.warning(
                f"⚠️ Telegram caption clipped to {TELEGRAM_CAPTION_MAX} chars for [{rubric}]"
            )
        if caption_out:
            log.info(f"📝 Telegram caption length={len(caption_out)} for [{rubric}]")
        success = await send_photo_to_telegram(
            png_bytes, rubric, caption=caption_out or None
        )

        # 7. Зберігаємо в історію
        if success:
            history_key = json.dumps(data, ensure_ascii=False)[:100]
            await history_mgr.add_used(rubric, history_key)
            if photo_url:
                await history_mgr.add_photo_url(rubric, photo_url)
            if rubric == "quote_motivation":
                theme_name = extra.get("quote_theme_name", "unknown")
                quote_preview = str(data.get("quote_en", "")).strip()[:120]
                log.info(f"📣 [NF] quote_motivation published | theme='{theme_name}' | quote='{quote_preview}'")
                await history_mgr.advance_quote_theme_index()
            if rubric == "vocabulary_15":
                tt = str(data.get("theme_title", "")).strip()
                if tt:
                    try:
                        await history_mgr.r.lpush("used:vocabulary_15_themes", tt)
                        await history_mgr.r.ltrim("used:vocabulary_15_themes", 0, 199)
                    except Exception as e:
                        log.error(f"❌ vocabulary_15 theme history: {e}")
                ws = data.get("words", [])
                sig = build_vocabulary_15_word_signature(ws)
                if sig:
                    await history_mgr.add_signature("vocabulary_15", sig)
            if rubric == "photo_relax":
                await history_mgr.advance_photo_relax_theme_index()
                await history_mgr.advance_photo_relax_voice_style_index()
                ss = data.get("sentences") or []
                if isinstance(ss, list) and len(ss) == 4:
                    await history_mgr.add_signature(
                        "photo_relax", build_photo_relax_signature([str(x) for x in ss])
                    )
            if rubric == "interesting_cities":
                pk = _interesting_cities_place_key(
                    str(data.get("city_name", "")),
                    str(data.get("country", "")),
                )
                if pk:
                    await history_mgr.add_interesting_city_place(pk)
                await history_mgr.add_signature(
                    "interesting_cities", build_interesting_cities_signature(data)
                )

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
        recent_signatures = await history_mgr.get_recent_signatures(rubric, limit=QUIZ_SIGNATURE_CHECK_WINDOW)
        extra = {}
        cefr_band = get_cefr_band_for_today()
        extra["cefr_band"] = cefr_band
        if rubric == "prepositions_quiz":
            subtype_idx = await history_mgr.get_prepositions_subtype_index()
            format_idx = await history_mgr.get_prepositions_format_index()
            subtype = PREPOSITIONS_SUBTYPES[subtype_idx]
            exercise_format = PREPOSITIONS_EXERCISE_FORMATS[format_idx]
            extra = {
                "prepositions_subtype": subtype["name"],
                "prepositions_subtype_desc": subtype["description"],
                "prepositions_subtype_examples": subtype["examples"],
                "exercise_format": exercise_format,
                "cefr_band": cefr_band,
            }
            log.info(f"📚 Prepositions selected subtype: {subtype['name']}")
            log.info(f"🧩 Prepositions selected format: {exercise_format}")
        elif rubric == "grammar_quiz":
            subtype_idx = await history_mgr.get_grammar_subtype_index()
            format_idx = await history_mgr.get_grammar_format_index()
            sentence_type_idx = await history_mgr.get_grammar_sentence_type_index()
            subtype = GRAMMAR_SUBTYPES[subtype_idx]
            exercise_format = GRAMMAR_EXERCISE_FORMATS[format_idx]
            sentence_type = GRAMMAR_SENTENCE_TYPES[sentence_type_idx]
            extra = {
                "grammar_subtype": subtype["name"],
                "grammar_subtype_desc": subtype["description"],
                "grammar_subtype_focus": subtype["focus"],
                "grammar_format": exercise_format,
                "grammar_sentence_type": sentence_type,
                "cefr_band": cefr_band,
            }
            log.info(f"🧱 Grammar selected subtype: {subtype['name']}")
            log.info(f"🧱 Grammar selected format: {exercise_format}")
            if subtype["name"] == "Verb Tenses":
                log.info(f"🧱 Grammar selected sentence type: {sentence_type}")
        elif rubric == "vocabulary_quiz":
            topic_idx = await history_mgr.get_vocabulary_topic_index()
            topic = SITUATION_CATEGORIES[topic_idx]
            extra = {
                "vocab_topic_name": topic["name"],
                "vocab_topic_desc": topic["description"],
                "cefr_band": cefr_band,
            }
            log.info(f"📘 Vocabulary selected theme: {topic['name']}")
        elif rubric == "confusing_words_quiz":
            subtype_idx = await history_mgr.get_confusing_words_subtype_index()
            format_idx = await history_mgr.get_confusing_words_format_index()
            subtype = CONFUSING_WORDS_SUBTYPES[subtype_idx]
            exercise_format = CONFUSING_WORDS_FORMATS[format_idx]
            extra = {
                "confusing_subtype": subtype["name"],
                "confusing_subtype_desc": subtype["description"],
                "confusing_subtype_examples": subtype["examples"],
                "confusing_format": exercise_format,
                "cefr_band": cefr_band,
            }
            log.info(f"🧠 ConfusingWords selected subtype: {subtype['name']}")
            log.info(f"🧠 ConfusingWords selected format: {exercise_format}")
        log.info(f"📣 [NF] {rubric} generation config | cefr='{cefr_band}' | extra={json.dumps(extra, ensure_ascii=False)[:220]}")

        log.info(f"🤖 Generating quiz for [{rubric}]...")
        data, attempts_used, validation_reason = await generate_quiz_content_with_validation(
            rubric=rubric,
            history=history,
            extra=extra,
            recent_signatures=recent_signatures,
            max_attempts=QUIZ_VALIDATION_MAX_ATTEMPTS,
        )
        log.info(f"✅ Quiz data: {json.dumps(data, ensure_ascii=False)[:200]}")
        log.info(f"📣 [NF] {rubric} validation | attempts={attempts_used} | status='{validation_reason}'")

        # 2. Публікуємо
        success = await send_quiz_to_telegram(data, rubric)

        # 3. Зберігаємо в історію
        if success:
            history_key = data.get("question", "")[:100]
            await history_mgr.add_used(rubric, history_key)
            signature = build_quiz_signature(rubric, data, extra)
            await history_mgr.add_signature(rubric, signature, max_items=QUIZ_SIGNATURE_HISTORY_LIMIT)
            log.info(f"📣 [NF] {rubric} published | signature='{signature[:140]}'")
            if rubric == "prepositions_quiz":
                await history_mgr.advance_prepositions_subtype_index()
                await history_mgr.advance_prepositions_format_index()
            if rubric == "grammar_quiz":
                await history_mgr.advance_grammar_subtype_index()
                await history_mgr.advance_grammar_format_index()
                if extra.get("grammar_subtype") == "Verb Tenses":
                    await history_mgr.advance_grammar_sentence_type_index()
            if rubric == "vocabulary_quiz":
                await history_mgr.advance_vocabulary_topic_index()
            if rubric == "confusing_words_quiz":
                await history_mgr.advance_confusing_words_subtype_index()
                await history_mgr.advance_confusing_words_format_index()

        elapsed = time.time() - start_time
        log.info(f"⏱️ [{rubric}] completed in {elapsed:.1f}s | success={success}")

    except Exception as e:
        log.error(f"❌ CRITICAL ERROR in quiz [{rubric}]: {e}", exc_info=True)
    finally:
        await history_mgr.release_lock(rubric)


async def publish_placeholder_rubric(rubric: str, redis_client: UpstashRedis) -> None:
    _ = redis_client
    log.warning(f"⏭️ [{rubric}] — rubric scheduled but not implemented yet; skipping publish")


async def publish_card(rubric: str, redis_client: UpstashRedis):
    """Ручний тест: GET /test/<rubric> (HealthHandler); для відео — GET /test/travel_video."""
    if rubric in PLACEHOLDER_RUBRICS:
        await publish_placeholder_rubric(rubric, redis_client)
    elif rubric in QUIZ_RUBRICS:
        await publish_quiz(rubric, redis_client)
    elif rubric == "travel_video":
        await publish_travel_video(rubric, redis_client)
    else:
        await publish_image_card(rubric, redis_client)


# ──────────────────────────────────────────────
# HTTP СЕРВЕР — health check + /test/{rubric}
# ──────────────────────────────────────────────
_redis_client_global: "UpstashRedis | None" = None

# Усі рубрики, для яких дозволено ручний запуск HTTP (у т.ч. travel_video з WEEKEND_16_00).
VALID_RUBRICS = sorted(
    set(WEEKDAY_10_00.values())
    | {r for _, r in QUIZ_SLOTS}
    | set(WEEKEND_16_00.values())
)

RUBRIC_HELP = "\n".join(f"  /test/{r}" for r in VALID_RUBRICS)

# Документація для GET / — усі заплановані рубрики (тест: GET /test/{rubric})
TEST_ENDPOINTS_DOC = f"""English A2 Bot v2.0 — OK

Manual test: GET /test/{{rubric}} — одноразово запускає публікацію (перевірте Telegram).
  Приклад відео: GET /test/travel_video — той самий пайплайн, що й у неділю 16:00 (sendVideo без підпису).

Пн–Пт 10:00 Europe/Kyiv — великі пости (PNG):
  quote_motivation     — цитата / мотивація + фото
  vocabulary_15        — 15 слів + IPA (кремова картка)
  daily_phrase         — фраза дня + приклад (en) + переклад (ua)
  situation_phrases    — 5 фраз (en/ua) для життєвої ситуації
  photo_relax          — природа: чисте фото PNG; 4 речення (en, «живий» п’ятничний релакс, ротація стилю голосу)

Пн–Пт — квізи (Telegram poll):
  grammar_quiz           (11:30)
  vocabulary_quiz        (13:00)
  confusing_words_quiz   (14:30)
  prepositions_quiz      (16:00)

Сб 16:00 Europe/Kyiv — чисте фото PNG + текст у caption:
  interesting_cities   — фото + 3–5 речень англ. (живий тон; без окремого гачка й без фінального питання)

Нд 16:00 Europe/Kyiv — відео 9:16 (сток Pexels/Pixabay → Gemini A2 текст → ElevenLabs / Google TTS → FFmpeg + бренд 2–3 с), без підпису:
  travel_video

Повний список шляхів для копіювання:
{RUBRIC_HELP}
"""


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.rstrip("/")

        # GET / — health check
        if path == "" or path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(TEST_ENDPOINTS_DOC.encode("utf-8"))
            return

        # GET /test/{rubric}
        if path.startswith("/test/"):
            rubric = path[6:]  # все після /test/
            if rubric not in VALID_RUBRICS:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                msg = f"Unknown rubric: '{rubric}'\n\nSee GET / for the full list.\n\n{RUBRIC_HELP}\n"
                self.wfile.write(msg.encode("utf-8"))
                log.warning(f"⚠️ /test/ called with unknown rubric: '{rubric}'")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
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
    log.info("🧪 GET / — список рубрик; GET /test/{rubric} — тестовий пост")
    server.serve_forever()


# ──────────────────────────────────────────────
# ПЛАНУВАЛЬНИК
# ──────────────────────────────────────────────
async def scheduler(redis_client: UpstashRedis):
    log.info("⏰ Scheduler started (Europe/Kyiv)")
    published_slots: set[str] = set()

    while True:
        now = datetime.now(TZ)
        slot_key = f"{now.date().isoformat()}|{now.hour:02d}:{now.minute:02d}"

        if now.hour == 0 and now.minute == 0:
            published_slots.clear()
            log.info("🔄 Reset published_slots for new calendar day (Kyiv)")

        rubric = get_rubric_for_datetime(now)
        if rubric and slot_key not in published_slots:
            published_slots.add(slot_key)
            log.info(f"⏰ Triggering [{rubric}] at {now.strftime('%Y-%m-%d %H:%M')} Kyiv")
            asyncio.create_task(publish_card(rubric, redis_client))

        await asyncio.sleep(30)


# ──────────────────────────────────────────────
# ТОЧКА ВХОДУ
# ──────────────────────────────────────────────
async def main():
    global _redis_client_global, _loop_global

    _get_required_env("TELEGRAM_BOT_TOKEN")
    _get_required_env("TELEGRAM_CHAT_ID")

    log.info("🤖 English A2 Bot v2.0 starting...")
    log.info(
        f"📋 Schedule Kyiv: Mon–Fri 10:00 {WEEKDAY_10_00}, "
        f"quizzes {QUIZ_SLOTS}, weekend {WEEKEND_16_00}"
    )
    log.info(f"🖼️ Image rubrics: {IMAGE_RUBRICS}")
    log.info("🎬 travel_video: окремий відео-пайплайн (неділя 16:00)")
    log.info(f"📝 Quiz rubrics: {QUIZ_RUBRICS}")
    log.info("🧪 HTTP: GET / — опис усіх тестових рубрик; GET /test/<rubric> — ручний запуск")

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
