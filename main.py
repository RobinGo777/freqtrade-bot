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
from io import BytesIO
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
PHOTO_URL_CHECK_WINDOW = 25
PHOTO_URL_REFETCH_ATTEMPTS = 2


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


def get_photo_query_for_situation(category: dict) -> str:
    holiday = get_today_holiday()
    if holiday:
        log.info(f"🎉 Holiday situation photo: {holiday['name']}")
        return holiday["photo_query"] + " celebration"
    today = date.today()
    season = get_season(today.month)
    if today.day <= 10:
        stage = "early"
    elif today.day <= 20:
        stage = "mid"
    else:
        stage = "late"

    base = category.get("photo_query", "everyday life realistic scene")
    season_style = random.choice(SITUATION_SEASON_STYLE.get(season, ["cinematic soft light"]))
    stage_style = random.choice(SITUATION_MONTH_STAGE_STYLE.get(stage, ["balanced composition"]))
    # Для 5 блоків потрібен "спокійний" фон із вільним простором під текст.
    query = (
        f"{base} {season_style} {stage_style} "
        "cinematic soft light minimal background soft bokeh copy space no text"
    )
    log.info(
        f"🎨 Situation photo query: '{query}' | season={season} month={today.month} day={today.day} stage={stage}"
    )
    return query


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
  "example_ua": "Ukrainian translation of the example sentence (max 140 characters)",
  "photo_query": "3-5 keywords for stock photo: scene + mood + style (cinematic soft light)"
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
  "quote_ua": "Ukrainian translation (natural, not word-for-word)",
  "photo_query": "3-6 keywords for green nature photo: forest, moss, valley, mountain lake, fog, rain atmosphere; only nature, no people, no city; style: moody cinematic"
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
- photo_query: prioritize green nature mood (forest, moss, valley, mountain lake, fog, rain, deep greenery)
- photo_query: ONLY nature landscapes, NO people, NO urban/city elements
- Good photo_query examples: "misty green forest cinematic", "mossy forest path fog moody", "emerald mountain lake overcast"
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


def get_cefr_band_for_today() -> str:
    # Пн-Вт A2, Ср-Чт A2+, Пт-Сб B1, Нд mixed review
    weekday = datetime.now().weekday()  # Mon=0 ... Sun=6
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
    photo_query = str(data.get("photo_query", "")).strip()

    if not quote_en or not quote_ua or not photo_query:
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

    if len(photo_query.split()) < 3:
        return False, "photo_query too short"
    if len(photo_query) > 140:
        return False, "photo_query too long"

    recent_norm = [_normalize_quote_text(item) for item in used_history[-25:]]
    if quote_norm and any(quote_norm in h or h in quote_norm for h in recent_norm if h):
        return False, "too similar to recent history"

    return True, "ok"


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
            theme_idx = await history_mgr.get_quote_theme_index()
            theme = QUOTE_THEMES[theme_idx]
            extra = {
                "quote_theme_name": theme["name"],
                "quote_theme_description": theme["description"],
            }
            log.info(f"🧭 quote_motivation selected theme: {theme['name']}")
            log.info(f"📣 [NF] quote_motivation theme selected | index={theme_idx} | name='{theme['name']}'")
            photo_query = await get_quote_photo_query(history_mgr)

        log.info(f"🔍 Photo query for [{rubric}]: '{photo_query}'")

        # 2. Завантажуємо фото
        use_topics = rubric != "situation_phrases"
        photo_url = await fetch_photo(photo_query, use_topics=use_topics)
        recent_photo_urls = await history_mgr.get_recent_photo_urls(rubric, limit=PHOTO_URL_CHECK_WINDOW)
        for _ in range(PHOTO_URL_REFETCH_ATTEMPTS):
            if photo_url and photo_url in set(recent_photo_urls):
                log.warning(f"⚠️ Repeated photo URL detected for [{rubric}] — refetching")
                photo_url = await fetch_photo(photo_query, use_topics=use_topics)
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

        # 3. Генеруємо контент
        history = await history_mgr.get_used(rubric)
        log.info(f"🤖 Generating content for [{rubric}]...")
        if rubric == "quote_motivation":
            data = await generate_quote_motivation_content(history, extra, max_attempts=3)
        else:
            data = await generate_content(rubric, history, extra)
        log.info(f"✅ Content: {json.dumps(data, ensure_ascii=False)[:200]}")

        # Якщо Gemini повернув кращий photo_query — оновлюємо фото
        ai_photo_query = data.get("photo_query", "").strip()
        # Для daily_phrase/situation_phrases тримаємо контрольований стиль, тому AI photo_query не застосовуємо.
        if rubric in {"daily_phrase", "situation_phrases", "quote_motivation"} and ai_photo_query and ai_photo_query != photo_query:
            log.info(f"🎨 {rubric}: skip AI photo query to keep controlled style | ai='{ai_photo_query}'")
            ai_photo_query = ""

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
            if photo_url:
                await history_mgr.add_photo_url(rubric, photo_url)
            if rubric == "quote_motivation":
                theme_name = extra.get("quote_theme_name", "unknown")
                quote_preview = str(data.get("quote_en", "")).strip()[:120]
                log.info(f"📣 [NF] quote_motivation published | theme='{theme_name}' | quote='{quote_preview}'")
                await history_mgr.advance_quote_theme_index()

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

    _get_required_env("TELEGRAM_BOT_TOKEN")
    _get_required_env("TELEGRAM_CHAT_ID")

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
