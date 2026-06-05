import os
import json
import time
import asyncio
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from google import generativeai as genai
from pydub import AudioSegment
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import razorpay

# ========================= CONFIG =========================
app = FastAPI(title="Digital Isai Backend")

# Environment / Config
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "YOUR_WHATSAPP_BUSINESS_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "YOUR_PHONE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory stores
lead_database: Dict[str, Dict] = {}
chat_contexts: Dict[str, list] = {}
freelancer_pool = []

BASE_DIR = Path("./DigitalIsai_data")
BASE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

# ====================== HELPERS ======================
async def send_whatsapp_message(to: str, text: str, media_url: Optional[str] = None):
    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text" if not media_url else "document" if media_url.endswith('.pdf') else "audio",
    }
    if media_url:
        if media_url.endswith(('.mp3', '.wav')):
            payload["type"] = "audio"
            payload["audio"] = {"link": media_url}
        else:
            payload["type"] = "document"
            payload["document"] = {"link": media_url, "filename": os.path.basename(media_url)}
    else:
        payload["text"] = {"body": text}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            logger.info(f"WA Sent: {resp.status_code} - {to}")
            return resp.json()
        except Exception as e:
            logger.error(f"WA Send failed: {e}")

def transcribe_voice_note(audio_url: str) -> str:
    return "[Voice note transcribed using Gemini]"

# ====================== TIMEOUT ======================
async def check_timeouts():
    now = datetime.now()
    to_remove = []
    for lead_id, data in list(lead_database.items()):
        if data.get('status') == 'In_Progress':
            elapsed = (now - data['start_time']).total_seconds()
            grace = 300 if data.get('upload_stream_active', False) else 0
            if elapsed > (30 * 60 + grace):
                await handle_timeout(lead_id)
                to_remove.append(lead_id)
    for lid in to_remove:
        lead_database.pop(lid, None)

async def handle_timeout(lead_id: str):
    data = lead_database.get(lead_id)
    if not data:
        return
    chat_log = chat_contexts.get(lead_id, [])
    lyrics = data.get('latest_lyrics', 'No lyrics yet')
    payload = {"lead_id": lead_id, "chat_log": chat_log[-20:], "lyrics": lyrics}
    freelancer_pool.append(payload)
    await send_whatsapp_message(data['client_phone'], "⏰ Our team is experiencing high demand. A fresh creator is picking this up shortly!")
    logger.info(f"Timeout handled: {lead_id}")

# ====================== SALES ENGINE ======================
SYSTEM_PROMPT = """Act as Ananya, a friendly, ultra-polite, native Tanglish (Tamil + English) human sales representative for Digital Isai.
You sell custom Tamil/Tanglish songs. Be warm, persuasive. Refine lyrics."""

async def process_client_message(lead_id: str, message: str, is_voice: bool = False, client_phone: str = ""):
    if lead_id not in lead_database:
        lead_database[lead_id] = {'start_time': datetime.now(), 'status': 'In_Progress', 'client_phone': client_phone}
        chat_contexts[lead_id] = []

    if is_voice:
        message = transcribe_voice_note(message)

    chat_contexts[lead_id].append({"role": "user", "content": message})

    confirm_keywords = re.compile(r'\b(ok|super|confirm|perfect|done|yes|finalize|proceed)\b', re.I)
    if confirm_keywords.search(message):
        await broadcast_to_freelancers(lead_id)
        return

    context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_contexts[lead_id][-10:]])
    full_prompt = f"{SYSTEM_PROMPT}\n\nChat history:\n{context}\n\nRespond naturally:"

    try:
        response = model.generate_content(full_prompt)
        reply_text = response.text
    except Exception as e:
        reply_text = "Sorry da, konjam issue. Try again!"

    chat_contexts[lead_id].append({"role": "assistant", "content": reply_text})
    await send_whatsapp_message(client_phone, reply_text)

async def broadcast_to_freelancers(lead_id: str):
    data = lead_database[lead_id]
    payload = {"lead_id": lead_id, "client_phone": data['client_phone'], "chat_log": chat_contexts.get(lead_id, []), "package": "Premium Custom Song"}
    freelancer_pool.append(payload)
    await send_whatsapp_message(data['client_phone'], "🎵 Great! Our creators are now composing your song. You'll get a 45-sec preview soon.")
    logger.info(f"Project broadcasted: {lead_id}")

# ====================== PRODUCTION ======================
@app.post("/freelancer/upload")
async def freelancer_upload(request: Request):
    form = await request.form()
    lead_id = form.get("lead_id")
    file = form.get("file")
    if not lead_id or not file:
        raise HTTPException(400, "Missing data")

    file_path = BASE_DIR / f"full_{lead_id}.mp3"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    audio = AudioSegment.from_mp3(file_path)
    preview = audio[:45000]
    preview_path = BASE_DIR / f"preview_{lead_id}.mp3"
    preview.export(preview_path, format="mp3", bitrate="128k")

    lead_database[lead_id]['full_track'] = str(file_path)
    lead_database[lead_id]['preview'] = str(preview_path)

    checkout_link = f"https://razorpay.link/{lead_id}"
    await send_whatsapp_message(lead_database[lead_id]['client_phone'], "🎧 Here's your 45-second preview!", media_url=str(preview_path))
    await send_whatsapp_message(lead_database[lead_id]['client_phone'], f"Pay securely: {checkout_link}")
    return {"status": "preview_sent"}

# ====================== RAZORPAY ======================
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request, background: BackgroundTasks):
    payload = await request.json()
    if payload.get('event') == 'payment.captured':
        payment = payload['payload']['payment']['entity']
        lead_id = payment.get('notes', {}).get('lead_id')
        if lead_id and lead_id in lead_database:
            amount = payment['amount'] / 100
            lead_database[lead_id]['ledger'] = {
                'admin': amount * 0.30,
                'ads': amount * 0.20,
                'freelancer': amount * 0.50,
                'status': 'Locked_Escrow'
            }
            background.add_task(process_successful_payment, lead_id, payment)
    return {"status": "ok"}

async def process_successful_payment(lead_id: str, payment: dict):
    data = lead_database[lead_id]
    invoice_path = await generate_invoice(lead_id, payment)
    await send_whatsapp_message(data['client_phone'], "✅ Payment successful! Here's your full song.", media_url=str(data['full_track']))
    await send_whatsapp_message(data['client_phone'], "📄 Invoice attached.", media_url=invoice_path)
    data['status'] = 'Completed'

async def generate_invoice(lead_id: str, payment: dict) -> str:
    pdf_path = BASE_DIR / f"invoice_{lead_id}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 700, "Digital Isai - Invoice")
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, f"Invoice ID: INV-{lead_id}")
    c.drawString(100, 630, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(100, 610, f"Amount: ₹{payment.get('amount', 0)/100}")
    c.save()
    return str(pdf_path)

# ====================== WEEKLY ======================
async def weekly_settlement():
    logger.info("Weekly settlement running...")

# ====================== WEBHOOK ======================
@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    try:
        messages = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [])
        for msg in messages:
            lead_id = msg['from']
            client_phone = msg['from']
            if msg['type'] == 'text':
                await process_client_message(lead_id, msg['text']['body'], client_phone=client_phone)
            elif msg['type'] == 'audio':
                audio_url = msg['audio']['id']
                await process_client_message(lead_id, audio_url, is_voice=True, client_phone=client_phone)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    return {"status": "processed"}

@app.get("/")
async def health():
    return {"status": "Digital Isai Backend Running"}

@app.on_event("startup")
async def startup_event():
    scheduler.add_job(check_timeouts, 'interval', seconds=30)
    scheduler.add_job(weekly_settlement, 'cron', day_of_week='fri', hour=23, minute=59)
    scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
