import streamlit as st
import requests
import json
from groq import Groq

# --- Load keys from Streamlit secrets (set these in Streamlit Cloud's secrets manager, not hardcoded) ---
SAHARA_API_KEY = st.secrets["SAHARA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)

def transcribe_sahara(audio_path, language="sw"):
    url = "https://infer.voice.intron.io/file/v1/upload/sync"
    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}"}
    files = {"audio_file_blob": open(audio_path, "rb")}
    data = {"audio_file_name": audio_path, "use_language_asr_input": language}
    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"Sahara API error {response.status_code}: {response.text[:300]}")

    try:
        return response.json()["data"]["audio_transcript"]
    except requests.exceptions.JSONDecodeError:
        raise Exception(f"Sahara returned non-JSON response: {response.text[:300]}")

def extract_chama_action(transcript):
    prompt = f"""You are extracting structured data from a chama (savings group) voice message.
The message may mix English and Swahili. The speaker is usually reporting their own action,
and may separately reference another member.

CRITICAL RULES:
- Only extract information explicitly stated. Do NOT infer or guess missing values.
- If no amount is mentioned, amount must be null — never 0, never invented.
- If the speaker uses future tense (e.g. "nitatuma" / "I will send") without saying they already paid, this is NOT a completed deposit.
- referenced_member must be an actual person's name, never a number or group description. If none, use "none".

Example:
Transcript: "Nimechelewa mwezi huu lakini nitatuma kiasi chote Ijumaa ijayo."
Correct extraction: {{"speaker_action": "late_payment_note", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "late_payment_note"}}

Now extract from this transcript:
Transcript: "{transcript}"

Extract exactly these five fields as JSON:
- speaker_action: one of "deposit", "payout_received", "late_payment_note", "membership_update", "other"
- amount: numeric amount, or null if none stated
- referenced_member: another person's name, or "none"
- referenced_member_context: brief reason, or "none"
- action_type: one of "deposit", "payout", "late_payment_note", "membership_update", "other"

Respond with ONLY valid JSON, no other text."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_tts(text, voice_accent="swahili", voice_gender="female", voice_language="en"):
    url = "https://infer.voice.intron.io/tts/v1/generate"
    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "voice_accent": voice_accent, "voice_gender": voice_gender, "voice_language": voice_language}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["data"]["audio_path"]

def generate_confirmation_text(entry):
    if entry["speaker_action"] == "deposit":
        text = f"Confirmed: deposit of {entry['amount']} logged."
        if entry["referenced_member"] != "none":
            text += f" Noted: {entry['referenced_member']} — {entry['referenced_member_context']}."
    elif entry["speaker_action"] == "late_payment_note":
        text = "Noted: you'll be sending your payment late. No amount logged yet."
    elif entry["speaker_action"] == "membership_update":
        text = "Group update noted."
    else:
        text = "Message logged, no financial action taken."
    return text

# --- UI ---
st.set_page_config(page_title="Chama Voice Agent", page_icon="🎙️")
st.title("🎙️ Chama Voice Agent")
st.write("Speak your contribution, payout note, or update in English, Swahili, or both.")

if "ledger" not in st.session_state:
    st.session_state.ledger = []

audio = st.audio_input("Record your message")

if audio is not None:
    with open("temp_input.wav", "wb") as f:
        f.write(audio.getvalue())

    with st.spinner("Transcribing..."):
        transcript = transcribe_sahara("temp_input.wav")
    st.write("**Transcript:**", transcript)

    with st.spinner("Extracting details..."):
        extracted = extract_chama_action(transcript)
    st.json(extracted)

    st.session_state.ledger.append(extracted)

    confirmation_text = generate_confirmation_text(extracted)
    with st.spinner("Generating spoken confirmation..."):
        audio_url = generate_tts(confirmation_text)
    st.write("**Confirmation:**", confirmation_text)
    st.audio(audio_url)

st.divider()
st.subheader("Ledger")
st.table(st.session_state.ledger)