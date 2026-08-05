import streamlit as st
import requests
import json
from groq import Groq
import re

# --- Load keys from Streamlit secrets ---
SAHARA_API_KEY = st.secrets["SAHARA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)

NUMBER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million",
    "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi",
    "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini", "themanini", "tisini",
    "mia", "elfu", "milioni"
]

SCALE_WORDS = ["elfu", "mia", "thousand", "hundred", "milioni", "million"]

VALID_SPEAKER_ACTIONS = {"deposit", "payout_received", "late_payment_note", "membership_update", "other"}


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
    if not transcript or len(transcript.strip()) < 5:
        return json.dumps({
            "reasoning": "Transcript empty or too short to contain meaningful speech.",
            "speaker_action": "other",
            "amount": None,
            "referenced_member": "none",
            "referenced_member_context": "none",
            "action_type": "other"
        })

    prompt = f"""You are extracting structured data from a chama (savings group) voice message.
The message may mix English and Swahili. The speaker is usually reporting their own action,
and may separately reference another member.

CRITICAL RULES:
- Only extract information explicitly stated. Do NOT infer or guess missing values.
- If no amount is mentioned, amount must be null — never 0, never invented, never a small placeholder number.
- Amounts may be stated as digits (e.g. "2000") or as words (e.g. "elfu mbili" / "two thousand"). Both are valid.
- If the speaker uses future tense (e.g. "nitatuma" / "I will send") without saying they already paid, this is NOT a completed deposit.
- Vague quantity words (e.g. "kidogo" / "a little" / "some") are NOT numbers. Never convert them into a numeric guess.
- referenced_member must be an actual person's name, never a number or group description. If none, use "none".

Example 1:
Transcript: "Nimechelewa mwezi huu lakini nitatuma kiasi chote Ijumaa ijayo."
Correct extraction: {{"reasoning": "Speaker is reporting their own status. Future tense 'nitatuma' means not yet paid. No amount stated in digits or words.", "speaker_action": "late_payment_note", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "late_payment_note"}}

Example 2:
Transcript: "Nimetuma pesa kidogo leo."
Correct extraction: {{"reasoning": "Speaker completed an action ('nimetuma'). 'Kidogo' is a vague qualifier, not a digit or number word, so no real amount was stated.", "speaker_action": "deposit", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "deposit"}}

Now extract from this transcript:
Transcript: "{transcript}"

First, in "reasoning", briefly state: (a) who is speaking, (b) what tense/timing they use,
(c) whether they personally took a financial action or only mentioned someone else's,
(d) whether an amount was stated as digits, as words, or not stated at all.
Then extract the five fields, matching your reasoning.

Respond with ONLY valid JSON in this exact structure, no other text:
{{"reasoning": "...", "speaker_action": "...", "amount": ..., "referenced_member": "...", "referenced_member_context": "...", "action_type": "..."}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def validate_extraction(transcript, extracted):
    if extracted.get("amount") is not None:
        transcript_lower = transcript.lower()
        has_digit = bool(re.search(r'\d', transcript))
        has_number_word = any(
            re.search(rf'\b{re.escape(word)}\b', transcript_lower)
            for word in NUMBER_WORDS
        )
        if not has_digit and not has_number_word:
            extracted["_original_amount_before_validation"] = extracted["amount"]
            extracted["amount"] = None
            extracted["_flag"] = "amount removed: no digit or number word found in transcript"
        elif extracted["amount"] is not None and extracted["amount"] < 100:
            has_scale_word = any(
                re.search(rf'\b{re.escape(word)}\b', transcript_lower)
                for word in SCALE_WORDS
            )
            if not has_scale_word:
                extracted["_flag"] = f"amount {extracted['amount']} may be missing a scale word (elfu/mia/thousand) — possible transcription drop"

    if extracted.get("speaker_action") not in VALID_SPEAKER_ACTIONS:
        extracted["_flag_invalid_speaker_action"] = extracted.get("speaker_action")
        extracted["speaker_action"] = "other"

    extracted.pop("reasoning", None)
    return extracted


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
st.set_page_config(page_title="HabaHub", page_icon="🎙️", layout="wide")
st.title("🎙️ HabaHub")
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
        extracted_raw = extract_chama_action(transcript)
        extracted = json.loads(extracted_raw) if isinstance(extracted_raw, str) else extracted_raw
        st.json(extracted)
        extracted = validate_extraction(transcript, extracted)

    if "_flag" in extracted:
        st.warning(f"⚠️ {extracted['_flag']}")
    if "_flag_invalid_speaker_action" in extracted:
        st.warning(f"⚠️ Unrecognized action type detected: '{extracted['_flag_invalid_speaker_action']}' — defaulted to 'other'.")

    st.subheader("Review before submitting")
    edited_speaker_action = st.selectbox(
        "Action", list(VALID_SPEAKER_ACTIONS),
        index=list(VALID_SPEAKER_ACTIONS).index(extracted["speaker_action"])
    )
    edited_amount = st.number_input(
        "Amount", value=float(extracted["amount"]) if extracted["amount"] is not None else 0.0,
        min_value=0.0, step=1.0
    )
    edited_referenced_member = st.text_input("Referenced member", value=extracted["referenced_member"])

    if st.button("Confirm and submit"):
        was_amended = (
            edited_speaker_action != extracted["speaker_action"]
            or (extracted["amount"] is not None and edited_amount != extracted["amount"])
            or (extracted["amount"] is None and edited_amount != 0.0)
            or edited_referenced_member != extracted["referenced_member"]
        )

        final_entry = dict(extracted)
        final_entry["speaker_action"] = edited_speaker_action
        final_entry["amount"] = edited_amount if edited_amount > 0 else None
        final_entry["referenced_member"] = edited_referenced_member
        final_entry["amended_by_human"] = was_amended
        final_entry["ai_original_amount"] = extracted["amount"]
        final_entry["ai_original_speaker_action"] = extracted["speaker_action"]

        st.session_state.ledger.append(final_entry)
        confirmation_text = generate_confirmation_text(final_entry)
        with st.spinner("Generating spoken confirmation..."):
            audio_url = generate_tts(confirmation_text)
        st.write("**Confirmation:**", confirmation_text)
        st.audio(audio_url)

st.divider()
st.subheader("Ledger")
st.table(st.session_state.ledger)