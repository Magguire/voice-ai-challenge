import streamlit as st
import requests
import json
from groq import Groq
import re
import pandas as pd
import torch
import string
import calendar
from datetime import date, timedelta

SAHARA_API_KEY = st.secrets["SAHARA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_API_KEY = st.secrets.get("HF_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

NUMBER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million",
    "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi",
    "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini", "themanini", "tisini",
    "mia", "elfu", "milioni"
]
SCALE_WORDS = ["elfu", "mia", "thousand", "hundred", "milioni", "million"]
VALID_SPEAKER_ACTIONS = {"deposit", "payout_received", "initiate_scheduled_payout", "initiate_loan_payout", "late_payment_note", "membership_update", "loan_request", "other"}

ADMIN_ALLOWED_ACTIONS = VALID_SPEAKER_ACTIONS
MEMBER_ALLOWED_ACTIONS = {"deposit", "payout_received", "late_payment_note", "loan_request", "other"}

USERS = {
    "Wendo": {"role": "admin"},
    "Juma": {"role": "member"},
}

if "current_user_name" not in st.session_state:
    st.session_state.current_user_name = "Wendo"

CURRENT_USER = st.session_state.current_user_name
IS_ADMIN = USERS[CURRENT_USER]["role"] == "admin"

MOCK_MEMBERS = {
    "Wendo":  {"ytd_contributed": 24000, "this_month_paid": True,  "last_paid": "2026-08-01", "owed": 0,     "loan_eligible": 20000, "payout_position": 3},
    "Grace":  {"ytd_contributed": 22000, "this_month_paid": True,  "last_paid": "2026-08-02", "owed": 0,     "loan_eligible": 18000, "payout_position": 1},
    "John":   {"ytd_contributed": 18000, "this_month_paid": False, "last_paid": "2026-06-28", "owed": 4000,  "loan_eligible": 12000, "payout_position": 2},
    "Amina":  {"ytd_contributed": 24000, "this_month_paid": True,  "last_paid": "2026-08-03", "owed": 0,     "loan_eligible": 20000, "payout_position": 4},
    "Peter":  {"ytd_contributed": 16000, "this_month_paid": False, "last_paid": "2026-07-15", "owed": 2000,  "loan_eligible": 10000, "payout_position": 5},
    "Juma":   {"ytd_contributed": 12000, "this_month_paid": True,  "last_paid": "2026-08-01", "owed": 0,     "loan_eligible": 8000,  "payout_position": 6},
}

MONTHS_FULL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MOCK_MONTHLY_TOTALS = pd.DataFrame({
    "Month": MONTHS_FULL,
    "Collected": [13000, 15000, 18000, 19500, 21000, 20500, 22000, 21000, 0, 0, 0, 0]
})

MOCK_MEMBER_MONTHLY = {
    name: [round(data["ytd_contributed"] / 8)] * 8 + [0, 0, 0, 0]
    for name, data in MOCK_MEMBERS.items()
}

SCHEDULE_START_YEAR = 2026
SCHEDULE_START_MONTH = 8

REAL_TODAY = date.today()
CURRENT_MONTH_INDEX = REAL_TODAY.month


def get_member_by_position(position):
    for name, data in MOCK_MEMBERS.items():
        if data["payout_position"] == position:
            return name
    return None


def compute_date_for_offset(offset_months):
    month_index = SCHEDULE_START_MONTH - 1 + offset_months
    year = SCHEDULE_START_YEAR + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


def build_initial_schedule():
    positions_sorted = sorted(MOCK_MEMBERS.items(), key=lambda x: x[1]["payout_position"])
    schedule = {data["payout_position"]: compute_date_for_offset(i) for i, (name, data) in enumerate(positions_sorted)}
    schedule[1] = date.today()  # override: position 1's payout is due today, for demo purposes
    return schedule


def get_next_friday(from_date):
    days_ahead = 4 - from_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return from_date + timedelta(days=days_ahead)


if "payout_schedule" not in st.session_state:
    st.session_state.payout_schedule = build_initial_schedule()

if "payouts_made_positions" not in st.session_state:
    st.session_state.payouts_made_positions = set()

if "simulated_today" not in st.session_state:
    st.session_state.simulated_today = date.today()

if "scheduled_reminders" not in st.session_state:
    st.session_state.scheduled_reminders = []


def get_pending_loan_for_member(member_name):
    for lr in st.session_state.loan_requests:
        if lr["member"] == member_name and lr["status"] == "pending":
            return lr
    return None


def transcribe_sahara(audio_path, language="sw"):
    url = "https://infer.voice.intron.io/file/v1/upload/sync"
    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}"}
    files = {"audio_file_blob": open(audio_path, "rb")}
    data = {"audio_file_name": audio_path, "use_language_asr_input": language}
    response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Sahara API error {response.status_code}: {response.text[:300]}")
    try:
        return response.json()["data"]["audio_transcript"]
    except requests.exceptions.JSONDecodeError:
        raise Exception(f"Sahara returned non-JSON response: {response.text[:300]}")


from huggingface_hub import InferenceClient

def transcribe_whisper_hf(audio_path):
    client = InferenceClient(provider="fal-ai", api_key=HF_API_KEY)
    result = client.automatic_speech_recognition(audio_path, model="openai/whisper-large-v3")
    return result.text

@st.cache_resource
def load_mms_model():
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id, target_lang="swh", token=HF_API_KEY)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang="swh", ignore_mismatched_sizes=True, token=HF_API_KEY)
    return processor, model

def transcribe_mms(audio_path):
    import torchaudio
    processor, model = load_mms_model()
    speech, sr = torchaudio.load(audio_path)
    if sr != 16000:
        speech = torchaudio.functional.resample(speech, sr, 16000)
    inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids)


def classify_intent(transcript):
    prompt = f"""Classify this chama voice message as either "statement" or "question".

"statement" includes: reporting a completed or future action (deposit, payment), requesting a loan
(e.g. "I want to borrow...", "Naomba mkopo..."), requesting a scheduled payout be sent, approving a loan
payout for another member, or any message where the speaker wants an action taken or recorded —
even if phrased politely as a request.

"question" is ONLY for messages purely seeking information with no action requested
(e.g. "When is my turn?", "How much do I owe?", "Nimechangia kiasi gani?").

Transcript: "{transcript}"

Respond with ONLY one word: "statement" or "question"."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


def answer_member_query(transcript, member_name):
    member_record = MOCK_MEMBERS.get(member_name, {})
    prompt = f"""You are a chama assistant. Answer the member's question using ONLY the data provided below.
Do not invent numbers not present in the data. This is illustrative demo data.

Member data for {member_name}: {json.dumps(member_record)}

Question: "{transcript}"

Give a short, direct spoken-style answer (1-2 sentences)."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def extract_chama_action(transcript, speaker_name):
    if not transcript or len(transcript.strip()) < 5:
        return json.dumps({
            "reasoning": "Transcript empty or too short to contain meaningful speech.",
            "speaker_action": "other", "amount": None,
            "referenced_member": "none", "referenced_member_context": "none", "action_type": "other"
        })
    prompt = f"""You are extracting structured data from a chama (savings group) voice message.
The message may mix English and Swahili.

IMPORTANT: The person speaking this message is logged in as "{speaker_name}". This is a known fact,
not something to guess. If the speaker mentions another person's name (e.g. "Juma", "Grace"), that
name refers to a DIFFERENT person than {speaker_name}, never to the speaker themselves — even if the
sentence is a command or approval naming that other person. Only treat an action as about {speaker_name}
if the speaker uses first-person language ("mimi", "yangu", "I", "my") or names no one else.

CRITICAL RULES:
- Only extract information explicitly stated. Do NOT infer or guess missing values.
- If no amount is mentioned, amount must be null — never 0, never invented, never a small placeholder number.
- Amounts may be stated as digits (e.g. "2000") or as words (e.g. "elfu mbili" / "two thousand"). Both are valid.
- If the speaker uses future tense (e.g. "nitatuma" / "I will send") without saying they already paid, this is NOT a completed deposit.
- Vague quantity words (e.g. "kidogo" / "a little" / "some") are NOT numbers. Never convert them into a numeric guess.
- referenced_member must be an actual person's name, never a number or group description, and never {speaker_name} unless {speaker_name} refers to themselves. If none, use "none".
- If the speaker is COMMANDING a regular scheduled payout be sent to a member (e.g. "Tuma X kwa Y"), this is "initiate_scheduled_payout".
- If the speaker is asking to BORROW money for themselves, this is "loan_request".
- If the speaker (an admin) is APPROVING or releasing an already-requested loan payout for another member (e.g. "Approve mkopo wa Juma", "Send Juma's loan"), this is "initiate_loan_payout", and referenced_member is that OTHER member, not {speaker_name}.

Example 1:
Transcript: "Nimechelewa mwezi huu lakini nitatuma kiasi chote Ijumaa ijayo."
Correct extraction: {{"reasoning": "Speaker is reporting their own status. Future tense 'nitatuma' means not yet paid. No amount stated in digits or words.", "speaker_action": "late_payment_note", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "late_payment_note"}}

Example 2:
Transcript: "Nimetuma pesa kidogo leo."
Correct extraction: {{"reasoning": "Speaker completed an action ('nimetuma'). 'Kidogo' is a vague qualifier, not a digit or number word, so no real amount was stated.", "speaker_action": "deposit", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "deposit"}}

Example 3:
Transcript: "Tuma elfu mbili kwa Grace leo."
Correct extraction: {{"reasoning": "Speaker is issuing a command to send a scheduled payout to another member, Grace — not the speaker.", "speaker_action": "initiate_scheduled_payout", "amount": 2000, "referenced_member": "Grace", "referenced_member_context": "payout recipient", "action_type": "payout"}}

Example 4:
Transcript: "Naomba mkopo wa elfu tano."
Correct extraction: {{"reasoning": "Speaker is requesting to borrow money for themselves from the group. No other member named.", "speaker_action": "loan_request", "amount": 5000, "referenced_member": "none", "referenced_member_context": "none", "action_type": "other"}}

Example 5:
Transcript: "Approve mkopo wa Juma."
Correct extraction: {{"reasoning": "Speaker is approving and releasing an existing loan payout for another member, Juma — the speaker is the admin taking the action, Juma is who it's about, not the speaker.", "speaker_action": "initiate_loan_payout", "amount": null, "referenced_member": "Juma", "referenced_member_context": "loan payout approval", "action_type": "payout"}}

Now extract from this transcript, spoken by {speaker_name}:
Transcript: "{transcript}"

First, in "reasoning", briefly state: (a) who is speaking (should be {speaker_name}), (b) what tense/timing they use,
(c) whether they personally took a financial action or are acting on behalf of / referencing someone else,
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
        has_number_word = any(re.search(rf'\b{re.escape(w)}\b', transcript_lower) for w in NUMBER_WORDS)
        if not has_digit and not has_number_word:
            extracted["_original_amount_before_validation"] = extracted["amount"]
            extracted["amount"] = None
            extracted["_flag"] = "amount removed: no digit or number word found in transcript"
        elif extracted["amount"] is not None and extracted["amount"] < 100:
            has_scale_word = any(re.search(rf'\b{re.escape(w)}\b', transcript_lower) for w in SCALE_WORDS)
            if not has_scale_word:
                extracted["_flag"] = f"amount {extracted['amount']} may be missing a scale word (elfu/mia/thousand) — possible transcription drop"
    if extracted.get("speaker_action") not in VALID_SPEAKER_ACTIONS:
        extracted["_flag_invalid_speaker_action"] = extracted.get("speaker_action")
        extracted["speaker_action"] = "other"
    extracted.pop("reasoning", None)
    return extracted


def check_scheduled_payout_eligibility(member_name, amount):
    member = MOCK_MEMBERS.get(member_name)
    if not member:
        return False, f"{member_name} is not a registered member."

    position = member["payout_position"]
    scheduled_date = st.session_state.payout_schedule[position]
    today = st.session_state.simulated_today

    if position in st.session_state.payouts_made_positions:
        return False, f"{member_name}'s payout for {scheduled_date.strftime('%d/%m/%Y')} has already been made."

    if today < scheduled_date:
        return False, f"{member_name} is not due yet. Scheduled date is {scheduled_date.strftime('%d/%m/%Y')}, today is {today.strftime('%d/%m/%Y')}."

    return True, f"{member_name} is eligible — scheduled date {scheduled_date.strftime('%d/%m/%Y')} has been reached. [SIMULATED] Would send {amount} to their registered phone number."


def mark_scheduled_payout_made(member_name):
    member = MOCK_MEMBERS.get(member_name)
    position = member["payout_position"]
    num_members = len(MOCK_MEMBERS)
    old_date = st.session_state.payout_schedule[position]
    old_offset = (old_date.year - SCHEDULE_START_YEAR) * 12 + (old_date.month - SCHEDULE_START_MONTH)
    new_offset = old_offset + num_members
    st.session_state.payout_schedule[position] = compute_date_for_offset(new_offset)


def check_loan_eligibility(member_name, amount):
    member = MOCK_MEMBERS.get(member_name)
    if not member:
        return False, f"{member_name} is not a registered member."
    if amount > member["loan_eligible"]:
        return False, f"{member_name} requested {amount}, which exceeds their eligible limit of {member['loan_eligible']}."
    return True, f"{member_name} is eligible for a loan of {amount}. [SIMULATED] Would send funds to their registered phone number."


def generate_tts(text, voice_accent="swahili", voice_gender="female", voice_language="en"):
    url = "https://infer.voice.intron.io/tts/v1/generate"
    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "voice_accent": voice_accent, "voice_gender": voice_gender, "voice_language": voice_language}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    return response.json()["data"]["audio_path"]


def generate_confirmation_text(entry):
    if entry["speaker_action"] == "deposit":
        text = f"Confirmed: deposit of {entry['amount']} logged."
        if entry["referenced_member"] != "none":
            text += f" Noted: {entry['referenced_member']} — {entry['referenced_member_context']}."
    elif entry["speaker_action"] == "initiate_scheduled_payout":
        text = f"Checking scheduled payout for {entry['referenced_member']}."
    elif entry["speaker_action"] == "initiate_loan_payout":
        text = f"Checking loan payout for {entry['referenced_member']}."
    elif entry["speaker_action"] == "loan_request":
        text = f"Your loan request for {entry['amount']} is ready. Please confirm below to submit it for admin approval."
    elif entry["speaker_action"] == "late_payment_note":
        text = "Noted: you'll be sending your payment late. No amount logged yet."
    elif entry["speaker_action"] == "membership_update":
        text = "Group update noted."
    else:
        text = "Message logged, no financial action taken."
    return text


def word_accuracy(ground_truth, model_output):
    if not model_output or model_output.startswith("[failed"):
        return 0.0

    def clean_words(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    gt_words = clean_words(ground_truth)
    model_words = clean_words(model_output)
    if not gt_words:
        return 0.0
    correct = len(gt_words & model_words)
    return round((correct / len(gt_words)) * 100, 1)


st.set_page_config(page_title="Habahub", page_icon="🎙️", layout="wide")

if "ledger" not in st.session_state:
    st.session_state.ledger = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []
if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None
if "loan_requests" not in st.session_state:
    st.session_state.loan_requests = []
if "pending_loan_review" not in st.session_state:
    st.session_state.pending_loan_review = None

st.sidebar.title("🎙️ Habahub")
st.sidebar.markdown(f"**{CURRENT_USER}** ({'Admin' if IS_ADMIN else 'Member'})")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Dashboard", "Record & Query", "Benchmark"])

def render_header(title):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(title)
    with col2:
        st.write("")
        with st.popover(f"👤 {CURRENT_USER}"):
            st.write(f"**{CURRENT_USER}**")
            st.caption("Admin" if IS_ADMIN else "Member")
            st.divider()
            new_user = st.selectbox("Switch user (demo only)", list(USERS.keys()), index=list(USERS.keys()).index(CURRENT_USER), key="user_switcher")
            if new_user != CURRENT_USER:
                st.session_state.current_user_name = new_user
                st.rerun()
            st.button("Log out", disabled=True, help="Demo only — not functional")

speaking_as = CURRENT_USER

# ============================================================
# PAGE 1: Record & Query
# ============================================================
if page == "Record & Query":
    render_header("Record & Query")
    st.caption(f"Logged in as: {speaking_as}")
    st.write("Speak a contribution, payment note, loan request, or update — or ask a question about your account — in English, Swahili, or both.")
    audio = st.audio_input("Record your message")

    if audio is not None:
        audio_bytes = audio.getvalue()
        audio_id = hash(audio_bytes)

        if st.session_state.last_processed_audio_id != audio_id:
            st.session_state.last_processed_audio_id = audio_id

            with open("temp_input.wav", "wb") as f:
                f.write(audio_bytes)

            with st.spinner("Transcribing..."):
                transcript = transcribe_sahara("temp_input.wav")
            st.session_state.current_transcript = transcript

            with st.spinner("Running benchmark comparison..."):
                try:
                    whisper_result = transcribe_whisper_hf("temp_input.wav")
                except Exception as e:
                    whisper_result = f"[failed: {e}]"
                try:
                    mms_result = transcribe_mms("temp_input.wav")
                except Exception as e:
                    mms_result = f"[failed: {e}]"

            st.session_state.benchmark_results.append({
                "ground_truth": transcript,
                "sahara": transcript,
                "whisper": whisper_result,
                "mms": mms_result
            })

            with st.spinner("Understanding your message..."):
                intent = classify_intent(transcript)
            st.session_state.current_intent = intent

            if intent != "question":
                with st.spinner("Extracting details..."):
                    extracted_raw = extract_chama_action(transcript, speaking_as)
                    extracted = json.loads(extracted_raw) if isinstance(extracted_raw, str) else extracted_raw
                    extracted = validate_extraction(transcript, extracted)
                st.session_state.current_extracted = extracted

        transcript = st.session_state.current_transcript
        intent = st.session_state.current_intent
        st.write("**Transcript:**", transcript)

        if intent == "question":
            with st.spinner("Checking records..."):
                answer = answer_member_query(transcript, speaking_as)
            st.write("**Answer:**", answer)
            with st.spinner("Generating spoken answer..."):
                audio_url = generate_tts(answer)
            st.audio(audio_url)

        else:
            extracted = st.session_state.current_extracted
            st.json(extracted)

            if "_flag" in extracted:
                st.warning(f"⚠️ {extracted['_flag']}")
            if "_flag_invalid_speaker_action" in extracted:
                st.warning(f"⚠️ Unrecognized action type: '{extracted['_flag_invalid_speaker_action']}' — defaulted to 'other'.")

            allowed_actions = list(ADMIN_ALLOWED_ACTIONS) if IS_ADMIN else list(MEMBER_ALLOWED_ACTIONS)
            if extracted["speaker_action"] not in allowed_actions:
                st.warning(f"⚠️ '{extracted['speaker_action']}' is not permitted for your role. Defaulted to 'other'.")
                extracted["speaker_action"] = "other"

            st.subheader("Review before submitting")
            edited_speaker_action = st.selectbox(
                "Action", allowed_actions,
                index=allowed_actions.index(extracted["speaker_action"])
            )
            edited_amount = st.number_input(
                "Amount", value=float(extracted["amount"]) if extracted["amount"] is not None else 0.0,
                min_value=0.0, step=1.0
            )

            if IS_ADMIN:
                edited_referenced_member = st.text_input("Referenced member", value=extracted["referenced_member"])
            else:
                st.text_input("Referenced member", value="none", disabled=True, help="Members can only log actions about themselves.")
                edited_referenced_member = "none"

            if st.button("Confirm and submit"):
                if not IS_ADMIN and edited_referenced_member not in ("none", CURRENT_USER):
                    st.error("❌ Members can only log actions about themselves.")
                else:
                    was_amended = (
                        edited_speaker_action != extracted["speaker_action"]
                        or (extracted["amount"] is not None and edited_amount != extracted["amount"])
                        or (extracted["amount"] is None and edited_amount != 0.0)
                        or edited_referenced_member != extracted["referenced_member"]
                    )
                    ai_action_taken = False

                    final_entry = dict(extracted)
                    final_entry["speaker"] = speaking_as
                    final_entry["speaker_action"] = edited_speaker_action
                    final_entry["amount"] = edited_amount if edited_amount > 0 else None
                    final_entry["referenced_member"] = edited_referenced_member
                    final_entry["amended_by_human"] = was_amended
                    final_entry["ai_original_amount"] = extracted["amount"]
                    final_entry["ai_original_speaker_action"] = extracted["speaker_action"]

                    confirmation_text = generate_confirmation_text(final_entry)

                    if edited_speaker_action == "initiate_scheduled_payout" and IS_ADMIN and edited_referenced_member not in ("none", ""):
                        eligible, elig_message = check_scheduled_payout_eligibility(edited_referenced_member, edited_amount)
                        ai_action_taken = True
                        if eligible:
                            mark_scheduled_payout_made(edited_referenced_member)
                            st.success(f"✅ SIMULATED PAYMENT — {elig_message}")
                            confirmation_text += f" Payment of {edited_amount} sent to {edited_referenced_member}. Simulated transaction — no real funds moved. Their schedule has been updated to the next cycle."
                        else:
                            st.error(f"❌ Payment blocked (simulated check) — {elig_message}")
                            confirmation_text += f" Payment blocked. {elig_message}"

                    if edited_speaker_action == "initiate_loan_payout" and IS_ADMIN and edited_referenced_member not in ("none", ""):
                        ai_action_taken = True
                        pending_loan = get_pending_loan_for_member(edited_referenced_member)
                        if not pending_loan:
                            st.error(f"❌ No pending loan request found for {edited_referenced_member}.")
                            confirmation_text += f" No pending loan request found for {edited_referenced_member}."
                        else:
                            eligible, elig_message = check_loan_eligibility(edited_referenced_member, pending_loan["amount"])
                            idx = st.session_state.loan_requests.index(pending_loan)
                            if eligible:
                                st.session_state.loan_requests[idx]["status"] = "approved"
                                st.success(f"✅ SIMULATED LOAN PAYMENT — {elig_message}")
                                confirmation_text += f" Loan payout of {pending_loan['amount']} approved and simulated as sent to {edited_referenced_member}."
                            else:
                                st.error(f"❌ Loan payout blocked — {elig_message}")
                                confirmation_text += f" Loan payout blocked. {elig_message}"

                    final_entry["ai_action_taken"] = ai_action_taken
                    st.session_state.ledger.append(final_entry)

                    if edited_speaker_action == "loan_request":
                        st.session_state.pending_loan_review = {
                            "member": speaking_as,
                            "amount": edited_amount
                        }

                    if edited_speaker_action == "late_payment_note":
                        st.session_state.show_reminder_offer = True
                        st.session_state.reminder_offer_member = speaking_as
                        st.session_state.reminder_offer_amount = final_entry.get("amount")

                    with st.spinner("Generating spoken confirmation..."):
                        audio_url = generate_tts(confirmation_text)
                    st.write("**Confirmation:**", confirmation_text)
                    st.audio(audio_url)

            if st.session_state.pending_loan_review:
                st.divider()
                st.subheader("Confirm loan request")
                lr = st.session_state.pending_loan_review
                st.write(f"Requesting **{lr['amount']}** on behalf of **{lr['member']}**.")
                col1, col2 = st.columns(2)
                if col1.button("Submit to admin for approval"):
                    st.session_state.loan_requests.append({
                        "member": lr["member"],
                        "amount": lr["amount"],
                        "status": "pending"
                    })
                    st.success("✅ Loan request submitted to admin for approval.")
                    st.session_state.pending_loan_review = None
                    st.rerun()
                if col2.button("Cancel request"):
                    st.session_state.pending_loan_review = None
                    st.rerun()

            if st.session_state.get("show_reminder_offer"):
                st.divider()
                next_friday = get_next_friday(st.session_state.simulated_today)
                st.subheader("📅 Schedule a payment reminder?")
                st.write(f"Would you like a reminder SMS scheduled for **next Friday, {next_friday.strftime('%d/%m/%Y')} at 9:00 AM**?")

                existing_amount = st.session_state.reminder_offer_amount
                if existing_amount is None:
                    st.info("No amount was mentioned in your message — please enter one to schedule the reminder.")
                reminder_amount = st.number_input(
                    "Amount owed",
                    value=float(existing_amount) if existing_amount is not None else 0.0,
                    min_value=0.0, step=1.0,
                    key="reminder_amount_input"
                )
                reminder_category = st.selectbox("Payment type", ["Monthly Contribution", "Loan"], key="reminder_category_input")

                if st.button("Yes, schedule reminder"):
                    if reminder_amount <= 0:
                        st.error("❌ Please enter an amount greater than 0 before scheduling.")
                    else:
                        st.session_state.scheduled_reminders.append({
                            "member": st.session_state.reminder_offer_member,
                            "amount": reminder_amount,
                            "category": reminder_category,
                            "date": next_friday.strftime("%d/%m/%Y"),
                            "time": "09:00",
                            "status": "scheduled"
                        })
                        st.success(f"✅ SIMULATED — {reminder_category} reminder of {reminder_amount} scheduled for {st.session_state.reminder_offer_member} on {next_friday.strftime('%d/%m/%Y')} at 9:00 AM. No real SMS sent.")
                        st.session_state.show_reminder_offer = False
                        st.rerun()
                if st.button("No thanks"):
                    st.session_state.show_reminder_offer = False
                    st.rerun()

    st.divider()
    st.subheader("Session Ledger")
    st.dataframe(st.session_state.ledger, width='stretch')

# ============================================================
# PAGE 2: Dashboard
# ============================================================
elif page == "Dashboard":
    render_header("Dashboard")
    st.caption("⚠️ Illustrative demo data — not live transaction history.")

    if IS_ADMIN:
        with st.expander("🛠️ Demo controls"):
            st.caption("Real 'today' is before most scheduled payout dates, so this lets you simulate a later date to demo payout eligibility live. Not a real clock override.")
            new_sim_date = st.date_input("Simulate current date", value=st.session_state.simulated_today)
            if new_sim_date != st.session_state.simulated_today:
                st.session_state.simulated_today = new_sim_date
                st.rerun()

        total_ytd = sum(m["ytd_contributed"] for m in MOCK_MEMBERS.values())
        this_month = MOCK_MONTHLY_TOTALS["Collected"].iloc[CURRENT_MONTH_INDEX - 1]
        last_month = MOCK_MONTHLY_TOTALS["Collected"].iloc[CURRENT_MONTH_INDEX - 2] if CURRENT_MONTH_INDEX >= 2 else 0
        paid_this_month = sum(1 for m in MOCK_MEMBERS.values() if m["this_month_paid"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total collected (YTD)", f"KES {total_ytd:,}")
        col2.metric("Collected this month", f"KES {this_month:,}", delta=f"{this_month - last_month:+,}")
        col3.metric("Members paid this month", f"{paid_this_month}/{len(MOCK_MEMBERS)}")
        col4.metric("Total owed to group", f"KES {sum(m['owed'] for m in MOCK_MEMBERS.values()):,}")

        st.subheader("Monthly collections (year to date)")
        months_to_show = MONTHS_FULL[:CURRENT_MONTH_INDEX]
        collected_to_show = MOCK_MONTHLY_TOTALS["Collected"].tolist()[:CURRENT_MONTH_INDEX]
        monthly_df = pd.DataFrame({"Month": months_to_show, "Collected": collected_to_show})
        monthly_df["Month"] = pd.Categorical(monthly_df["Month"], categories=months_to_show, ordered=True)
        st.bar_chart(monthly_df.set_index("Month"))

        st.subheader("Member contributions")
        member_df = pd.DataFrame([
            {"Member": name, "YTD Contributed": v["ytd_contributed"], "Paid this month": "✅" if v["this_month_paid"] else "❌",
             "Last paid": v["last_paid"], "Owed": v["owed"], "Loan eligible": v["loan_eligible"]}
            for name, v in MOCK_MEMBERS.items()
        ])
        st.dataframe(member_df, width='stretch', hide_index=True)

        st.subheader("Payout rotation order")
        rotation_df = pd.DataFrame([
            {"Position": v["payout_position"], "Member": name, "Scheduled Date": st.session_state.payout_schedule[v["payout_position"]].strftime("%d/%m/%Y")}
            for name, v in MOCK_MEMBERS.items()
        ]).sort_values("Position")
        st.dataframe(rotation_df, width='stretch', hide_index=True)

        st.subheader("Loan requests")
        tab_pending, tab_approved, tab_rejected = st.tabs(["Pending", "Approved", "Rejected"])

        with tab_pending:
            pending = [lr for lr in st.session_state.loan_requests if lr["status"] == "pending"]
            if not pending:
                st.info("No pending loan requests.")
            else:
                for idx, lr in enumerate(st.session_state.loan_requests):
                    if lr["status"] != "pending":
                        continue
                    col1, col2, col3 = st.columns([3, 2, 2])
                    col1.write(f"**{lr['member']}** requests **{lr['amount']}**")
                    if col2.button("Approve", key=f"approve_{idx}"):
                        eligible, elig_message = check_loan_eligibility(lr["member"], lr["amount"])
                        if eligible:
                            st.session_state.loan_requests[idx]["status"] = "approved"
                            st.session_state.ledger.append({
                                "speaker": CURRENT_USER,
                                "speaker_action": "initiate_loan_payout",
                                "amount": lr["amount"],
                                "referenced_member": lr["member"],
                                "referenced_member_context": "loan approval",
                                "action_type": "payout",
                                "amended_by_human": False,
                                "ai_action_taken": True
                            })
                            st.success(f"✅ SIMULATED LOAN PAYMENT — {elig_message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {elig_message}")
                    if col3.button("Reject", key=f"reject_{idx}"):
                        st.session_state.loan_requests[idx]["status"] = "rejected"
                        st.rerun()

        with tab_approved:
            approved = [lr for lr in st.session_state.loan_requests if lr["status"] == "approved"]
            if not approved:
                st.info("No approved loan requests yet.")
            else:
                st.dataframe(pd.DataFrame(approved), width='stretch', hide_index=True)

        with tab_rejected:
            rejected = [lr for lr in st.session_state.loan_requests if lr["status"] == "rejected"]
            if not rejected:
                st.info("No rejected loan requests.")
            else:
                st.dataframe(pd.DataFrame(rejected), width='stretch', hide_index=True)

        st.subheader("Scheduled payment reminders")
        st.caption("Members' self-reported commitments to pay late — for tracking, not enforcement.")
        if not st.session_state.scheduled_reminders:
            st.info("No scheduled reminders yet.")
        else:
            reminders_df = pd.DataFrame(st.session_state.scheduled_reminders)
            reminders_df["Amount"] = reminders_df["amount"].apply(lambda a: f"KES {a:,.0f}" if a else "Not specified")
            display_df = reminders_df[["member", "Amount", "category", "date", "time", "status"]].rename(columns={
                "member": "Member", "category": "Type", "date": "Date", "time": "Time", "status": "Status"
            })
            st.dataframe(display_df, width='stretch', hide_index=True)

    else:
        my_data = MOCK_MEMBERS.get(CURRENT_USER, {})
        st.subheader(f"Your account — {CURRENT_USER}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Your YTD contributions", f"KES {my_data.get('ytd_contributed', 0):,}")
        col2.metric("Paid this month", "✅" if my_data.get("this_month_paid") else "❌")
        col3.metric("Amount owed", f"KES {my_data.get('owed', 0):,}")
        st.write(f"**Last paid:** {my_data.get('last_paid', 'N/A')}")
        st.write(f"**Loan eligible up to:** KES {my_data.get('loan_eligible', 0):,}")

        my_position = my_data.get('payout_position')
        my_date = st.session_state.payout_schedule.get(my_position)
        st.write(f"**Your payout position:** {my_position} (scheduled: {my_date.strftime('%d/%m/%Y') if my_date else 'N/A'})")

        st.subheader("Your monthly contributions (year to date)")
        months_to_show = MONTHS_FULL[:CURRENT_MONTH_INDEX]
        my_monthly_values = MOCK_MEMBER_MONTHLY.get(CURRENT_USER, [0]*12)[:CURRENT_MONTH_INDEX]
        my_monthly_df = pd.DataFrame({"Month": months_to_show, "Contributed": my_monthly_values})
        my_monthly_df["Month"] = pd.Categorical(my_monthly_df["Month"], categories=months_to_show, ordered=True)
        st.bar_chart(my_monthly_df.set_index("Month"))

        st.subheader("Your loan requests")
        my_loans = [lr for lr in st.session_state.loan_requests if lr["member"] == CURRENT_USER]
        if not my_loans:
            st.info("No loan requests yet.")
        else:
            st.dataframe(pd.DataFrame(my_loans), width='stretch', hide_index=True)

# ============================================================
# PAGE 3: Benchmark
# ============================================================
elif page == "Benchmark":
    render_header("Benchmark")
    st.caption("Live results from Record & Query, benchmarked against editable ground truth.")

    if not st.session_state.benchmark_results:
        st.info("No recordings yet — use Record & Query to generate benchmark data.")
    else:
        all_sahara, all_whisper, all_mms = [], [], []

        for i, entry in enumerate(st.session_state.benchmark_results):
            st.markdown(f"**Recording {i+1}**")

            col1, col2 = st.columns([5, 1])
            with col1:
                gt_input = st.text_area(
                    f"Ground truth — Recording {i+1}",
                    value=entry["ground_truth"],
                    key=f"gt_input_{i}",
                    label_visibility="collapsed"
                )
            with col2:
                st.write("")
                if st.button("Update", key=f"update_btn_{i}"):
                    st.session_state.benchmark_results[i]["ground_truth"] = gt_input
                    st.rerun()

            applied_gt = st.session_state.benchmark_results[i]["ground_truth"]

            sahara_acc = word_accuracy(applied_gt, entry["sahara"])
            whisper_acc = word_accuracy(applied_gt, entry["whisper"])
            mms_acc = word_accuracy(applied_gt, entry["mms"])

            all_sahara.append(sahara_acc)
            all_whisper.append(whisper_acc)
            all_mms.append(mms_acc)

            comp_df = pd.DataFrame([
                {"Model": "Sahara", "Transcript": entry["sahara"], "Word Accuracy": f"{sahara_acc}%"},
                {"Model": "Whisper", "Transcript": entry["whisper"], "Word Accuracy": f"{whisper_acc}%"},
                {"Model": "MMS", "Transcript": entry["mms"], "Word Accuracy": f"{mms_acc}%"},
            ])
            st.dataframe(comp_df, width='stretch', hide_index=True)
            st.divider()

        st.subheader("Average accuracy across all recordings")
        avg_col1, avg_col2, avg_col3 = st.columns(3)
        avg_col1.metric("Sahara avg", f"{round(sum(all_sahara)/len(all_sahara), 1)}%")
        avg_col2.metric("Whisper avg", f"{round(sum(all_whisper)/len(all_whisper), 1)}%")
        avg_col3.metric("MMS avg", f"{round(sum(all_mms)/len(all_mms), 1)}%")