import streamlit as st
import requests
import json
from groq import Groq
import re
import pandas as pd
import calendar
import zipfile
import io
from datetime import date, timedelta

SAHARA_API_KEY = st.secrets["SAHARA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)

BONGO_NAME = "Bongo"

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

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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

BENCHMARK_SAMPLE_START_INDEX = 4

# --- Unified loan economics: 8% interest, 3% processing fee, 2% penalty per month late ---
LOAN_REQUEST_DEFAULT_INTEREST = 0.08
LOAN_REQUEST_DEFAULT_FEE = 0.03
PENALTY_RATE_PER_MONTH_LATE = 0.02


# ============================================================
# Date / schedule helpers
# ============================================================

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


def get_next_weekday_on_or_after(from_date, weekday_name, strictly_after=True):
    if weekday_name not in WEEKDAY_NAMES:
        return None
    target = WEEKDAY_NAMES.index(weekday_name)
    days_ahead = target - from_date.weekday()
    if days_ahead < 0 or (days_ahead == 0 and strictly_after):
        days_ahead += 7
    return from_date + timedelta(days=days_ahead)


def get_next_friday(from_date):
    return get_next_weekday_on_or_after(from_date, "Friday")


def get_first_of_next_month(from_date):
    if from_date.month == 12:
        return date(from_date.year + 1, 1, 1)
    return date(from_date.year, from_date.month + 1, 1)


if "payout_schedule" not in st.session_state:
    st.session_state.payout_schedule = build_initial_schedule()
if "payouts_made_positions" not in st.session_state:
    st.session_state.payouts_made_positions = set()
if "simulated_today" not in st.session_state:
    st.session_state.simulated_today = date.today()
if "scheduled_reminders" not in st.session_state:
    st.session_state.scheduled_reminders = []
if "benchmark_samples" not in st.session_state:
    st.session_state.benchmark_samples = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


def get_pending_loan_for_member(member_name):
    for lr in st.session_state.loan_requests:
        if lr["member"] == member_name and lr["status"] == "pending":
            return lr
    return None


def log_chat(member_name, role, text):
    """role is 'user' or 'bongo'. Chat history is per-member, not shared across users."""
    st.session_state.chat_history.setdefault(member_name, []).append({"role": role, "text": text})


# ============================================================
# Sahara STT / TTS
# ============================================================

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


def generate_tts(text, voice_accent="swahili", voice_gender="female", voice_language="en"):
    url = "https://infer.voice.intron.io/tts/v1/generate"
    headers = {"Authorization": f"Bearer {SAHARA_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "voice_accent": voice_accent, "voice_gender": voice_gender, "voice_language": voice_language}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    return response.json()["data"]["audio_path"]


# ============================================================
# Groq: intent, query routing, extraction, commitment-date parsing
# ============================================================

def classify_intent(transcript):
    prompt = f"""Classify this chama voice message as either "statement" or "question".

"statement" means the speaker is DIRECTLY INITIATING an action right now, with enough specifics to act on
(e.g. "I want to borrow 5000", "Naomba mkopo wa elfu tano", "Tuma pesa kwa Grace", "I've paid my
contribution", "Approve Juma's loan"). This includes future-tense payment commitments like "Nitalipa
Ijumaa" since those log a real commitment.

"question" means the speaker is seeking information, asking HOW something works, asking IF something is
possible, or asking for clarification — even if the topic is loans, payouts, or payments — as long as they
are NOT giving a specific amount/action to execute right now. Examples: "How do I apply for a loan?",
"Ninawezaje kuomba mkopo?", "Can members borrow money?", "What happens if I pay late?", "When is my turn?".
These are questions, not requests, even though they mention loans or payments.

The key test: does the speaker give a concrete amount, name, or date to act on RIGHT NOW? If yes →
statement. If they're only asking about process, possibility, or general information → question.

Transcript: "{transcript}"

Respond with ONLY one word: "statement" or "question"."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


def classify_query_category(transcript):
    prompt = f"""Classify this chama question into exactly one category:
- "own_account": about the speaker's own balance, payments, loan status/terms, or their own payout position
- "payout_rotation": asking who is next to receive the scheduled monthly payout, or when
- "admin_aggregate": asking about OTHER members' data, group-wide totals, loan statistics/counts, who has/hasn't paid, or any bookkeeping question not about the speaker themselves
- "definition": asking what a term or concept means (e.g. "what is interest")
- "other": anything else

Transcript: "{transcript}"

Respond with ONLY one word: own_account, payout_rotation, admin_aggregate, definition, or other."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


def get_member_loan_context(member_name):
    """Real, computed loan figures for this member — used so Bongo's follow-up answers (penalty in KES,
    first payment date, etc.) come from actual numbers, never estimates from memory."""
    context = {"pending_loan": None, "approved_loans": []}

    pending = get_pending_loan_for_member(member_name)
    if pending:
        dur = pending.get("suggested_duration_months", 4)
        est = calculate_loan_estimate(pending["amount"], dur)
        penalty_kes = round(pending["amount"] * PENALTY_RATE_PER_MONTH_LATE, 2)
        context["pending_loan"] = {
            "amount": pending["amount"],
            "duration_months": dur,
            "estimated_monthly_payment": est["monthly"],
            "estimated_total_repayable": est["total"],
            "penalty_per_late_payment_kes": penalty_kes,
            "first_payment_note": "First payment will be due 30 days after admin approval — not yet approved, so no exact date yet."
        }

    for lr in st.session_state.loan_requests:
        if lr["member"] == member_name and lr["status"] == "approved":
            lr = update_schedule_penalties(lr)
            first_due = lr["schedule"][0]["due_date"] if lr.get("schedule") else None
            context["approved_loans"].append({
                "principal": lr["amount"],
                "monthly_payment": lr["monthly_payment"],
                "total_repayable": lr["total_repayable"],
                "penalty_per_late_payment_kes": round(lr["monthly_payment"] * PENALTY_RATE_PER_MONTH_LATE, 2),
                "first_payment_due_date": first_due,
                "installment_schedule": lr["schedule"],
            })

    return context


def answer_member_query(transcript, member_name, is_admin):
    member_record = dict(MOCK_MEMBERS.get(member_name, {}))
    member_record["loans"] = get_member_loan_context(member_name)
    fallback_contact = "customer service" if is_admin else "the chama admin"

    prompt = f"""You are {BONGO_NAME}, a friendly chama (savings group) voice assistant, speaking DIRECTLY
to the person whose data this is. The message may be in English, Swahili, or a mix.

ALWAYS address the speaker as "you" / "your" — this data belongs to them personally. Never refer to them
by name in the third person (e.g. never say "Juma's balance is..." — say "your balance is...").

There are four kinds of questions you might be asked:
1. Questions about THEIR OWN account or loans (balance, when they last paid, loan terms, penalty amounts,
   first payment date). Answer using ONLY the data provided below, including the "loans" section. Penalties
   are always 2% of the relevant amount (the loan principal if not yet approved, or that month's installment
   if approved) — use the exact penalty_per_late_payment_kes figures already computed in the data, do not
   recalculate or guess.
2. Questions asking what a TERM or CONCEPT means. Answer in plain, simple language.
3. Questions asking HOW something works or HOW to do something (e.g. "how do I request a loan", "how can I
   schedule a payment reminder", "ninawezaje kuomba mkopo") — this is a CLARIFICATION, not a request. Explain
   the process in plain language (mention relevant numbers like their own eligibility, or the interest/fee/
   penalty rates, if helpful), then end by offering to help them actually do it, e.g. "Would you like me to
   start that for you now — just tell me the amount." Do NOT treat this as if the action has already happened.
4. Anything you cannot answer from the data below or a general explanation — do NOT guess or invent an
   answer. Instead, respond exactly along these lines: "I don't have that information — please contact
   {fallback_contact} for help with that."

Member data for {member_name} (illustrative demo data): {json.dumps(member_record)}

Question: "{transcript}"

Give a short, direct, spoken-style answer (1-3 sentences, up to 4 for process explanations)."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def answer_payout_rotation_query(member_name, is_admin):
    """Members get only position numbers (no names). Admins get the recipient's name and their own date too."""
    positions_sorted = sorted(MOCK_MEMBERS.items(), key=lambda x: x[1]["payout_position"])
    next_position, next_name = None, None
    for name, data in positions_sorted:
        pos = data["payout_position"]
        if pos not in st.session_state.payouts_made_positions:
            next_position, next_name = pos, name
            break

    my_position = MOCK_MEMBERS.get(member_name, {}).get("payout_position")

    if is_admin:
        next_date = st.session_state.payout_schedule.get(next_position)
        my_date = st.session_state.payout_schedule.get(my_position)
        return (
            f"Position {next_position} is next — that's {next_name}, scheduled for "
            f"{next_date.strftime('%d/%m/%Y') if next_date else 'an unset date'}. "
            f"Your own payout (position {my_position}) is scheduled for "
            f"{my_date.strftime('%d/%m/%Y') if my_date else 'an unset date'}."
        )
    else:
        return f"Position {next_position} is next in line to receive the payout. You are position {my_position}."


def answer_admin_aggregate_query(transcript):
    """Answers bookkeeping-style questions from real session data, filtered as requested. Admin-only — never called for members."""
    approved = [lr for lr in st.session_state.loan_requests if lr["status"] == "approved"]
    this_month_start = date(st.session_state.simulated_today.year, st.session_state.simulated_today.month, 1)

    this_month_loans = []
    for lr in approved:
        approval_date_str = lr.get("approval_date")
        if not approval_date_str:
            continue
        try:
            approval_dt = pd.to_datetime(approval_date_str, format="%d/%m/%Y").date()
        except (ValueError, TypeError):
            continue
        if approval_dt >= this_month_start:
            this_month_loans.append(lr)

    total_this_month = sum(lr["amount"] for lr in this_month_loans)

    summary_data = {
        "loans_approved_this_month_count": len(this_month_loans),
        "loans_approved_this_month_total": total_this_month,
        "total_pending_loan_requests": len([lr for lr in st.session_state.loan_requests if lr["status"] == "pending"]),
        "total_rejected_loan_requests": len([lr for lr in st.session_state.loan_requests if lr["status"] == "rejected"]),
        "members_not_paid_this_month": [n for n, d in MOCK_MEMBERS.items() if not d["this_month_paid"]],
        "total_owed_group_wide": sum(d["owed"] for d in MOCK_MEMBERS.values()),
    }

    prompt = f"""You are {BONGO_NAME}, a chama admin assistant with full bookkeeping access.
Answer using ONLY this data — do not invent numbers. If the question cannot be answered from this data,
say: "I don't have that information — please contact customer service for help with that."

Data: {json.dumps(summary_data)}

Question: "{transcript}"

Give a short, direct spoken-style answer (1-2 sentences)."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    table = [{"Member": lr["member"], "Amount": lr["amount"], "Approved": lr.get("approval_date", "N/A")} for lr in this_month_loans]
    return response.choices[0].message.content, table


def answer_restricted_query(member_name):
    """Fallback for members asking admin-only aggregate questions — states the restriction, then offers what they CAN know."""
    my_eligible = MOCK_MEMBERS.get(member_name, {}).get("loan_eligible", 0)
    return f"That information isn't available to members — it's for group administrators only. What I can tell you: you're eligible for a loan of up to {my_eligible}."


def extract_chama_action(transcript, speaker_name):
    if not transcript or len(transcript.strip()) < 5:
        return json.dumps({
            "reasoning": "Transcript empty or too short to contain meaningful speech.",
            "speaker_action": "other", "amount": None,
            "referenced_member": "none", "referenced_member_context": "none", "action_type": "other",
            "loan_duration_months": None
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
- If the speaker uses future tense (e.g. "nitatuma" / "I will send" / "nitalipa" / "I will pay") without saying they already paid, this is NOT a completed deposit — it is "late_payment_note".
- Vague quantity words (e.g. "kidogo" / "a little" / "some") are NOT numbers. Never convert them into a numeric guess.
- referenced_member must be an actual person's name, never a number or group description, and never {speaker_name} unless {speaker_name} refers to themselves. If none, use "none".
- If the speaker is COMMANDING a regular scheduled payout be sent to a member (e.g. "Tuma X kwa Y"), this is "initiate_scheduled_payout".
- If the speaker is asking to BORROW money for themselves, this is "loan_request".
- If the speaker (an admin) is APPROVING or releasing an already-requested loan payout for another member (e.g. "Approve mkopo wa Juma", "Send Juma's loan"), this is "initiate_loan_payout", and referenced_member is that OTHER member, not {speaker_name}.
- If the speaker (admin) specifies a loan duration in months (e.g. "kwa miezi sita" / "for six months"), extract it as loan_duration_months. If not mentioned, loan_duration_months is null.

Example 1:
Transcript: "Nimechelewa mwezi huu lakini nitatuma kiasi chote Ijumaa ijayo."
Correct extraction: {{"reasoning": "Speaker is reporting their own status. Future tense 'nitatuma' means not yet paid. No amount stated in digits or words.", "speaker_action": "late_payment_note", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "late_payment_note", "loan_duration_months": null}}

Example 2:
Transcript: "Nimetuma pesa kidogo leo."
Correct extraction: {{"reasoning": "Speaker completed an action ('nimetuma'). 'Kidogo' is a vague qualifier, not a digit or number word, so no real amount was stated.", "speaker_action": "deposit", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "deposit", "loan_duration_months": null}}

Example 3:
Transcript: "Tuma elfu mbili kwa Grace leo."
Correct extraction: {{"reasoning": "Speaker is issuing a command to send a scheduled payout to another member, Grace — not the speaker.", "speaker_action": "initiate_scheduled_payout", "amount": 2000, "referenced_member": "Grace", "referenced_member_context": "payout recipient", "action_type": "payout", "loan_duration_months": null}}

Example 4:
Transcript: "Naomba mkopo wa elfu tano."
Correct extraction: {{"reasoning": "Speaker is requesting to borrow money for themselves from the group. No other member named.", "speaker_action": "loan_request", "amount": 5000, "referenced_member": "none", "referenced_member_context": "none", "action_type": "other", "loan_duration_months": null}}

Example 5:
Transcript: "Approve mkopo wa Juma kwa miezi sita."
Correct extraction: {{"reasoning": "Admin is approving Juma's loan and specifying a 6-month repayment duration.", "speaker_action": "initiate_loan_payout", "amount": null, "referenced_member": "Juma", "referenced_member_context": "loan payout approval", "action_type": "payout", "loan_duration_months": 6}}

Example 6:
Transcript: "Nitalipa Ijumaa ijayo."
Correct extraction: {{"reasoning": "Speaker commits to a future payment ('nitalipa' = I will pay). No amount stated.", "speaker_action": "late_payment_note", "amount": null, "referenced_member": "none", "referenced_member_context": "none", "action_type": "late_payment_note", "loan_duration_months": null}}

Now extract from this transcript, spoken by {speaker_name}:
Transcript: "{transcript}"

First, in "reasoning", briefly state: (a) who is speaking (should be {speaker_name}), (b) what tense/timing they use,
(c) whether they personally took a financial action or are acting on behalf of / referencing someone else,
(d) whether an amount was stated as digits, as words, or not stated at all.
Then extract the six fields, matching your reasoning.

Respond with ONLY valid JSON in this exact structure, no other text:
{{"reasoning": "...", "speaker_action": "...", "amount": ..., "referenced_member": "...", "referenced_member_context": "...", "action_type": "...", "loan_duration_months": ...}}"""
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
    if "loan_duration_months" not in extracted:
        extracted["loan_duration_months"] = None
    extracted.pop("reasoning", None)
    return extracted


def parse_commitment_date_reference(transcript):
    """LLM classifies WHICH kind of date reference was spoken; Python does all actual date arithmetic."""
    prompt = f"""A chama member said the following, committing to pay at some future point.
Identify what kind of date reference they used.

Transcript: "{transcript}"

Respond with ONLY valid JSON in this exact structure:
{{"reference_type": "...", "weekday": "...", "days_ahead": ..., "explicit_day": ..., "explicit_month": ...}}

Where reference_type is exactly one of:
- "weekday" (e.g. "next Friday", "Ijumaa ijayo", "Jumatatu") — set "weekday" to the English weekday name (Monday..Sunday)
- "tomorrow" (e.g. "kesho")
- "day_after_tomorrow" (e.g. "keshokutwa")
- "next_week" (e.g. "wiki ijayo", generic, no specific day named)
- "next_month" (e.g. "mwezi ujao", "next month")
- "in_days" (e.g. "in three days") — set "days_ahead" to the integer number of days
- "explicit_date" (a specific calendar date was stated) — set "explicit_day" (1-31) and "explicit_month" (1-12)
- "unclear" (no usable date reference found)

Set fields that don't apply to null. Respond with ONLY the JSON, no other text."""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def resolve_commitment_date(reference, today):
    ref_type = reference.get("reference_type")
    try:
        if ref_type == "weekday":
            return get_next_weekday_on_or_after(today, reference.get("weekday"))
        elif ref_type == "tomorrow":
            return today + timedelta(days=1)
        elif ref_type == "day_after_tomorrow":
            return today + timedelta(days=2)
        elif ref_type == "next_week":
            return today + timedelta(days=7)
        elif ref_type == "next_month":
            return get_first_of_next_month(today)
        elif ref_type == "in_days":
            days = reference.get("days_ahead")
            return today + timedelta(days=int(days)) if days else None
        elif ref_type == "explicit_date":
            day = reference.get("explicit_day")
            month = reference.get("explicit_month")
            if day and month:
                year = today.year
                candidate = date(year, int(month), int(day))
                if candidate < today:
                    candidate = date(year + 1, int(month), int(day))
                return candidate
    except (ValueError, TypeError):
        return None
    return None


# ============================================================
# Payout / loan business logic
# ============================================================

def check_scheduled_payout_eligibility(member_name, amount, viewer=None):
    member = MOCK_MEMBERS.get(member_name)
    subject = personalize_subject(member_name, viewer)
    is_self = subject == "you"
    if not member:
        return False, f"{member_name} is not a registered member."

    position = member["payout_position"]
    scheduled_date = st.session_state.payout_schedule[position]
    today = st.session_state.simulated_today
    possessive = "your" if is_self else f"{member_name}'s"

    if position in st.session_state.payouts_made_positions:
        return False, f"{possessive} payout for {scheduled_date.strftime('%d/%m/%Y')} has already been made."

    if today < scheduled_date:
        verb = "You are" if is_self else f"{member_name} is"
        return False, f"{verb} not due yet. Scheduled date is {scheduled_date.strftime('%d/%m/%Y')}, today is {today.strftime('%d/%m/%Y')}."

    verb_are = "are" if is_self else "is"
    return True, f"{subject} {verb_are} eligible — scheduled date {scheduled_date.strftime('%d/%m/%Y')} has been reached. [SIMULATED] Would send {amount} to {'your' if is_self else 'their'} registered phone number."


def mark_scheduled_payout_made(member_name):
    member = MOCK_MEMBERS.get(member_name)
    position = member["payout_position"]
    num_members = len(MOCK_MEMBERS)
    old_date = st.session_state.payout_schedule[position]
    old_offset = (old_date.year - SCHEDULE_START_YEAR) * 12 + (old_date.month - SCHEDULE_START_MONTH)
    new_offset = old_offset + num_members
    st.session_state.payout_schedule[position] = compute_date_for_offset(new_offset)


def personalize_subject(name, viewer):
    """Returns 'you' if name is the same person the message is being delivered to, else their actual name.
    Prevents Juma from being told 'Juma is eligible...' inside his own account."""
    return "you" if viewer is not None and name == viewer else name


def check_loan_eligibility(member_name, amount, viewer=None):
    member = MOCK_MEMBERS.get(member_name)
    subject = personalize_subject(member_name, viewer)
    is_self = subject == "you"
    if not member:
        return False, f"{member_name} is not a registered member."
    if amount > member["loan_eligible"]:
        possessive = "your" if is_self else f"{member_name}'s"
        verb = "You requested" if is_self else f"{member_name} requested"
        return False, f"{verb} {amount}, which exceeds {possessive} eligible limit of {member['loan_eligible']}."
    verb_are = "are" if is_self else "is"
    return True, f"{subject} {verb_are} eligible for a loan of {amount}."


def calculate_loan_estimate(principal, duration_months, interest_rate=LOAN_REQUEST_DEFAULT_INTEREST, fee_rate=LOAN_REQUEST_DEFAULT_FEE):
    interest = round(principal * interest_rate, 2)
    fee = round(principal * fee_rate, 2)
    total = round(principal + interest + fee, 2)
    monthly = round(total / duration_months, 2) if duration_months else None
    return {"interest": interest, "fee": fee, "total": total, "monthly": monthly}


def calculate_loan_terms(principal, duration_months, interest_rate=LOAN_REQUEST_DEFAULT_INTEREST, processing_fee_rate=LOAN_REQUEST_DEFAULT_FEE, approval_date=None):
    """Flat (non-compounding) interest, calculated once on principal. Simplification, disclosed as such."""
    if approval_date is None:
        approval_date = st.session_state.simulated_today
    interest_amount = round(principal * interest_rate, 2)
    processing_fee = round(principal * processing_fee_rate, 2)
    total_repayable = round(principal + interest_amount + processing_fee, 2)
    monthly_payment = round(total_repayable / duration_months, 2)

    schedule = []
    for i in range(1, duration_months + 1):
        due_date = approval_date + timedelta(days=30 * i)
        schedule.append({
            "month": i,
            "due_date": due_date.strftime("%d/%m/%Y"),
            "amount_due": monthly_payment,
            "status": "pending",
            "penalty_accrued": 0.0
        })

    return {
        "interest_rate": interest_rate,
        "interest_amount": interest_amount,
        "processing_fee_rate": processing_fee_rate,
        "processing_fee": processing_fee,
        "duration_months": duration_months,
        "total_repayable": total_repayable,
        "monthly_payment": monthly_payment,
        "approval_date": approval_date.strftime("%d/%m/%Y"),
        "schedule": schedule,
    }


def update_schedule_penalties(loan_request):
    """Marks overdue installments and computes accrued penalty at 2%/month late. Does not yet mark installments as paid."""
    if "schedule" not in loan_request:
        return loan_request
    today = st.session_state.simulated_today
    for entry in loan_request["schedule"]:
        due = pd.to_datetime(entry["due_date"], format="%d/%m/%Y").date()
        if entry["status"] == "pending" and today > due:
            entry["status"] = "overdue"
            entry["penalty_accrued"] = round(entry["amount_due"] * PENALTY_RATE_PER_MONTH_LATE, 2)
    return loan_request


def generate_confirmation_text(entry):
    if entry["speaker_action"] == "deposit":
        text = f"Confirmed: deposit of {entry['amount']} logged."
        if entry["referenced_member"] != "none":
            text += f" Noted: {entry['referenced_member']} — {entry['referenced_member_context']}."
    elif entry["speaker_action"] == "initiate_scheduled_payout":
        text = f"Checking scheduled payout for {entry['referenced_member']}."
    elif entry["speaker_action"] == "initiate_loan_payout":
        text = f"Checking loan payout for {entry['referenced_member']}."
    elif entry["speaker_action"] == "late_payment_note":
        text = "Noted: you'll be sending your payment late."
    elif entry["speaker_action"] == "membership_update":
        text = "Group update noted."
    else:
        text = "Message logged, no financial action taken."
    return text


# ============================================================
# Page config + visual polish
# ============================================================

st.set_page_config(page_title="Habahub", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    :root {
        --habahub-primary: #1F7A5C;
        --habahub-accent: #E8A33D;
        --habahub-bg: #FAF7F2;
    }

    .stApp {
        background-color: var(--habahub-bg);
    }

    section[data-testid="stSidebar"] {
        background-color: #14503A;
    }
    section[data-testid="stSidebar"] * {
        color: #F5F0E6 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #EAE3D6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #6B6154;
    }
    [data-testid="stMetricValue"] {
        color: var(--habahub-primary);
    }

    h1, h2, h3 {
        color: #22332C;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton>button[kind="primary"] {
        background-color: var(--habahub-primary);
        border: none;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #EAE3D6;
        background-color: white;
    }

    hr {
        border-color: #EAE3D6;
    }

    .chat-bubble-user {
        text-align: right;
        background: #DCF8C6;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 6px 0;
        display: block;
        max-width: 85%;
        margin-left: auto;
    }
    .chat-bubble-bongo {
        text-align: left;
        background: #FFFFFF;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 6px 0;
        display: block;
        max-width: 85%;
        margin-right: auto;
        border: 1px solid #EAE3D6;
    }
</style>
""", unsafe_allow_html=True)

if "ledger" not in st.session_state:
    st.session_state.ledger = []
if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None
if "loan_requests" not in st.session_state:
    st.session_state.loan_requests = []
if "pending_loan_review" not in st.session_state:
    st.session_state.pending_loan_review = None
if "loan_summary_text" not in st.session_state:
    st.session_state.loan_summary_text = None
if "loan_summary_duration" not in st.session_state:
    st.session_state.loan_summary_duration = None

st.sidebar.title("🎙️ Habahub")
st.sidebar.markdown(f"**{CURRENT_USER}** ({'Admin' if IS_ADMIN else 'Member'})")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Dashboard", "Record & Query", "Benchmark Data"])

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
    st.caption(f"Logged in as: {speaking_as} — talking with {BONGO_NAME}")

    col_main, col_chat = st.columns([2, 1])

    with col_main:
        st.write("Speak a contribution, payment note, loan request, or update — or ask about your account, the payout rotation, or what a term means — in English, Swahili, or both.")
        audio = st.audio_input("Record your message")

        if audio is not None:
            audio_bytes = audio.getvalue()
            audio_id = hash(audio_bytes)

            if st.session_state.last_processed_audio_id != audio_id:
                st.session_state.last_processed_audio_id = audio_id
                st.session_state.current_audio_bytes = audio_bytes

                with open("temp_input.wav", "wb") as f:
                    f.write(audio_bytes)

                with st.spinner("Transcribing..."):
                    transcript = transcribe_sahara("temp_input.wav")
                st.session_state.current_transcript = transcript
                log_chat(speaking_as, "user", transcript)

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

            with st.expander("📥 Save this recording for the benchmark dataset"):
                gt_edit = st.text_area("Ground truth (edit to match what was actually said)", value=transcript, key="gt_edit_current")
                if st.button("Save as benchmark sample"):
                    sample_index = BENCHMARK_SAMPLE_START_INDEX + len(st.session_state.benchmark_samples)
                    st.session_state.benchmark_samples.append({
                        "index": sample_index,
                        "audio_bytes": st.session_state.current_audio_bytes,
                        "ground_truth": gt_edit
                    })
                    st.success(f"✅ Saved as testcase{sample_index}. View and download it on the Benchmark Data page.")

            if intent == "question":
                with st.spinner("Checking records..."):
                    category = classify_query_category(transcript)
                    loan_table = None
                    if category == "payout_rotation":
                        answer = answer_payout_rotation_query(speaking_as, IS_ADMIN)
                    elif category == "admin_aggregate":
                        if IS_ADMIN:
                            answer, loan_table = answer_admin_aggregate_query(transcript)
                        else:
                            answer = answer_restricted_query(speaking_as)
                    else:
                        answer = answer_member_query(transcript, speaking_as, IS_ADMIN)
                st.write("**Answer:**", answer)
                if loan_table:
                    st.dataframe(pd.DataFrame(loan_table), use_container_width=True, hide_index=True)
                log_chat(speaking_as, "bongo", answer)
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
                        block_loan_request = False

                        if edited_speaker_action == "loan_request":
                            eligible, elig_message = check_loan_eligibility(speaking_as, edited_amount, viewer=speaking_as)
                            ai_action_taken = True
                            if not eligible:
                                block_loan_request = True
                                confirmation_text = f"Sorry — {elig_message}"
                                st.error(f"❌ {elig_message}")
                            else:
                                st.session_state.pending_loan_review = {"member": speaking_as, "amount": edited_amount}
                                confirmation_text = f"Your requested amount of {edited_amount} is within your eligible limit. Let's set up the repayment details below."

                        if edited_speaker_action == "initiate_scheduled_payout" and IS_ADMIN and edited_referenced_member not in ("none", ""):
                            eligible, elig_message = check_scheduled_payout_eligibility(edited_referenced_member, edited_amount, viewer=speaking_as)
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
                            spoken_duration = extracted.get("loan_duration_months")
                            if not pending_loan:
                                st.error(f"❌ No pending loan request found for {edited_referenced_member}.")
                                confirmation_text += f" No pending loan request found for {edited_referenced_member}."
                            elif not spoken_duration:
                                st.error("❌ Please state the loan duration in months (e.g. 'kwa miezi sita') before approving.")
                                confirmation_text += " Please state the loan duration in months before approving."
                            else:
                                eligible, elig_message = check_loan_eligibility(edited_referenced_member, pending_loan["amount"], viewer=speaking_as)
                                idx = st.session_state.loan_requests.index(pending_loan)
                                if eligible:
                                    terms = calculate_loan_terms(pending_loan["amount"], spoken_duration)
                                    st.session_state.loan_requests[idx]["status"] = "approved"
                                    st.session_state.loan_requests[idx].update(terms)
                                    st.success(f"✅ SIMULATED LOAN PAYMENT — {elig_message} Monthly payment: {terms['monthly_payment']} over {spoken_duration} months.")
                                    confirmation_text += f" Loan payout of {pending_loan['amount']} approved over {spoken_duration} months. Monthly payment: {terms['monthly_payment']}."
                                else:
                                    st.error(f"❌ Loan payout blocked — {elig_message}")
                                    confirmation_text += f" Loan payout blocked. {elig_message}"

                        final_entry["ai_action_taken"] = ai_action_taken
                        if not block_loan_request:
                            st.session_state.ledger.append(final_entry)

                        if edited_speaker_action == "late_payment_note":
                            with st.spinner("Working out the date you mentioned..."):
                                ref = parse_commitment_date_reference(transcript)
                                resolved_date = resolve_commitment_date(ref, st.session_state.simulated_today)
                            st.session_state.show_reminder_offer = True
                            st.session_state.reminder_offer_member = speaking_as
                            st.session_state.reminder_offer_amount = final_entry.get("amount")
                            st.session_state.reminder_offer_date = resolved_date
                            st.session_state.reminder_offer_date_was_parsed = resolved_date is not None

                        log_chat(speaking_as, "bongo", confirmation_text)
                        with st.spinner("Generating spoken confirmation..."):
                            audio_url = generate_tts(confirmation_text)
                        st.write("**Confirmation:**", confirmation_text)
                        st.audio(audio_url)

                if st.session_state.pending_loan_review:
                    st.divider()
                    st.subheader(f"💬 {BONGO_NAME} needs a few more details")
                    lr = st.session_state.pending_loan_review
                    st.write(f"Before submitting your request for **{lr['amount']}**, choose your preferred repayment period.")
                    suggested_duration = st.number_input(
                        "Repayment period (months)", min_value=2, max_value=6, value=4, key="loan_req_duration"
                    )

                    if st.button(f"Get my loan summary from {BONGO_NAME}"):
                        estimate = calculate_loan_estimate(lr["amount"], suggested_duration)
                        penalty_kes = round(lr["amount"] * PENALTY_RATE_PER_MONTH_LATE, 2)
                        summary_text = (
                            f"Based on a loan of {lr['amount']} over {suggested_duration} months, at "
                            f"{LOAN_REQUEST_DEFAULT_INTEREST*100:.0f}% interest and {LOAN_REQUEST_DEFAULT_FEE*100:.0f}% processing fee, "
                            f"your total repayable amount is {estimate['total']}, meaning a monthly payment of {estimate['monthly']}. "
                            f"If any payment is late, a penalty of {PENALTY_RATE_PER_MONTH_LATE*100:.0f} percent applies — "
                            f"that's {penalty_kes} shillings per late payment. Your first payment will be due 30 days "
                            f"after your loan is approved by the admin."
                        )
                        st.session_state.loan_summary_text = summary_text
                        st.session_state.loan_summary_duration = suggested_duration
                        log_chat(speaking_as, "bongo", summary_text)
                        with st.spinner("Generating spoken summary..."):
                            audio_url = generate_tts(summary_text)
                        st.write(f"**{BONGO_NAME}:**", summary_text)
                        st.audio(audio_url)

                    if st.session_state.loan_summary_text:
                        st.info(st.session_state.loan_summary_text)
                        col1, col2 = st.columns(2)
                        if col1.button("Submit to admin for approval"):
                            st.session_state.loan_requests.append({
                                "member": lr["member"],
                                "amount": lr["amount"],
                                "suggested_duration_months": st.session_state.loan_summary_duration,
                                "status": "pending"
                            })
                            st.success("✅ Loan request submitted to admin for approval.")
                            st.session_state.pending_loan_review = None
                            st.session_state.loan_summary_text = None
                            st.rerun()
                        if col2.button("Cancel request"):
                            st.session_state.pending_loan_review = None
                            st.session_state.loan_summary_text = None
                            st.rerun()

                if st.session_state.get("show_reminder_offer"):
                    st.divider()
                    st.subheader("📅 Schedule a payment reminder?")

                    resolved_date = st.session_state.get("reminder_offer_date")
                    was_parsed = st.session_state.get("reminder_offer_date_was_parsed")

                    if was_parsed:
                        st.write(f"Based on what you said, you're planning to pay on **{resolved_date.strftime('%A, %d/%m/%Y')}**. We can send a reminder that morning at 9:00 AM.")
                    else:
                        st.info("We couldn't work out the exact date you meant — please pick one below.")
                        resolved_date = resolved_date or get_next_friday(st.session_state.simulated_today)

                    reminder_date = st.date_input("Reminder date", value=resolved_date, min_value=st.session_state.simulated_today, key="reminder_date_input")

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
                                "date": reminder_date.strftime("%d/%m/%Y"),
                                "time": "09:00",
                                "status": "scheduled"
                            })
                            reminder_msg = f"Reminder scheduled for {reminder_date.strftime('%d/%m/%Y')} at 9:00 AM. No real SMS sent — simulated."
                            log_chat(speaking_as, "bongo", reminder_msg)
                            st.success(f"✅ SIMULATED — {reminder_category} reminder of {reminder_amount} scheduled for {st.session_state.reminder_offer_member} on {reminder_date.strftime('%d/%m/%Y')} at 9:00 AM. No real SMS sent.")
                            st.session_state.show_reminder_offer = False
                            st.rerun()
                    if st.button("No thanks"):
                        st.session_state.show_reminder_offer = False
                        st.rerun()

        st.divider()
        st.subheader("Session Ledger")
        st.dataframe(st.session_state.ledger, use_container_width=True)

    with col_chat:
        st.markdown(f"### 💬 Chat with {BONGO_NAME}")
        st.caption("Your own conversation history — not visible to other users.")
        history = st.session_state.chat_history.get(speaking_as, [])
        chat_box = st.container(height=650)
        with chat_box:
            if not history:
                st.caption("No messages yet. Record something to start chatting with Bongo.")
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='chat-bubble-user'>{msg['text']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble-bongo'><b>{BONGO_NAME}:</b> {msg['text']}</div>", unsafe_allow_html=True)

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
        st.bar_chart(monthly_df.set_index("Month"), color="#1F7A5C")

        st.subheader("Member contributions")
        member_df = pd.DataFrame([
            {"Member": name, "YTD Contributed": v["ytd_contributed"], "Paid this month": "✅" if v["this_month_paid"] else "❌",
             "Last paid": v["last_paid"], "Owed": v["owed"], "Loan eligible": v["loan_eligible"]}
            for name, v in MOCK_MEMBERS.items()
        ])
        st.dataframe(member_df, use_container_width=True, hide_index=True)

        st.subheader("Payout rotation order")
        rotation_df = pd.DataFrame([
            {"Position": v["payout_position"], "Member": name, "Scheduled Date": st.session_state.payout_schedule[v["payout_position"]].strftime("%d/%m/%Y")}
            for name, v in MOCK_MEMBERS.items()
        ]).sort_values("Position")
        st.dataframe(rotation_df, use_container_width=True, hide_index=True)

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
                    st.write(f"**{lr['member']}** requests **{lr['amount']}**")
                    with st.expander(f"Set terms and approve — {lr['member']}"):
                        st.caption(f"Member suggested: {lr.get('suggested_duration_months', 4)} months")
                        fee_pct = st.number_input("Processing fee (%)", value=LOAN_REQUEST_DEFAULT_FEE*100, min_value=0.0, max_value=100.0, step=0.5, key=f"fee_{idx}")
                        interest_pct = st.number_input("Interest (%)", value=LOAN_REQUEST_DEFAULT_INTEREST*100, min_value=0.0, max_value=100.0, step=0.5, key=f"interest_{idx}")
                        duration = st.number_input("Duration (months)", min_value=2, max_value=6, value=lr.get("suggested_duration_months", 4), key=f"duration_{idx}")
                        if st.button("Confirm approval", key=f"confirm_approve_{idx}"):
                            eligible, elig_message = check_loan_eligibility(lr["member"], lr["amount"], viewer=CURRENT_USER)
                            if eligible:
                                terms = calculate_loan_terms(lr["amount"], duration, interest_rate=interest_pct/100, processing_fee_rate=fee_pct/100)
                                st.session_state.loan_requests[idx]["status"] = "approved"
                                st.session_state.loan_requests[idx].update(terms)
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
                                st.success(f"✅ SIMULATED LOAN PAYMENT — {elig_message} Monthly payment: {terms['monthly_payment']} over {duration} months.")
                                st.rerun()
                            else:
                                st.error(f"❌ {elig_message}")
                    if st.button("Reject", key=f"reject_{idx}"):
                        st.session_state.loan_requests[idx]["status"] = "rejected"
                        st.rerun()

        with tab_approved:
            approved = [lr for lr in st.session_state.loan_requests if lr["status"] == "approved"]
            if not approved:
                st.info("No approved loan requests yet.")
            else:
                for lr in approved:
                    lr = update_schedule_penalties(lr)
                    st.markdown(
                        f"**{lr['member']}** — Principal: {lr['amount']} | "
                        f"Interest: {lr['interest_rate']*100:.0f}% ({lr['interest_amount']}) | "
                        f"Fee: {lr['processing_fee_rate']*100:.0f}% ({lr['processing_fee']}) | "
                        f"Total repayable: {lr['total_repayable']} | Monthly: {lr['monthly_payment']}"
                    )
                    with st.expander(f"Payment schedule — {lr['member']}"):
                        st.dataframe(pd.DataFrame(lr["schedule"]), use_container_width=True, hide_index=True)
                    st.divider()

        with tab_rejected:
            rejected = [lr for lr in st.session_state.loan_requests if lr["status"] == "rejected"]
            if not rejected:
                st.info("No rejected loan requests.")
            else:
                st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)

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
            st.dataframe(display_df, use_container_width=True, hide_index=True)

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
        st.bar_chart(my_monthly_df.set_index("Month"), color="#E8A33D")

        st.subheader("Your loan requests")
        my_loans = [lr for lr in st.session_state.loan_requests if lr["member"] == CURRENT_USER]
        if not my_loans:
            st.info("No loan requests yet.")
        else:
            for lr in my_loans:
                st.write(f"**Amount:** {lr['amount']} — **Status:** {lr['status']}")
                if lr["status"] == "approved":
                    lr = update_schedule_penalties(lr)
                    st.write(
                        f"Interest: {lr['interest_rate']*100:.0f}% | Fee: {lr['processing_fee_rate']*100:.0f}% | "
                        f"Duration: {lr['duration_months']} months | Monthly: {lr['monthly_payment']} | Total: {lr['total_repayable']}"
                    )
                    st.dataframe(pd.DataFrame(lr["schedule"]), use_container_width=True, hide_index=True)
                st.divider()

        st.subheader("Your scheduled reminders")
        my_reminders = [r for r in st.session_state.scheduled_reminders if r["member"] == CURRENT_USER]
        if not my_reminders:
            st.info("No reminders scheduled yet.")
        else:
            st.dataframe(pd.DataFrame(my_reminders), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3: Benchmark Data
# ============================================================
elif page == "Benchmark Data":
    render_header("Benchmark Data")
    st.caption("⚠️ Storage here is temporary — tied to this browser session only. Download files before closing the tab or redeploying the app, or they will be lost.")

    if not st.session_state.benchmark_samples:
        st.info("No benchmark samples saved yet. Save recordings from the Record & Query page.")
    else:
        for i, sample in enumerate(st.session_state.benchmark_samples):
            idx = sample["index"]
            st.markdown(f"**testcase{idx}**")

            col1, col2 = st.columns([4, 1])
            with col1:
                st.audio(sample["audio_bytes"])
                new_gt = st.text_area(f"Ground truth — testcase{idx}", value=sample["ground_truth"], key=f"bench_gt_{i}", label_visibility="collapsed")
            with col2:
                if st.button("Update", key=f"bench_update_{i}"):
                    st.session_state.benchmark_samples[i]["ground_truth"] = new_gt
                    st.success("Updated.")
                if st.button("Delete", key=f"bench_delete_{i}"):
                    st.session_state.benchmark_samples.pop(i)
                    st.rerun()

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                f"Download testcase{idx}.wav",
                data=sample["audio_bytes"],
                file_name=f"testcase{idx}.wav",
                mime="audio/wav",
                key=f"dl_wav_{i}"
            )
            dl_col2.download_button(
                f"Download testcase{idx}.txt",
                data=sample["ground_truth"],
                file_name=f"testcase{idx}.txt",
                mime="text/plain",
                key=f"dl_txt_{i}"
            )
            st.divider()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for sample in st.session_state.benchmark_samples:
                idx = sample["index"]
                zf.writestr(f"testcase{idx}.wav", sample["audio_bytes"])
                zf.writestr(f"testcase{idx}.txt", sample["ground_truth"])
        zip_buffer.seek(0)

        st.download_button(
            "⬇️ Download all as .zip",
            data=zip_buffer,
            file_name="benchmark_samples.zip",
            mime="application/zip"
        )