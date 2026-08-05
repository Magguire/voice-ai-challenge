import streamlit as st
import requests
import json
from groq import Groq
import re
import pandas as pd
import torch

# --- Load keys from Streamlit secrets ---
SAHARA_API_KEY = st.secrets["SAHARA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_API_KEY = st.secrets.get("HF_API_KEY") # new secret, free HF account + free API token


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

CURRENT_USER = "Wendo"
IS_ADMIN = True

# --- Mock chama data (clearly illustrative, not real transaction history) ---
MOCK_MEMBERS = {
    "Wendo":  {"ytd_contributed": 24000, "this_month_paid": True,  "last_paid": "2026-08-01", "owed": 0,     "loan_eligible": 20000, "payout_position": 3},
    "Grace":  {"ytd_contributed": 22000, "this_month_paid": True,  "last_paid": "2026-08-02", "owed": 0,     "loan_eligible": 18000, "payout_position": 1},
    "John":   {"ytd_contributed": 18000, "this_month_paid": False, "last_paid": "2026-06-28", "owed": 4000,  "loan_eligible": 12000, "payout_position": 2},
    "Amina":  {"ytd_contributed": 24000, "this_month_paid": True,  "last_paid": "2026-08-03", "owed": 0,     "loan_eligible": 20000, "payout_position": 4},
    "Peter":  {"ytd_contributed": 16000, "this_month_paid": False, "last_paid": "2026-07-15", "owed": 2000,  "loan_eligible": 10000, "payout_position": 5},
}

MOCK_MONTHLY_TOTALS = pd.DataFrame({
    "Month": ["Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    "Collected": [18000, 19500, 21000, 20500, 22000, 21000]
})


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



def transcribe_whisper_hf(audio_path):
    api_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-small"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    with open(audio_path, "rb") as f:
        data = f.read()
    response = requests.post(api_url, headers=headers, data=data, timeout=60)
    if response.status_code != 200:
        raise Exception(f"HF Whisper API error {response.status_code}: {response.text[:300]}")
    return response.json().get("text", "")

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
    prompt = f"""Classify this chama voice message as either "statement" (reporting an action, like a deposit or payment)
or "question" (asking for information, like account status, balance, loan eligibility, or payout timing).

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


def extract_chama_action(transcript):
    if not transcript or len(transcript.strip()) < 5:
        return json.dumps({
            "reasoning": "Transcript empty or too short to contain meaningful speech.",
            "speaker_action": "other", "amount": None,
            "referenced_member": "none", "referenced_member_context": "none", "action_type": "other"
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
    gt_words = set(ground_truth.lower().split())
    model_words = set(model_output.lower().split())
    if not gt_words:
        return 0.0
    correct = len(gt_words & model_words)
    return round((correct / len(gt_words)) * 100, 1)


# --- Page config ---
st.set_page_config(page_title="Habahub", page_icon="🎙️", layout="wide")

if "ledger" not in st.session_state:
    st.session_state.ledger = []
    
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

# --- Sidebar ---
st.sidebar.title("🎙️ Habahub")
st.sidebar.markdown(f"**{CURRENT_USER}** {'(Admin)' if IS_ADMIN else ''}")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Dashboard", "Record & Query", "Benchmark"])

def render_header(title):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(title)
    with col2:
        st.write("")  # vertical spacer to align with title
        with st.popover(f"👤 {CURRENT_USER}"):
            st.write(f"**{CURRENT_USER}**")
            st.caption("Admin" if IS_ADMIN else "Member")
            st.divider()
            st.button("Log out", disabled=True, help="Demo only — not functional")

speaking_as = CURRENT_USER

# ============================================================
# PAGE 1: Record & Query
# ============================================================
if page == "Record & Query":
    render_header("Record & Query")
    st.write("Speak a contribution, payment note, update — or ask a question about your account — in English, Swahili, or both.")
    audio = st.audio_input("Record your message")


    if audio is not None:
        with open("temp_input.wav", "wb") as f:
            f.write(audio.getvalue())

        with st.spinner("Transcribing..."):
            transcript = transcribe_sahara("temp_input.wav")
        st.write("**Transcript:**", transcript)

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

        if intent == "question":
            with st.spinner("Checking records..."):
                answer = answer_member_query(transcript, speaking_as)
            st.write("**Answer:**", answer)
            with st.spinner("Generating spoken answer..."):
                audio_url = generate_tts(answer)
            st.audio(audio_url)

        else:
            with st.spinner("Extracting details..."):
                extracted_raw = extract_chama_action(transcript)
                extracted = json.loads(extracted_raw) if isinstance(extracted_raw, str) else extracted_raw
                st.json(extracted)
                extracted = validate_extraction(transcript, extracted)

            if "_flag" in extracted:
                st.warning(f"⚠️ {extracted['_flag']}")
            if "_flag_invalid_speaker_action" in extracted:
                st.warning(f"⚠️ Unrecognized action type: '{extracted['_flag_invalid_speaker_action']}' — defaulted to 'other'.")

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
                final_entry["speaker"] = speaking_as
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
    st.subheader("Session Ledger")
    st.dataframe(st.session_state.ledger, use_container_width=True)

# ============================================================
# PAGE 2: Dashboard
# ============================================================
elif page == "Dashboard":
    render_header("Dashboard")
    st.caption("⚠️ Illustrative demo data — not live transaction history.")

    total_ytd = sum(m["ytd_contributed"] for m in MOCK_MEMBERS.values())
    this_month = MOCK_MONTHLY_TOTALS["Collected"].iloc[-1]
    last_month = MOCK_MONTHLY_TOTALS["Collected"].iloc[-2]
    paid_this_month = sum(1 for m in MOCK_MEMBERS.values() if m["this_month_paid"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total collected (YTD)", f"KES {total_ytd:,}")
    col2.metric("Collected this month", f"KES {this_month:,}", delta=f"{this_month - last_month:+,}")
    col3.metric("Members paid this month", f"{paid_this_month}/{len(MOCK_MEMBERS)}")
    col4.metric("Total owed to group", f"KES {sum(m['owed'] for m in MOCK_MEMBERS.values()):,}")

    st.subheader("Monthly collections")
    st.bar_chart(MOCK_MONTHLY_TOTALS.set_index("Month"))

    st.subheader("Member contributions")
    member_df = pd.DataFrame([
        {"Member": name, "YTD Contributed": v["ytd_contributed"], "Paid this month": "✅" if v["this_month_paid"] else "❌",
         "Last paid": v["last_paid"], "Owed": v["owed"], "Loan eligible": v["loan_eligible"]}
        for name, v in MOCK_MEMBERS.items()
    ])
    st.dataframe(member_df, use_container_width=True, hide_index=True)

    st.subheader("Payout rotation order")
    rotation_df = pd.DataFrame([
        {"Position": v["payout_position"], "Member": name}
        for name, v in MOCK_MEMBERS.items()
    ]).sort_values("Position")
    st.dataframe(rotation_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: Benchmark
# ============================================================
elif page == "Benchmark":
    render_header("Benchmark")
    st.caption("Live results from Record & Query, benchmarked against editable ground truth.")

    if not st.session_state.benchmark_results:
        st.info("No recordings yet — use Record & Query to generate benchmark data.")
    else:
        for i, entry in enumerate(st.session_state.benchmark_results):
            st.markdown(f"**Recording {i+1}**")
            edited_gt = st.text_area(f"Ground truth (edit if incorrect) — Recording {i+1}", value=entry["ground_truth"], key=f"gt_{i}")
            st.session_state.benchmark_results[i]["ground_truth"] = edited_gt

            sahara_acc = word_accuracy(edited_gt, entry["sahara"])
            whisper_acc = word_accuracy(edited_gt, entry["whisper"])
            mms_acc = word_accuracy(edited_gt, entry["mms"])

            comp_df = pd.DataFrame([
                {"Model": "Sahara", "Transcript": entry["sahara"], "Word Accuracy": f"{sahara_acc}%"},
                {"Model": "Whisper", "Transcript": entry["whisper"], "Word Accuracy": f"{whisper_acc}%"},
                {"Model": "MMS", "Transcript": entry["mms"], "Word Accuracy": f"{mms_acc}%"},
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.divider()




#     render_header("Benchmark")
#     st.caption("Comparison across 3 speech models on real code-switched test audio, recorded and evaluated during development.")

#     BENCHMARK_DATA = [
#         {
#             "Test Case": "testcase1",
#             "Ground Truth": "Nimeweka 2000 leo. That's my deposit ya loan repayment, na next Friday ni zamu ya Grace.",
#             "Sahara": "Nimeweka 2,000 leo. That's my deposit Yelon repayment na next Friday ni za Grace.",
#             "Whisper": "Nimeweka 2000 lewa. That's my deposit, a loan repayment na next Friday nizamu ya grace.",
#             "MMS": "nimeweka 200leo dhat's may dipositi ya lon repayment na next fridey ni zamu ya greece"
#         },
#         {
#             "Test Case": "testcase2",
#             "Ground Truth": "Nimechelewa this month, but nitatuma the full amount next Friday, hiyo iko sawa?",
#             "Sahara": "Nimechelewa mwezi huu lakini nitatuma kiasi chote Ijumaa ijayo. Hiyo iko sawa.",
#             "Whisper": "Imechelewa this month, but nita tumade full amount next Friday. Yoi kusawa.",
#             "MMS": "nimechelewa this month bat nitatuma the fuu la mount next fridey yoo iko sawa"
#         },
#         {
#             "Test Case": "testcase3",
#             "Ground Truth": "Leo tulipata three new members kwenye group, so tuta-adjust rotation kidogo, utapokea payout yako mwezi ujao instead of this month.",
#             "Sahara": "Leo tulipata members 3 wapya kwenye group, so tutafanya rotation kidogo, utapokea payout yako mwezi ujao instead of this month.",
#             "Whisper": "Leotuli patatrinu membas kweni grub, sututad adjust rotation kidogo, utapokia peia utiako muizi ujao instead of this month.",
#             "MMS": "leo tulipata 3 new members kwenye group su tutaadjust tratition kidogo utapokea peauti yako mwezi ujao instead of this month"
#         }
#     ]

#     st.dataframe(pd.DataFrame(BENCHMARK_DATA), use_container_width=True, hide_index=True)

#     st.subheader("Key findings")
#     st.markdown("""
#     - **Sahara** preserves code-switched sentence structure and grammar most reliably, but shows run-to-run instability on financial-domain terms specifically (e.g. "loan repayment" → "Yelon repayment" in one run, correct in another).
#     - **Whisper** correctly transcribed the one financial term Sahara mangled ("loan repayment," testcase1), but broke down more severely on basic word boundaries and grammar in longer, faster code-switched speech.
#     - **MMS** (locked to Swahili via `target_lang="swh"`) produced phonetic, uncapitalized transcriptions that treat English words as if sounding them out in Swahili spelling — a distinct third failure mode, consistent with running a monolingual model on code-switched input.
#     - No model was reliably accurate across all three test cases; each had a different, characteristic failure pattern rather than a simple "better/worse" ranking.
#     """)