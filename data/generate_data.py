"""
Generates a labeled dataset of telecom IVR commands for intent classification.
Uses template + slot-filling + phrasing variation (including light Pidgin/casual
phrasing) to build a reasonably diverse dataset without needing to scrape data.
"""
import csv
import random

random.seed(42)

# ---- Slot values used to fill templates ----
NAMES = ["Chidi", "my mum", "my broda", "Ngozi", "the office", "Emeka", "Blessing"]
AMOUNTS = ["100 naira", "200 naira", "500 naira", "1000 naira", "50 naira"]
DATA_PLANS = ["1GB", "2GB", "500MB", "5GB weekly", "10GB monthly"]

INTENTS = {
    "check_balance": [
        "check my balance",
        "what is my airtime balance",
        "how much money do I have on my line",
        "balance enquiry",
        "I want to know my balance",
        "abeg check my balance for me",
        "wetin be my balance now",
        "show me my account balance",
        "tell me how much credit I have",
        "how much airtime remain",
    ],
    "recharge_airtime": [
        "recharge my line with {amount}",
        "top up my airtime with {amount}",
        "load {amount} credit on my phone",
        "I want to recharge {amount}",
        "add {amount} airtime abeg",
        "please recharge my number with {amount}",
        "buy me {amount} recharge card",
        "fund my line {amount}",
    ],
    "check_data_balance": [
        "check my data balance",
        "how much data do I have left",
        "data balance enquiry",
        "wetin remain for my data",
        "show my remaining data",
        "how much MB or GB I get now",
        "I want to see my internet data balance",
        "abeg how much data I still get",
        "tell me my remaining internet data",
        "what is my current data usage balance",
    ],
    "buy_data_bundle": [
        "buy {plan} data bundle",
        "I want to subscribe to {plan} plan",
        "activate {plan} internet plan for me",
        "give me {plan} data",
        "abeg add {plan} data on my line",
        "purchase {plan} bundle",
        "load {plan} data for me now",
    ],
    "report_network_issue": [
        "my network is not working",
        "I am not getting network in my area",
        "network dey bad for my side",
        "I can't make calls, network is poor",
        "report network problem",
        "my internet is very slow",
        "no signal for my phone since morning",
        "network keeps dropping calls",
        "I dey experience network issue for my location",
        "why my network so bad since yesterday",
        "there is no network coverage where I am",
        "my calls keep disconnecting",
        "internet on my phone is not loading anything",
    ],
    "speak_to_agent": [
        "connect me to a customer care agent",
        "I want to speak to a human agent",
        "transfer me to customer service",
        "put me through to an agent",
        "I need to talk to someone about my account",
        "abeg connect me to person wey fit help me",
        "escalate this to a live agent",
        "I want to talk to a real person now",
        "please put me on hold for an agent",
        "I no wan talk to robot, give me person",
    ],
    "block_sim": [
        "block my SIM card, I lost my phone",
        "please deactivate my line, it was stolen",
        "I want to block this number urgently",
        "my phone don loss, block the SIM abeg",
        "suspend my line immediately",
        "block this number, someone stole my phone",
        "abeg block my SIM sharp sharp, dem steal my phone",
        "deactivate this number now, it's not safe",
        "I need to block my line before someone uses it",
        "my phone was stolen, please block it now",
    ],
    "unblock_sim": [
        "unblock my SIM card",
        "reactivate my line please",
        "I want to restore my blocked number",
        "unblock this number for me",
        "my line was blocked by mistake, please unblock it",
        "abeg unblock my line, na mistake dem block am",
        "please restore my SIM, it was blocked wrongly",
        "I need my number reactivated",
        "can you unblock this line for me",
    ],
    "change_plan": [
        "change my current plan",
        "I want to switch to a different tariff plan",
        "move me to the cheaper plan",
        "upgrade my plan to the premium one",
        "downgrade my current subscription plan",
        "abeg change my plan to the student plan",
        "I want to migrate to a new tariff",
        "switch me from this plan to another one",
        "help me change my subscription plan",
        "I no like this plan again, change am for me",
    ],
    "check_offers": [
        "what offers do you have for me",
        "any promo available now",
        "show me current data promotions",
        "wetin be the latest offer",
        "are there any discounts on data bundles",
        "tell me about ongoing promos",
        "abeg any offer dey now",
        "what discounts do you currently have",
        "is there any special package available",
        "list out your current promotions for me",
    ],
    "activate_roaming": [
        "activate international roaming on my line",
        "I am travelling, please enable roaming",
        "turn on roaming service for me",
        "I need roaming activated before my trip",
        "enable my line for use abroad",
        "abeg activate roaming, I dey travel soon",
        "I want to use my line when I travel out of the country",
        "can you turn on international roaming for me",
        "please set up roaming before my flight",
    ],
    "report_fraud": [
        "I want to report a fraudulent transaction",
        "someone used my line to scam people",
        "report suspicious activity on my account",
        "I received a fraud SMS pretending to be from you",
        "I want to flag unauthorized charges on my line",
        "somebody don use my number scam person",
        "abeg I wan report one fraud transaction",
        "there is an unauthorized deduction on my account",
        "I think my account has been compromised",
    ],
    "general_complaint": [
        "I want to file a complaint",
        "I am not happy with your service",
        "this network is disappointing me",
        "I have an issue I want to complain about",
        "make I lodge complaint about una service",
        "your customer service is poor, I want to complain",
        "I get complaint about how una dey handle customers",
        "I am dissatisfied with the service provided",
        "I want to raise an official complaint",
    ],
    "call_contact": [
        "call {name}",
        "please dial {name}",
        "phone {name} for me",
        "I want to call {name}",
        "ring {name}",
    ],
    "greeting": [
        "hello",
        "good morning",
        "hi there",
        "good afternoon",
        "how far",
        "how you dey",
    ],
    "goodbye": [
        "thank you, goodbye",
        "that's all, bye",
        "ok thank you",
        "no more questions, bye bye",
        "alright thanks a lot",
    ],
    "thank_you": [
        "thank you very much",
        "thanks a lot",
        "I appreciate your help",
        "thank you, that helped me well",
        "e do, thank you",
        "abeg thank you o, God bless",
        "I'm grateful for your help",
    ],
    "yes": [
        "yes",
        "yes that's correct",
        "that's right",
        "abeg yes na",
        "correct",
        "e correct",
        "yes please go ahead",
    ],
    "no": [
        "no",
        "that's not correct",
        "no thank you",
        "abeg no",
        "e no correct",
        "not that one",
        "no please",
    ],
    "cancel": [
        "cancel that",
        "please cancel this request",
        "I don't want to continue, cancel it",
        "abeg cancel am",
        "stop the process",
        "never mind, cancel",
        "I want to cancel this",
    ],
    "repeat": [
        "can you repeat that",
        "I didn't hear you, say it again",
        "please repeat",
        "abeg repeat wetin you talk",
        "come again",
        "I no hear well, repeat am",
        "say that one more time",
    ],
    "are_you_a_bot": [
        "am I talking to a real person",
        "are you a robot",
        "is this a human or a machine",
        "you be person or na machine",
        "are you an actual agent or AI",
        "na human dey answer me or na bot",
    ],
}


def fill_template(template: str) -> str:
    text = template
    if "{amount}" in text:
        text = text.replace("{amount}", random.choice(AMOUNTS))
    if "{plan}" in text:
        text = text.replace("{plan}", random.choice(DATA_PLANS))
    if "{name}" in text:
        text = text.replace("{name}", random.choice(NAMES))
    return text


def generate_dataset(samples_per_intent: int = 40):
    rows = []
    for intent, templates in INTENTS.items():
        for _ in range(samples_per_intent):
            template = random.choice(templates)
            text = fill_template(template)
            rows.append({"text": text, "intent": intent})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate_dataset(samples_per_intent=40)
    out_path = "/home/claude/voice_command_classifier/data/intents_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "intent"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} examples across {len(INTENTS)} intents -> {out_path}")
