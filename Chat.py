import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import time

warnings.filterwarnings("ignore")

# 🔐 Cargar variables desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY1")
print("KEY OK:", (api_key[:10] + "..." if api_key else None))


if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

# 🤖 Configuración del modelo OpenRouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY1"],
    model_name="mistralai/mistral-7b-instruct",
    temperature=0.7,
)

# 🗨️ Bucle de chat básico
print("💬 Chatbot Mistral vía OpenRouter (escribe 'salir' para terminar)\n")

Meta_prompt = "Actua como un especialista internacional en cortes de carne de res de todo tipo que usa dialectos de la costa caribe Colombiana para expresarse. Pregunta: "
chat_summary = ""
recent_msgs = []
MAX_PAIRS = 2
MAX_CHARS = 1200

def format_recent(recent):
    lines = []
    for role, txt in recent:
        tag = "User" if role == "user" else "Bot"
        lines.append(f"{tag}: {txt}")
    return "\n".join(lines) if lines else "(empty)"

def should_summarize(recent, limit=MAX_CHARS):
    return sum(len(t) for _, t in recent) > limit or len(recent) >= MAX_PAIRS * 2

def summarize_chat(llm, summary, recent):
    prompt = f"""
You are a summarizer. Combine the previous summary (S) with the recent dialogue (D)
into a NEW concise summary that keeps only facts, decisions, preferences, and open tasks.
Avoid filler and duplicates. Return ONLY the new summary.

S:
{summary or "(empty)"}

D:
{format_recent(recent)}
""".strip()
    result = llm.invoke([HumanMessage(content=prompt)])
    new_summary = getattr(result, "content", None) or (
        result.get("content") if isinstance(result, dict) else str(result)
    )
    return (new_summary or "").strip()
# --- end added ---

# (rename Meta_prompt -> persona to keep names in English)
persona = "Actua como un especialista internacional en cortes de carne de res de todo tipo que usa dialectos de la costa caribe Colombiana para expresarse. Pregunta: "


while True:
    user_input = input("👤 Tú: ")
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break

    try:
        # --- changed: build a compact prompt using summary + recent, not full Memo ---
        full_prompt = f"""
    {persona}

    Chat summary:
    {chat_summary or "(empty)"}

    Recent messages:
    {format_recent(recent_msgs)}

    User question:
    {user_input}
    """.strip()

        response = llm.invoke([HumanMessage(content=full_prompt)])

        try:
            # Igual que antes: AIMessage
            bot_text = response.content.strip()
            print(f"🤖 Bot: {bot_text}\n")
        except AttributeError:
            # Igual que antes: dict o lista
            if isinstance(response, dict) and "content" in response:
                bot_text = response["content"].strip()
                print(f"🤖 Bot: {bot_text}\n")
            elif isinstance(response, list) and len(response) > 0:
                bot_text = response[0].content.strip()
                print(f"🤖 Bot: {bot_text}\n")
            else:
                bot_text = str(response).strip()
                print(f"🤖 Bot: {bot_text}\n")

        # --- changed: update short-term window instead of concatenating Memo ---
        recent_msgs.append(("user", user_input))
        recent_msgs.append(("bot", bot_text))

        # Keep only last N pairs
        if len(recent_msgs) > MAX_PAIRS * 2:
            recent_msgs = recent_msgs[-MAX_PAIRS * 2:]

        # Summarize when needed, then keep only last exchange for freshness
        if should_summarize(recent_msgs):
            chat_summary = summarize_chat(llm, chat_summary, recent_msgs)
            recent_msgs = recent_msgs[-2:]  # keep only last (user, bot)

    except Exception as e:
        print(f"❌ Error: {e}\n")
