import os
from agents.factory import build_vertex_agent

PROJECT_ID  = os.getenv("PROJECT_ID",  "evidence-synthesis-gemma")
LOCATION    = os.getenv("LOCATION",    "us-central1")
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "mg-endpoint-0f4af99c-9fde-4bd7-8966-8d155e5c91f9")
DEDICATED_DNS = "mg-endpoint-0f4af99c-9fde-4bd7-8966-8d155e5c91f9.us-central1-1050333749476.prediction.vertexai.goog"

agent = build_vertex_agent(
    project_id=PROJECT_ID,
    location=LOCATION,
    endpoint_id=ENDPOINT_ID,
    dedicated_dns_or_predict_url=DEDICATED_DNS,
    temperature=0.2,
    max_tokens=1200,
)

agent.set_system("You are concise and helpful.")

print("Type your message. Commands: /reset, /system <text>, /exit")
while True:
    try:
        user = input("you> ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            agent.reset()
            print("(history cleared)")
            continue
        if user.startswith("/system "):
            agent.set_system(user[len("/system "):].strip())
            print("(system prompt updated)")
            continue

        reply = agent.say(user)
        print(f"assistant> {reply}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        break
