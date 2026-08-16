from flask import Flask, render_template, request, Response
import json
import ollama

app = Flask(__name__)

# College data
with open("college_data.json", "r", encoding="utf-8") as file:
    college_data = json.load(file)


def make_context(data, parent=""):
    result = []

    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{parent} {key}".strip()

            if isinstance(value, (dict, list)):
                result.append(make_context(value, name))
            else:
                result.append(f"{name}: {value}")

    elif isinstance(data, list):
        for item in data:
            result.append(make_context(item, parent))

    return "\n".join(result)


college_context = make_context(college_data)


SYSTEM_PROMPT = """
You are StudentBot, a helpful AI assistant for students.

Understand English, Hindi and Hinglish.

You can answer:
College questions, Programming, Python, DSA, AI,
Machine Learning, DBMS, Mathematics, Science,
General Knowledge, Career, Exams and everyday questions.

RULES:

1. Answer directly.
2. Keep simple questions short.
3. Explain educational questions clearly.
4. Reply in the student's language when possible.
5. Never repeat the question.
6. Never introduce yourself unnecessarily.
7. For GNIOT questions, ONLY use the supplied GNIOT information.
8. Never invent GNIOT information.
9. If GNIOT information is not available, clearly say so.
"""


COLLEGE_KEYWORDS = [
    "gniot",
    "gni",
    "college",
    "library",
    "hostel",
    "attendance",
    "fee",
    "fees",
    "department",
    "faculty",
    "teacher",
    "admission",
    "campus",
    "canteen",
    "transport",
    "address",
    "location",
    "located",
    "contact",
    "phone",
    "exam",
    "scholarship",
    "laboratory",
    "lab",
    "course",
    "branch",
    "hod",
    "principal"
]


def is_college_question(message):
    message = message.lower()

    return any(
        keyword in message
        for keyword in COLLEGE_KEYWORDS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data:
        return Response(
            "Please type a question.",
            mimetype="text/plain"
        )

    user_message = data.get(
        "message",
        ""
    ).strip()

    if not user_message:
        return Response(
            "Please type a question.",
            mimetype="text/plain"
        )


    # College question
    if is_college_question(user_message):

        prompt = f"""
{SYSTEM_PROMPT}

GNIOT OFFICIAL INFORMATION:

{college_context}

STUDENT QUESTION:

{user_message}

Answer using ONLY the GNIOT information above.
Do not guess.
"""


    # General question
    else:

        prompt = f"""
{SYSTEM_PROMPT}

STUDENT QUESTION:

{user_message}

Answer directly and concisely.
"""


    try:

        def generate():

            stream = ollama.chat(
                model="llama3.2:3b",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                stream=True,

                keep_alive="10m",

                options={
                    "temperature": 0.2,
                    "num_predict": 128
                }
            )

            for chunk in stream:

                text = chunk.get(
                    "message",
                    {}
                ).get(
                    "content",
                    ""
                )

                if text:
                    yield text


        return Response(
            generate(),
            mimetype="text/plain"
        )


    except Exception as error:

        print(
            "Ollama Error:",
            error
        )

        return Response(
            "Sorry, I could not connect to the local AI.",
            mimetype="text/plain"
        )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )