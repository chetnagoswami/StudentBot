from flask import Flask, render_template, request, Response
import json
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)


# =========================================================
# SETTINGS
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = None

if HF_TOKEN:
    client = InferenceClient(
        api_key=HF_TOKEN,
        provider="auto"
    )


# =========================================================
# LOAD COLLEGE DATA
# =========================================================

try:

    with open(
        "college_data.json",
        "r",
        encoding="utf-8"
    ) as file:

        college_data = json.load(file)

except Exception as error:

    print("College data loading error:", error)

    college_data = {}


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are StudentBot, a helpful AI assistant for students.

You can answer questions about:

- GNIOT
- College information
- Programming
- Python
- Java
- C++
- DSA
- AI
- Machine Learning
- Data Science
- DBMS
- Computer Science
- Mathematics
- Science
- Career
- Exams
- Study questions
- General questions

IMPORTANT RULES:

1. Answer directly.
2. Keep simple questions short.
3. Explain educational questions clearly.
4. Understand English, Hindi and Hinglish.
5. Reply in the same language as the student whenever possible.
6. Do not repeat the student's question.
7. Do not unnecessarily introduce yourself.
8. Never invent official college information.
9. If official college information is not available, clearly say that the information is not available.
10. Be helpful and student-friendly.
"""


# =========================================================
# FLATTEN COLLEGE JSON
# =========================================================

def flatten_data(data, path=""):

    result = []

    if isinstance(data, dict):

        for key, value in data.items():

            current_path = (
                f"{path} {key}".strip()
            )

            if isinstance(
                value,
                (dict, list)
            ):

                result.extend(
                    flatten_data(
                        value,
                        current_path
                    )
                )

            else:

                result.append({
                    "key": current_path.lower(),
                    "value": str(value)
                })

    elif isinstance(data, list):

        for item in data:

            if isinstance(
                item,
                (dict, list)
            ):

                result.extend(
                    flatten_data(
                        item,
                        path
                    )
                )

            else:

                result.append({
                    "key": path.lower(),
                    "value": str(item)
                })

    return result


flat_data = flatten_data(
    college_data
)


# =========================================================
# COLLEGE QUESTION DETECTION
# =========================================================

COLLEGE_KEYWORDS = [

    "gniot",
    "gni",
    "college",
    "library",
    "hostel",
    "address",
    "location",
    "located",
    "branch",
    "branches",
    "btech",
    "course",
    "courses",
    "fees",
    "fee",
    "admission",
    "faculty",
    "department",
    "transport",
    "canteen",
    "laboratory",
    "lab",
    "scholarship",
    "attendance",
    "exam",
    "examination",
    "result",
    "event",
    "events",
    "contact"

]


def is_college_question(question):

    question = question.lower()

    for keyword in COLLEGE_KEYWORDS:

        if keyword in question:

            return True

    return False


# =========================================================
# SEARCH COLLEGE DATA
# =========================================================

def search_college_data(question):

    question = question.lower()

    results = []

    # -----------------------------------------
    # Address / Location
    # -----------------------------------------

    if any(
        word in question
        for word in [
            "address",
            "location",
            "located",
            "kahan",
            "where"
        ]
    ):

        for item in flat_data:

            if any(
                word in item["key"]
                for word in [
                    "address",
                    "location",
                    "located"
                ]
            ):

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Library
    # -----------------------------------------

    elif "library" in question:

        for item in flat_data:

            if "library" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Hostel
    # -----------------------------------------

    elif "hostel" in question:

        for item in flat_data:

            if "hostel" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Branch / BTech
    # -----------------------------------------

    elif (
        "branch" in question
        or "branches" in question
        or "btech" in question
    ):

        for item in flat_data:

            if any(
                word in item["key"]
                for word in [
                    "branch",
                    "btech"
                ]
            ):

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Fees
    # -----------------------------------------

    elif (
        "fee" in question
        or "fees" in question
    ):

        for item in flat_data:

            if "fee" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Admission
    # -----------------------------------------

    elif "admission" in question:

        for item in flat_data:

            if "admission" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Attendance
    # -----------------------------------------

    elif "attendance" in question:

        for item in flat_data:

            if "attendance" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Exam / Result
    # -----------------------------------------

    elif any(
        word in question
        for word in [
            "exam",
            "examination",
            "result"
        ]
    ):

        for item in flat_data:

            if any(
                word in item["key"]
                for word in [
                    "exam",
                    "examination",
                    "result"
                ]
            ):

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Scholarship
    # -----------------------------------------

    elif "scholarship" in question:

        for item in flat_data:

            if "scholarship" in item["key"]:

                results.append(
                    item["value"]
                )


    # -----------------------------------------
    # Contact
    # -----------------------------------------

    elif any(
        word in question
        for word in [
            "contact",
            "phone",
            "number",
            "email"
        ]
    ):

        for item in flat_data:

            if "contact" in item["key"]:

                results.append(
                    item["value"]
                )


    # Remove duplicates

    unique_results = []

    for result in results:

        if result not in unique_results:

            unique_results.append(
                result
            )

    return unique_results


# =========================================================
# CREATE COLLEGE ANSWER
# =========================================================

def create_college_answer(
    question,
    results
):

    if not results:

        return None


    question = question.lower()


    # Address

    if any(
        word in question
        for word in [
            "address",
            "location",
            "located",
            "kahan",
            "where"
        ]
    ):

        return (
            "GNIOT location information:\n\n"
            +
            "\n".join(
                "• " + result
                for result in results
            )
        )


    # Library

    if "library" in question:

        return (
            "GNIOT library information:\n\n"
            +
            "\n".join(
                "• " + result
                for result in results
            )
        )


    # Hostel

    if "hostel" in question:

        return (
            "GNIOT hostel information:\n\n"
            +
            "\n".join(
                "• " + result
                for result in results
            )
        )


    # Branches

    if (
        "branch" in question
        or "branches" in question
        or "btech" in question
    ):

        return (
            "GNIOT B.Tech information:\n\n"
            +
            "\n".join(
                "• " + result
                for result in results
            )
        )


    # Default

    return (
        "\n".join(
            "• " + result
            for result in results
        )
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# CHAT API
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
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


    # =====================================================
    # FAST COLLEGE ANSWER
    # =====================================================

    if is_college_question(
        user_message
    ):

        college_results = search_college_data(
            user_message
        )


        if college_results:

            direct_answer = create_college_answer(
                user_message,
                college_results
            )


            if direct_answer:

                return Response(
                    direct_answer,
                    mimetype="text/plain"
                )


    # =====================================================
    # HUGGING FACE AI
    # =====================================================

    if client is None:

        return Response(
            "AI service is not configured yet. Please add HF_TOKEN in the server environment.",
            mimetype="text/plain"
        )


    try:

        def generate():

            stream = client.chat.completions.create(

                model=HF_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],

                stream=True,

                max_tokens=256,

                temperature=0.2
            )


            for chunk in stream:

                try:

                    text = (
                        chunk
                        .choices[0]
                        .delta
                        .content
                    )

                    if text:

                        yield text

                except Exception:

                    continue


        return Response(
            generate(),
            mimetype="text/plain"
        )


    except Exception as error:

        print(
            "Hugging Face Error:",
            error
        )

        return Response(
            "Sorry, AI service is temporarily unavailable. Please try again.",
            mimetype="text/plain"
        )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )