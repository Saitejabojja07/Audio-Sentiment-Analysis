from fpdf import FPDF

def format_conversations(transcript):
    conversations = []
    speaker_prev = None
    current_conversation = ""

    for word_data in transcript["results"]["channels"][0]["alternatives"][0]["words"]:
        word = word_data["word"]
        speaker = word_data["speaker"]

        if speaker_prev is None:
            speaker_prev = speaker
            current_conversation += f"[Speaker:{speaker}] {word} "
        elif speaker_prev == speaker:
            current_conversation += word + " "
        else:
            conversations.append(current_conversation.strip())
            current_conversation = f"[Speaker:{speaker}] {word} "
            speaker_prev = speaker

    # Add the last conversation
    conversations.append(current_conversation.strip())

    return conversations

def save_conversations_to_pdf(conversations, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for conversation in conversations:
        pdf.cell(200, 10, txt=conversation + "\n", ln=True, align="L")

    pdf.output(filename)