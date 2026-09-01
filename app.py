import gradio as gr

from modules.tokenization import (
    spacy_tokenization,
    modern_tokenization,
)

from modules.preprocessing import (
    remove_stopwords,
    lemmatize
)

from modules.bow import (
    bag_of_words
)

from modules.ner import (
    named_entity_recognition
)


DEFAULT_TEXT = """
Barack Obama visited New York in 2020.

He gave an interesting speech about technology.
"""


# Custom CSS for a slightly different UI
custom_css = """
body {
    background: linear-gradient(135deg, #f5f7fa, #e8eef7);
}

.main-title {
    text-align: center;
    color: #243b53;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #627d98;
    font-size: 16px;
    margin-bottom: 20px;
}

.input-box {
    border: 2px solid #bcccdc;
    border-radius: 12px;
}

.tab-nav button {
    font-weight: 600;
}

.output-box {
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #829ab1;
    font-size: 13px;
    margin-top: 20px;
}
"""


with gr.Blocks(
    title="NLP Toolkit",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css
) as app:

    # Header
    gr.Markdown(
        """
        <div class="main-title">🧠 NLP Toolkit</div>
        <div class="subtitle">
            Explore Natural Language Processing techniques interactively
        </div>
        """
    )

    # Input section
    with gr.Group():
        gr.Markdown("### ✍️ Enter Text")

        text_input = gr.Textbox(
            label="Text for Analysis",
            placeholder="Type or paste your paragraph here...",
            value=DEFAULT_TEXT,
            lines=8,
            elem_classes="input-box"
        )

        gr.Markdown(
            """
            💡 **Tip:** Enter any sentence or paragraph and choose an
            NLP operation from the tabs below.
            """
        )

    # NLP Tabs
    with gr.Tabs():

        # ---------------- TOKENIZATION ----------------
        with gr.Tab("🔤 Tokenizer"):

            gr.Markdown(
                """
                ## 🔤 Word Tokenization

                Break your text into individual words, numbers and
                punctuation using **spaCy**.
                """
            )

            tokenize_button = gr.Button(
                "🚀 Tokenize Text",
                variant="primary"
            )

            token_output = gr.Dataframe(
                headers=["Token"],
                label="Tokenized Text",
                elem_classes="output-box"
            )

            tokenize_button.click(
                fn=spacy_tokenization,
                inputs=text_input,
                outputs=token_output
            )

        # ---------------- MODERN TOKENIZERS ----------------
        with gr.Tab("🤖 Transformer Tokens"):

            gr.Markdown(
                """
                ## 🤖 Transformer Tokenization

                See how modern NLP models split text into tokens.
                """
            )

            model_dropdown = gr.Dropdown(
                choices=[
                    "BERT",
                    "GPT-2",
                    "T5",
                    "DeepSeek"
                ],
                value="BERT",
                label="Choose a Model"
            )

            modern_button = gr.Button(
                "🔍 Generate Tokens",
                variant="primary"
            )

            modern_output = gr.Dataframe(
                headers=[
                    "Token Number",
                    "Token"
                ],
                label="Model Token Output",
                elem_classes="output-box"
            )

            modern_button.click(
                fn=modern_tokenization,
                inputs=[
                    text_input,
                    model_dropdown
                ],
                outputs=modern_output
            )

        # ---------------- STOP WORDS ----------------
        with gr.Tab("🚫 Stopword Filter"):

            gr.Markdown(
                """
                ## 🚫 Stopword Removal

                Remove common words that usually carry less information
                for many NLP applications.
                """
            )

            stop_button = gr.Button(
                "🧹 Clean Text",
                variant="primary"
            )

            cleaned_output = gr.Textbox(
                label="Cleaned Text",
                lines=6,
                elem_classes="output-box"
            )

            removed_output = gr.Textbox(
                label="Words Removed",
                lines=3,
                elem_classes="output-box"
            )

            stop_button.click(
                fn=remove_stopwords,
                inputs=text_input,
                outputs=[
                    cleaned_output,
                    removed_output
                ]
            )

        # ---------------- LEMMATIZATION ----------------
        with gr.Tab("🌱 Word Lemmas"):

            gr.Markdown(
                """
                ## 🌱 Lemmatization

                Convert words into their basic dictionary form.

                **Example:** running → run
                """
            )

            lemma_button = gr.Button(
                "🌿 Find Lemmas",
                variant="primary"
            )

            lemma_output = gr.Dataframe(
                headers=[
                    "Original Word",
                    "Lemma",
                    "POS"
                ],
                label="Lemmatization Details",
                elem_classes="output-box"
            )

            lemma_button.click(
                fn=lemmatize,
                inputs=text_input,
                outputs=lemma_output
            )

        # ---------------- BAG OF WORDS ----------------
        with gr.Tab("📊 Word Frequency"):

            gr.Markdown(
                """
                ## 📊 Bag of Words

                Analyze the text by counting how frequently each word
                appears.
                """
            )

            bow_button = gr.Button(
                "📈 Analyze Word Frequency",
                variant="primary"
            )

            bow_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Frequency"
                ],
                label="Word Frequency Table",
                elem_classes="output-box"
            )

            vocabulary_output = gr.Textbox(
                label="📚 Vocabulary",
                elem_classes="output-box"
            )

            bow_button.click(
                fn=bag_of_words,
                inputs=text_input,
                outputs=[
                    bow_output,
                    vocabulary_output
                ]
            )

        # ---------------- NER ----------------
        with gr.Tab("🏷️ Entity Finder"):

            gr.Markdown(
                """
                ## 🏷️ Named Entity Recognition

                Detect important entities such as people, organizations,
                locations, dates and monetary values.
                """
            )

            ner_button = gr.Button(
                "🎯 Detect Entities",
                variant="primary"
            )

            ner_output = gr.Dataframe(
                headers=[
                    "Entity",
                    "Label",
                    "Description"
                ],
                label="Detected Entities",
                elem_classes="output-box"
            )

            ner_button.click(
                fn=named_entity_recognition,
                inputs=text_input,
                outputs=ner_output
            )

    # Footer
    gr.Markdown(
        """
        <div class="footer">
            🧠 NLP Toolkit • Text Processing & Analysis Laboratory
        </div>
        """
    )


app.launch()
