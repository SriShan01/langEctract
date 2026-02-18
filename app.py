from flask import Flask, render_template, request, jsonify
import langextract as lx
import textwrap
import os

app = Flask(__name__)

# 1. Define the prompt and extraction rules
# prompt = textwrap.dedent("""\
#     Extract characters, emotions, and relationships in order of appearance.
#     Use exact text for extractions. Do not paraphrase or overlap entities.
#     Provide meaningful attributes for each entity to add context.""")

prompt = textwrap.dedent("""\
Extract the following entities in order of appearance:

- patient
- doctor
- lab_test
- test_result
- diagnosis
- prescription

Use exact text for extractions. Do not paraphrase or overlap entities.

For each entity:
- patient: include age and gender if mentioned
- doctor: include specialty if mentioned
- lab_test: include test type
- test_result: include value and unit if available
- prescription: include drug name, dosage, frequency, and duration
""")


# 2. Provide a high-quality example to guide the model
# examples = [
#     lx.data.ExampleData(
#         text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
#         extractions=[
#             lx.data.Extraction(
#                 extraction_class="character",
#                 extraction_text="ROMEO",
#                 attributes={"emotional_state": "wonder"}
#             ),
#             lx.data.Extraction(
#                 extraction_class="emotion",
#                 extraction_text="But soft!",
#                 attributes={"feeling": "gentle awe"}
#             ),
#             lx.data.Extraction(
#                 extraction_class="relationship",
#                 extraction_text="Juliet is the sun",
#                 attributes={"type": "metaphor"}
#             ),
#         ]
#     )
# ]

examples = [
    lx.data.ExampleData(
        text=(
            "Patient John Smith, a 52-year-old male, visited Dr. Sarah Lee, "
            "a cardiologist. His blood pressure was 150/95 mmHg. "
            "He was diagnosed with hypertension and prescribed "
            "Lisinopril 10 mg once daily for 30 days."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="patient",
                extraction_text="John Smith",
                attributes={
                    "age": "52",
                    "gender": "male"
                }
            ),
            lx.data.Extraction(
                extraction_class="doctor",
                extraction_text="Dr. Sarah Lee",
                attributes={
                    "specialty": "cardiologist"
                }
            ),
            lx.data.Extraction(
                extraction_class="lab_test",
                extraction_text="blood pressure",
                attributes={
                    "test_type": "vital sign"
                }
            ),
            lx.data.Extraction(
                extraction_class="test_result",
                extraction_text="150/95 mmHg",
                attributes={
                    "value": "150/95",
                    "unit": "mmHg"
                }
            ),
            lx.data.Extraction(
                extraction_class="diagnosis",
                extraction_text="hypertension",
                attributes={}
            ),
            lx.data.Extraction(
                extraction_class="prescription",
                extraction_text="Lisinopril 10 mg once daily for 30 days",
                attributes={
                    "drug": "Lisinopril",
                    "dosage": "10 mg",
                    "frequency": "once daily",
                    "duration": "30 days"
                }
            ),
        ]
    )
]

# The input text to be processed
# input_text = "Elizabeth clutched Darcy's hand, trembling with both fear and excitement as they met in the moonlit garden."

# input_text = (
#     "Patient John Smith, a 52-year-old male, visited Dr. Sarah Lee, "
#     "a cardiologist, on March 10. His blood pressure was 150/95 mmHg. "
#     "Lipid panel showed LDL cholesterol of 190 mg/dL. "
#     "He was diagnosed with hypertension and prescribed Lisinopril 10 mg "
#     "once daily for 30 days."
# )

# # Run the extraction
# result = lx.extract(
#     text_or_documents=input_text,
#     prompt_description=prompt,
#     examples=examples,
#     model_id="gemini-2.5-flash",
# )

# # Save the results to a JSONL file
# lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# # Generate the visualization from the file
# html_content = lx.visualize("extraction_results.jsonl")
# with open("visualization.html", "w") as f:
#     if hasattr(html_content, 'data'):
#         f.write(html_content.data)  # For Jupyter/Colab
#     else:
#         f.write(html_content)

# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    user_text = request.json.get("text", "")

    result = lx.extract(
        text_or_documents=user_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash",
    )

    # Convert to simple JSON structure
    output = []
    for e in result.extractions:
        output.append({
            "class": e.extraction_class,
            "text": e.extraction_text,
            "attributes": e.attributes
        })

    return jsonify(output)


# if __name__ == "__main__":
#     app.run(debug=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)