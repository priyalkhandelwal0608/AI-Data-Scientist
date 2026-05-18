import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

def generate_report(results):

    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(
        "reports/final_report.pdf"
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "AI Data Scientist Agent Report",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 12))

    for model, score in results.items():

        text = Paragraph(
            f"{model}: Accuracy = {score}",
            styles['BodyText']
        )

        elements.append(text)

        elements.append(Spacer(1, 10))

    doc.build(elements)