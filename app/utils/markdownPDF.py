import os
from fastapi.responses import Response
import markdown
from weasyprint import HTML
import tempfile

def markdown_pdf(markdown_txt: str) -> str:
    # Convert markdown text to HTML, enabling table support
    html_content = markdown.markdown(markdown_txt, extensions=["tables"])

    # CSS styles for headings and tables with grid lines
    css_styles = """
    <style>
        h1 {
            color: green;
            font-weight: bold;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        table, th, td {
            border: 1px solid #444;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
    """

    # Combine CSS, company header, and HTML content
    full_html = f"{css_styles}<h1>FinSolve Technologies</h1>{html_content}"

    # Create a temporary PDF file and write the rendered HTML to it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_path = tmp_pdf.name
        HTML(string=full_html).write_pdf(pdf_path)  # Render PDF

    return pdf_path  # Return path to the generated PDF file


# Helper async function to serve the generated PDF as an HTTP response
async def download_pdf(markdown_txt: str):
    pdf_path = markdown_pdf(markdown_txt)  # Convert markdown to PDF
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()  # Read PDF bytes
    os.remove(pdf_path)  # Clean up the temporary PDF file

    # Return the PDF as a streaming response with correct media type
    return Response(
        content=pdf_bytes,
        media_type="application/pdf"
    )
