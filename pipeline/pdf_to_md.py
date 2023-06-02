import os
import pdfplumber


def pdf_to_md(pdf_path: str, md_path: str):
    """
    Convert a PDF file to Markdown format.
    Extracts text and tables from a PDF and writes them as Markdown.
    Args:
        pdf_path (str): Path to the input PDF file.
        md_path (str): Path to the output Markdown file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file {pdf_path} does not exist.")

    markdown = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            markdown += f"\n\n---\n\n## Page {page_num}\n\n"
            markdown += text if text else "*No text found on this page.*"

            tables = page.extract_tables()
            for table in tables:
                markdown += "\n\n"
                for row in table:
                    markdown += "| " + " | ".join(cell or "" for cell in row) + " |\n"
                markdown += "\n"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    return True

def process_pdfs(pdf_dir="data/10/english", md_dir="books/10/english"):
    os.makedirs(md_dir, exist_ok=True)

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(pdf_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".md"
            output_path = os.path.join(md_dir, output_filename)

            print(f"Processing: {filename}")
            pdf_to_md(input_path, output_path)

    print("All PDFs converted to Markdown.")


if __name__ == "__main__":
    pdf_dir = "../data/10/social"
    md_dir = "../books/10/social"
    process_pdfs(pdf_dir, md_dir)
