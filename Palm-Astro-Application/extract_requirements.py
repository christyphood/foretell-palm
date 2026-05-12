import sys

def extract_text(pdf_path):
    try:
        import pypdf
        print("Using pypdf")
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    try:
        import PyPDF2
        print("Using PyPDF2")
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        return "Error: Neither pypdf nor PyPDF2 is installed. Please install one of them (e.g., pip install pypdf)."
    except Exception as e:
        return f"Error reading PDF: {e}"

if __name__ == "__main__":
    pdf_file = "Python Test 1 Palm History (1).pdf"
    print(extract_text(pdf_file))
