from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Read the text file
with open('data/PMarcaBlogs.txt', 'r', encoding='utf-8') as file:
    text_content = file.read()

# Create a PDF document
doc = SimpleDocTemplate("data/pdf_samples/PMarcaBlogs.pdf", pagesize=letter)
story = []

# Define styles
styles = getSampleStyleSheet()
normal_style = styles['Normal']

# Split text into paragraphs and add to the story
paragraphs = text_content.split('\n\n')
for paragraph in paragraphs:
    if paragraph.strip():
        p = Paragraph(paragraph.replace('\n', '<br/>'), normal_style)
        story.append(p)
        story.append(Spacer(1, 12))  # Add space between paragraphs

# Build the PDF
doc.build(story)

print("PDF created successfully at data/pdf_samples/PMarcaBlogs.pdf") 