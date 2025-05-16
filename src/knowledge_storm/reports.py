import os
import re
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.units import inch

class ResearchReportGenerator:
    """
    Generates a professionally formatted research report from STORM Wiki output.
    Supports PDF generation with IEEE-style citations, title page, TOC, headers/footers.
    """

    def __init__(self, output_dir: str, topic: str):
        self.output_dir = output_dir
        self.topic = topic
        self.citations_dict = {}
        
        # Determine if we're pointing to the topic subfolder or the results folder
        # Check if we need to append the topic name to the path
        topic_dir = os.path.join(output_dir, topic.replace(' ', '_'))
        if os.path.isdir(topic_dir):
            print(f"Found topic directory: {topic_dir}")
            self.base_dir = topic_dir
        else:
            # Try to find a subfolder that matches the topic
            for subdir in os.listdir(output_dir):
                full_subdir = os.path.join(output_dir, subdir)
                if os.path.isdir(full_subdir) and topic.lower() in subdir.lower():
                    print(f"Found matching topic directory: {full_subdir}")
                    self.base_dir = full_subdir
                    break
            else:
                print(f"Using provided directory directly: {output_dir}")
                self.base_dir = output_dir
        
        # Define paths based on base_dir
        self.research_path = os.path.join(self.base_dir, "raw_search_results.json")
        self.url_to_info_path = os.path.join(self.base_dir, "url_to_info.json")
        self.outline_path = os.path.join(self.base_dir, "storm_gen_outline.txt")
        self.direct_outline_path = os.path.join(self.base_dir, "direct_gen_outline.txt")
        self.article_path = os.path.join(self.base_dir, "storm_gen_article.txt")
        self.polished_article_path = os.path.join(self.base_dir, "storm_gen_article_polished.txt")
        
        # Get the maximum citation number from url_to_info.json
        self.max_citations = self._get_max_citation_number()
        print(f"Maximum citation number from data: {self.max_citations}")
        
        # Print file paths and existence for debugging
        self._print_file_status()
    
    def _get_max_citation_number(self) -> int:
        """Get the maximum citation number from url_to_info.json."""
        try:
            if os.path.exists(self.url_to_info_path):
                with open(self.url_to_info_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "url_to_unified_index" in data:
                        # Extract all citation numbers from the index
                        citation_numbers = list(data["url_to_unified_index"].values())
                        if citation_numbers:
                            return max(citation_numbers)  # Return the maximum number
            
            # If we can't read from url_to_info.json, check raw_search_results.json
            if os.path.exists(self.research_path):
                with open(self.research_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return len(data)  # Use the count of sources as max
        except Exception as e:
            print(f"Error determining maximum citation number: {e}")
        
        # Default if we can't determine from files
        return 100  # Better to have a larger default than too small
    
    def _print_file_status(self):
        """Print file paths and check if they exist."""
        files = [
            ("Research data", self.research_path),
            ("URL to info", self.url_to_info_path),
            ("Outline", self.outline_path),
            ("Direct outline", self.direct_outline_path),
            ("Article", self.article_path),
            ("Polished article", self.polished_article_path)
        ]
        
        print("\nChecking input files:")
        for name, path in files:
            exists = os.path.exists(path)
            status = "FOUND" if exists else "NOT FOUND"
            print(f"  {name}: {path} - {status}")

    def extract_abstract(self) -> str:
        """Extract abstract from the polished article."""
        polished_content = ""
        
        # Try to read from polished article first
        if os.path.exists(self.polished_article_path):
            try:
                with open(self.polished_article_path, encoding='utf-8') as f:
                    polished_content = f.read()
                    print(f"Successfully read polished article: {len(polished_content)} characters")
            except Exception as e:
                print(f"Error reading polished article: {e}")
        
        # If that fails, try regular article
        if not polished_content and os.path.exists(self.article_path):
            try:
                with open(self.article_path, encoding='utf-8') as f:
                    polished_content = f.read()
                    print(f"Successfully read article: {len(polished_content)} characters")
            except Exception as e:
                print(f"Error reading article: {e}")
        
        # If we still have no content, return default
        if not polished_content:
            print("Warning: No article content found for abstract extraction")
            return "Abstract not available."
            
        # Look for summary section first
        summary_match = re.search(r'#\s*summary\s*\n\n(.*?)(?=\n#|\Z)', polished_content, re.DOTALL | re.IGNORECASE)
        if summary_match:
            return summary_match.group(1).strip()
        
        # Try to extract abstract from dedicated section
        abstract_match = re.search(r'(?:Abstract|Introduction):\s*(.*?)(?:\n\n|\n#)', polished_content, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        
        # Fallback to first substantial paragraph
        paragraphs = polished_content.split('\n\n')
        for p in paragraphs:
            if len(p) > 100 and not p.startswith('#'):
                return p.strip()
        
        return "Abstract not available."

    def format_ieee_citation(self, source: Dict[str, Any], ref_num: int) -> str:
        """Format and register an IEEE-style citation."""
        url = source.get('url', '')
        title = source.get('title', url.split('/')[-1] if '/' in url else 'Unknown source')
        try:
            domain = urlparse(url).netloc
            author = publisher = domain.replace('www.', '')
        except Exception:
            author = publisher = "Unknown"
        year = datetime.now().year
        access = datetime.now().strftime('%d %B %Y')
        citation = f"[{ref_num}] {author}, \"{title},\" {publisher}, {year}. [Online]. Available: {url}. [Accessed: {access}]."
        self.citations_dict[url] = ref_num
        return citation

    def extract_citations(self) -> List[str]:
        """Extract and format all citations."""
        citations = []
        sources = []
        
        # Try to read from url_to_info.json first for better metadata
        if os.path.exists(self.url_to_info_path):
            try:
                with open(self.url_to_info_path, encoding='utf-8') as f:
                    data = json.load(f)
                    url_to_info = data.get("url_to_info", {})
                    url_to_index = data.get("url_to_unified_index", {})
                    
                    # Create a list of sources with metadata, sorted by citation index
                    source_list = []
                    for url, index in url_to_index.items():
                        info = url_to_info.get(url, {})
                        if not info:
                            info = {"url": url}
                        source_list.append((index, info))
                    
                    # Sort by citation index
                    source_list.sort(key=lambda x: x[0])
                    sources = [info for _, info in source_list]
                    
                    print(f"Extracted {len(sources)} sources from url_to_info.json")
            except Exception as e:
                print(f"Error reading url_to_info.json: {e}")
                sources = []  # Reset sources if there was an error
        
        # Fallback to raw_search_results.json if needed
        if not sources and os.path.exists(self.research_path):
            try:
                with open(self.research_path, encoding='utf-8') as f:
                    raw_data = json.load(f)
                    print(f"Successfully loaded raw_search_results.json")
                    
                    if isinstance(raw_data, dict):
                        # If it's a dictionary, extract URLs as keys
                        sources = [{"url": url} for url in raw_data.keys() 
                                  if isinstance(url, str) and url.startswith('http')]
                    elif isinstance(raw_data, list):
                        # If it's a list, assume URLs directly
                        sources = [{"url": url} for url in raw_data 
                                  if isinstance(url, str) and url.startswith('http')]
                    
                    print(f"Extracted {len(sources)} citation sources from raw_search_results.json")
            except Exception as e:
                print(f"Error extracting citations: {e}")
        
        # Format citations - respect the maximum determined from url_to_info.json
        actual_max = min(len(sources), self.max_citations)
        for idx, source in enumerate(sources[:actual_max]):
            citations.append(self.format_ieee_citation(source, idx + 1))
        
        print(f"Generated {len(citations)} formatted citations (max allowed: {self.max_citations})")
        return citations

    def process_content_for_citations(self, content: str) -> str:
        """Process content to handle inline citations."""
        # Replace citation numbers in brackets with IEEE format if needed
        return content

    def generate_pdf(self) -> str:
        """Generate the PDF research report."""
        print(f"\nGenerating research report for topic: {self.topic}")
        
        # Extract all citations first
        citations = self.extract_citations()
        
        # Set up reportlab components
        styles = getSampleStyleSheet()

        # Adjust heading styles
        styles['Heading1'].alignment = TA_LEFT
        styles['Heading1'].fontName = 'Times-Bold'
        styles['Heading1'].fontSize = 14
        styles['Heading1'].spaceAfter = 12
        styles['Heading1'].spaceBefore = 24
        styles['Heading1'].keepWithNext = True

        styles['Heading2'].alignment = TA_LEFT
        styles['Heading2'].fontName = 'Times-Bold'
        styles['Heading2'].fontSize = 12
        styles['Heading2'].spaceAfter = 10
        styles['Heading2'].spaceBefore = 12
        styles['Heading2'].keepWithNext = True

        styles['Heading3'].alignment = TA_LEFT
        styles['Heading3'].fontName = 'Times-Bold'
        styles['Heading3'].fontSize = 11
        styles['Heading3'].spaceAfter = 8
        styles['Heading3'].spaceBefore = 10
        styles['Heading3'].keepWithNext = True

        # Add custom styles
        if 'Justify' not in styles:
            styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontName='Times-Roman', fontSize=11))
        if 'Title' not in styles:
            styles.add(ParagraphStyle(name='Title', alignment=TA_CENTER, fontName='Times-Bold', fontSize=20, spaceAfter=30))
        if 'Author' not in styles:
            styles.add(ParagraphStyle(name='Author', alignment=TA_CENTER, fontName='Times-Roman', fontSize=12))
        if 'TOCHeading' not in styles:
            styles.add(ParagraphStyle(name='TOCHeading', alignment=TA_CENTER, fontName='Times-Bold', fontSize=14, spaceAfter=20))
        if 'TOCEntry' not in styles:
            styles.add(ParagraphStyle(name='TOCEntry', alignment=TA_LEFT, fontName='Times-Roman', fontSize=11, leftIndent=20, spaceAfter=6))
        if 'Abstract' not in styles:
            styles.add(ParagraphStyle(name='Abstract', alignment=TA_JUSTIFY, fontName='Times-Italic', fontSize=10))
        if 'Citation' not in styles:
            styles.add(ParagraphStyle(name='Citation', alignment=TA_LEFT, fontName='Times-Roman', fontSize=10, leftIndent=24, firstLineIndent=-24))

        # Ensure output directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        safe_topic = self.topic.replace(' ', '_').replace('/', '_')
        pdf_path = os.path.join(self.base_dir, f"{safe_topic}_research_paper.pdf")
        print(f"Will generate PDF at: {pdf_path}")

        # Create PDF template with headers/footers
        class ResearchPaperTemplate(SimpleDocTemplate):
            def __init__(self, filename, **kwargs):
                self.topic = kwargs.pop('topic', 'Research Paper')
                self.page = 0
                super().__init__(filename, **kwargs)
            def beforePage(self):
                self.page += 1
                self.canv.saveState()
                if self.page > 1:
                    self.canv.setFont('Times-Roman', 9)
                    self.canv.drawString(72, 780, self.topic)
                    self.canv.drawRightString(letter[0] - 72, 780, datetime.now().strftime('%B %d, %Y'))
                    self.canv.line(72, 775, letter[0] - 72, 775)
                self.canv.setFont('Times-Roman', 9)
                self.canv.drawCentredString(letter[0] / 2, 30, f"Page {self.page}")
                self.canv.line(72, 40, letter[0] - 72, 40)
                self.canv.restoreState()

        doc = ResearchPaperTemplate(pdf_path, pagesize=letter, rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=72, topic=self.topic)
        story = []

        # Title Page
        story.append(Spacer(1, 2 * inch))
        story.append(Paragraph(self.topic, styles['Title']))
        story.append(Spacer(1, inch))
        today = datetime.now().strftime('%B %d, %Y')
        story.append(Paragraph("Research Report", styles['Author']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Generated by AvePoint Research Assistant", styles['Author']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(today, styles['Author']))
        story.append(PageBreak())

        # Table of Contents
        story.append(Paragraph("Table of Contents", styles['TOCHeading']))
        
        # Read article content for TOC and main content
        print("Reading article content for TOC and main content")
        article_content = ""
        if os.path.exists(self.polished_article_path):
            try:
                with open(self.polished_article_path, encoding='utf-8') as f:
                    article_content = f.read()
                    print(f"Using polished article content: {len(article_content)} characters")
            except Exception as e:
                print(f"Error reading polished article: {e}")
        
        if not article_content and os.path.exists(self.article_path):
            try:
                with open(self.article_path, encoding='utf-8') as f:
                    article_content = f.read()
                    print(f"Using regular article content: {len(article_content)} characters")
            except Exception as e:
                print(f"Error reading article: {e}")
        
        # Generate TOC from headings
        if article_content:
            headings = re.findall(r'^(#+)\s+(.*?)$', article_content, re.MULTILINE)
            page_num = 3
            print(f"Found {len(headings)} headings for TOC")
            for level, title in headings:
                lvl = len(level)
                if lvl <= 2:
                    indent = "&nbsp;" * ((lvl - 1) * 4)
                    toc_entry = f"{indent}{title} {'.' * (40 - len(title))} {page_num}"
                    story.append(Paragraph(toc_entry, styles['TOCEntry']))
                    page_num += 1
        else:
            print("Warning: No article content found for TOC generation")
        
        story.append(PageBreak())

        # Abstract
        print("Extracting abstract")
        story.append(Paragraph("Abstract", styles['Heading1']))
        abstract = self.extract_abstract()
        story.append(Paragraph(abstract, styles['Abstract']))
        story.append(Spacer(1, 24))

        # Main Content
        print("Processing main content")
        has_intro = has_concl = False
        
        if article_content:
            content = self.process_content_for_citations(article_content)
            has_intro = bool(re.search(r'#+\s+(Introduction|Overview)', content, re.IGNORECASE))
            has_concl = bool(re.search(r'#+\s+(Conclusion|Summary|Final\s*Thoughts)', content, re.IGNORECASE))
            
            # Split content by headings
            sections = re.split(r'(#+\s+.*)', content)
            for section in sections:
                if section.strip():
                    if section.startswith('#'):
                        lvl = len(re.match(r'#+', section).group(0))
                        heading = section.strip('#').strip()
                        if lvl == 1:
                            story.append(Paragraph(heading, styles['Heading1']))
                        elif lvl == 2:
                            story.append(Paragraph(heading, styles['Heading2']))
                        else:
                            story.append(Paragraph(heading, styles['Heading3']))
                    else:
                        for p in section.split('\n\n'):
                            if p.strip():
                                story.append(Paragraph(self._format_markdown(p), styles['Justify']))
                                story.append(Spacer(1, 6))
        else:
            print("Warning: No article content found for main content")
        
        # Add introduction if missing
        if not has_intro:
            print("Adding introduction section")
            idx = 0
            for i, el in enumerate(story):
                if hasattr(el, 'text') and 'Abstract' in el.text:
                    idx = i + 3
                    break
            story.insert(idx, Paragraph("Introduction", styles['Heading1']))
            story.insert(idx + 1, Paragraph(
                f"This research paper explores {self.topic}, analyzing key aspects and findings from available literature. "
                f"The following sections present detailed information on various dimensions of this subject.", 
                styles['Justify']))
            story.insert(idx + 2, Spacer(1, 12))
        
        # Add conclusion if missing
        if not has_concl:
            print("Adding conclusion section")
            story.append(Paragraph("Conclusion", styles['Heading1']))
            story.append(Paragraph(
                f"This report has examined {self.topic} through multiple perspectives. "
                f"The research suggests important implications for understanding this subject "
                f"and provides a foundation for further investigation.", 
                styles['Justify']))
            story.append(Spacer(1, 12))

        # References
        print(f"Adding references section with {len(citations)} citations")
        story.append(PageBreak())
        story.append(Paragraph("References", styles['Heading1']))
        for citation in citations:
            story.append(Paragraph(citation, styles['Citation']))
            story.append(Spacer(1, 6))

        # Generate PDF
        try:
            doc.build(story)
            print(f"Research paper PDF created at: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return ""

    def _format_markdown(self, text: str) -> str:
        """Convert basic markdown to ReportLab tags."""
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
        return text

def generate_research_report(output_dir: str, topic: str) -> str:
    """Generate a research report with IEEE citations."""
    print(f"Generating research report for '{topic}' using data from '{output_dir}'")
    generator = ResearchReportGenerator(output_dir, topic)
    return generator.generate_pdf()