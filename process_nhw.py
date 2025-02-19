import numpy as np
import xml.etree.ElementTree as ET
import re
import json

xml_path = 'data/nethackwiki_current.xml'

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespaces = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
    return root, namespaces

def clean_text(text):
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_game_content(root, namespaces):
    relevant_articles = []
    
    print(f"Root tag: {root.tag}") # Debug
    print(f"Number of children: {len(root)}") # Debug
    
    pages = root.findall('.//mw:page', namespaces)
    print(f"Number of pages found: {len(pages)}") # Debug
    
    for page in pages:
        try:
            title_elem = page.find('.//mw:title', namespaces)
            text_elem = page.find('.//mw:text', namespaces)
            
            if title_elem is None or text_elem is None:
                continue
                
            title = title_elem.text
            text = text_elem.text
            
            if not title or not text:
                continue
            
            if any(prefix in title for prefix in ['Talk:', 'User:', 'Template:', 'Category:', 'MediaWiki:']):
                continue
                
            cleaned_text = clean_text(text)
            
            # Create structured document. This can be modified to include more information.
            document = {
                'title': title,
                'content': cleaned_text,
                'type': classify_content(title, cleaned_text)
            }
            
            relevant_articles.append(document)
            
        except Exception as e:
            print(f"Error processing page: {e}")
    
    return relevant_articles

def classify_content(title, text):
    """Classify the type of content for better RAG retrieval"""
    if any(keyword in title.lower() for keyword in ['monster', 'creatures']):
        return 'monster'
    elif any(keyword in title.lower() for keyword in ['potion', 'scroll', 'wand', 'ring', 'amulet', 'armor', 'weapon']):
        return 'item'
    elif any(keyword in title.lower() for keyword in ['strategy', 'tips', 'guide']):
        return 'strategy'
    elif any(keyword in title.lower() for keyword in ['dungeon', 'level', 'branch']):
        return 'location'
    else:
        return 'general'

def main():
    try:
        print(f"Attempting to read XML from: {xml_path}")
        root, namespaces = parse_xml(xml_path)
        articles = extract_game_content(root, namespaces)
        
        print(f"\nTotal processed articles: {len(articles)}")
        
        if len(articles) > 0:
            content_types = {}
            for article in articles:
                content_types[article['type']] = content_types.get(article['type'], 0) + 1
            
            print("\nContent type distribution:")
            for content_type, count in content_types.items():
                print(f"{content_type}: {count}")
                
            print("\nFirst few article titles:")
            for article in articles[:5]:
                print(f"- {article['title']} ({article['type']})")

        with open('\data\processed_nethack_wiki.json', 'w') as f:
            json.dump(articles, f, indent=2)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
