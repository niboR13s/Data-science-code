
# import os
# import subprocess
# import docx
# from docx import Document
# from docx.oxml.shared import OxmlElement, qn
# from docx.opc.constants import RELATIONSHIP_TYPE

# def add_hyperlink(paragraph, url, text):
#     """
#     A helper function to add a clickable hyperlink to a paragraph.
#     python-docx does not support this natively, so we must manipulate the XML.
#     """
#     # This gets access to the document.xml.rels file and creates a new relation id
#     part = paragraph.part
#     r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

#     # Create the w:hyperlink tag and add needed values
#     hyperlink = OxmlElement('w:hyperlink')
#     hyperlink.set(qn('r:id'), r_id, )

#     # Create a w:r element (run) inside the hyperlink
#     new_run = OxmlElement('w:r')
#     rPr = OxmlElement('w:rPr')

#     # Add color (blue)
#     c = OxmlElement('w:color')
#     c.set(qn('w:val'), "0000FF")
#     rPr.append(c)

#     # Add underline
#     u = OxmlElement('w:u')
#     u.set(qn('w:val'), "single")
#     rPr.append(u)

#     new_run.append(rPr)
#     new_run.text = text
#     hyperlink.append(new_run)

#     # Add the hyperlink to the paragraph
#     paragraph._p.append(hyperlink)

#     return hyperlink

# def get_git_remote_info():
#     """
#     Automatically detects the git remote URL and current branch.
#     """
#     try:
#         remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
#         branch = subprocess.check_output(["git", "branch", "--show-current"]).decode().strip()

#         if remote_url.startswith("git@"):
#             remote_url = remote_url.replace(":", "/")
#             remote_url = remote_url.replace("git@", "https://")
        
#         if remote_url.endswith(".git"):
#             remote_url = remote_url[:-4]

#         return f"{remote_url}/blob/{branch}"

#     except Exception as e:
#         print(f"Warning: Could not auto-detect Git info ({e}). Using placeholder.")
#         return "https://github.com/YOUR_USER/YOUR_REPO/blob/main"

# def get_file_description(file_path):
#     """
#     Reads the first 10 lines to find comments.
#     """
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             lines = f.readlines()
#             comments = []
#             for line in lines[:10]: 
#                 stripped = line.strip()
#                 if stripped.startswith(("#", "//", "/*")):
#                     comments.append(stripped)
            
#             if comments:
#                 return " ".join(comments)
#     except Exception:
#         pass
#     return None

# def generate_weekly_reports():
#     # --- CONFIGURATION ---
#     base_folder = "Homework"
#     output_folder = "Homework_doxmaker" 
#     allowed_extensions = [".py", ".txt", ".md", ".java", ".html", ".css", ".js", ".sql",".ipynb"]
    
#     github_base_url = get_git_remote_info()
#     print(f"--- Detected Git URL: {github_base_url} ---")
#     # ---------------------

#     if not os.path.exists(output_folder):
#         try:
#             os.makedirs(output_folder)
#             print(f"--- Created output folder: {output_folder} ---")
#         except OSError as e:
#             print(f"Error creating directory {output_folder}: {e}")
#             return

#     if not os.path.exists(base_folder):
#         print(f"Error: Source folder '{base_folder}' not found.")
#         return

#     for item in os.listdir(base_folder):
#         week_path = os.path.join(base_folder, item)

#         if os.path.isdir(week_path):
#             print(f"Processing folder: {item}...")
            
#             doc = Document()
#             doc.add_heading(f'Homework Overview: {item}', 0)
            
#             files_found = False

#             for root, dirs, files in os.walk(week_path):
#                 for file in files:
#                     _, ext = os.path.splitext(file)
#                     if ext.lower() in allowed_extensions:
#                         files_found = True
#                         file_full_path = os.path.join(root, file)
                        
#                         # 1. Heading (Filename)
#                         doc.add_heading(file, level=2)

#                         # 2. Dynamic GitHub Link (Clickable!)
#                         rel_path = os.path.relpath(file_full_path, start=os.getcwd())
#                         url_path = rel_path.replace("\\", "/")
#                         full_link = f"{github_base_url}/{url_path}"

#                         p_link = doc.add_paragraph()
#                         p_link.add_run("Link: ").bold = True
                        
#                         # Use our helper function to add the clickable link
#                         add_hyperlink(p_link, full_link, "View on GitHub")

#                         # 3. Description
#                         desc_text = get_file_description(file_full_path)
#                         if desc_text:
#                             p_desc = doc.add_paragraph()
#                             p_desc.add_run("Description: ").bold = True
#                             p_desc.add_run(desc_text).italic = True

#             if files_found:
#                 output_filename = f"Overview_{item}.docx"
#                 full_save_path = os.path.join(output_folder, output_filename)
                
#                 doc.save(full_save_path)
#                 print(f"  -> Saved to: {full_save_path}")
#             else:
#                 print(f"  -> No valid files found in {item}")

#     print("--- All done! ---")

# if __name__ == "__main__":
#     generate_weekly_reports()



import os
import subprocess
import json
from docx import Document
from docx.oxml.shared import OxmlElement, qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import RGBColor

def add_hyperlink(paragraph, url, text):
    """
    A helper function to add a clickable hyperlink to a paragraph.
    """
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id, )

    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    c = OxmlElement('w:color')
    c.set(qn('w:val'), "0000FF")
    rPr.append(c)

    u = OxmlElement('w:u')
    u.set(qn('w:val'), "single")
    rPr.append(u)

    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)

    paragraph._p.append(hyperlink)

    return hyperlink

def get_git_remote_info():
    """
    Automatically detects the git remote URL and current branch.
    """
    try:
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"]).decode().strip()

        if remote_url.startswith("git@"):
            remote_url = remote_url.replace(":", "/")
            remote_url = remote_url.replace("git@", "https://")
        
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        return f"{remote_url}/blob/{branch}"

    except Exception as e:
        print(f"Warning: Could not auto-detect Git info ({e}). Using placeholder.")
        return "https://github.com/YOUR_USER/YOUR_REPO/blob/main"

def get_file_description(file_path):
    """
    Extracts a description based on file type.
    - .ipynb: Extracts text from the first Markdown cell.
    - Code files: Extracts comments from the first 10 lines.
    """
    _, ext = os.path.splitext(file_path)
    
    try:
        # HANDLING JUPYTER NOTEBOOKS
        if ext.lower() == '.ipynb':
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Iterate through cells to find the first markdown cell
                for cell in data.get('cells', []):
                    if cell.get('cell_type') == 'markdown':
                        # The source is usually a list of strings (lines)
                        source_content = cell.get('source', [])
                        if isinstance(source_content, list):
                            return "".join(source_content).strip()
                        return str(source_content).strip()
            return None # No markdown cell found

        # HANDLING REGULAR CODE FILES
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                comments = []
                for line in lines[:10]: 
                    stripped = line.strip()
                    if stripped.startswith(("#", "//", "/*")):
                        comments.append(stripped)
                
                if comments:
                    return " ".join(comments)

    except Exception as e:
        print(f"Error reading description for {file_path}: {e}")
    
    return None

def generate_weekly_reports():
    # --- CONFIGURATION ---
    base_folder = "Homework"
    output_folder = "Homework_doxmaker" 
    # Added .ipynb to the allowed extensions
    allowed_extensions = [".py", ".txt", ".md", ".java", ".html", ".css", ".js", ".sql", ".ipynb"]
    
    github_base_url = get_git_remote_info()
    print(f"--- Detected Git URL: {github_base_url} ---")
    # ---------------------

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except OSError as e:
            print(f"Error creating directory {output_folder}: {e}")
            return

    if not os.path.exists(base_folder):
        print(f"Error: Source folder '{base_folder}' not found.")
        return

    for item in os.listdir(base_folder):
        week_path = os.path.join(base_folder, item)

        if os.path.isdir(week_path):
            print(f"Processing folder: {item}...")
            
            doc = Document()
            doc.add_heading(f'Homework Overview: {item}', 0)
            
            files_found = False

            for root, dirs, files in os.walk(week_path):
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext.lower() in allowed_extensions:
                        files_found = True
                        file_full_path = os.path.join(root, file)
                        
                        # 1. Heading (Filename)
                        doc.add_heading(file, level=2)

                        # 2. Dynamic GitHub Link
                        rel_path = os.path.relpath(file_full_path, start=os.getcwd())
                        url_path = rel_path.replace("\\", "/")
                        full_link = f"{github_base_url}/{url_path}"

                        p_link = doc.add_paragraph()
                        p_link.add_run("Link: ").bold = True
                        add_hyperlink(p_link, full_link, "View on GitHub")

                        # 3. Description (Markdown cell or comments)
                        desc_text = get_file_description(file_full_path)
                        if desc_text:
                            p_desc = doc.add_paragraph()
                            p_desc.add_run("Description: ").bold = True
                            # Limit description length if it's huge (optional safety)
                            if len(desc_text) > 500:
                                desc_text = desc_text[:500] + "..."
                            p_desc.add_run(desc_text).italic = True

            if files_found:
                output_filename = f"Overview_{item}.docx"
                full_save_path = os.path.join(output_folder, output_filename)
                
                doc.save(full_save_path)
                print(f"  -> Saved to: {full_save_path}")
            else:
                print(f"  -> No valid files found in {item}")

    print("--- All done! ---")

if __name__ == "__main__":
    generate_weekly_reports()