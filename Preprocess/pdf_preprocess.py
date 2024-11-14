
import pdfplumber
import pymupdf
import pytesseract
from PIL import Image
import io
from pdf2image import convert_from_path

def read_pdf_pdfplumber_with_ocr(pdf_loc, page_infos: list = None):
    """
    Read PDF file with OCR into text using pdfplumber and pytesseract with a little
    help from pymupdf.

    Args:
        pdf_loc (str): The location of the PDF file.
        page_infos (list): The range of pages to extract text from. This is not used.
    
    Returns:
        str: The extracted text from the PDF file.
        bool: Whether OCR was used to extract text.
    """
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    pdf_pymu = pymupdf.open(pdf_loc)

    ocred = False

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for page_num, (page, page_pymu) in enumerate(zip(pages,pdf_pymu)):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text

        if len(text) > 200:
            continue

        ocred = True

        # check if there is image in the page
        image_text = ''
        images = page_pymu.get_images(full=True)
        if not images:
            continue
        if len(images)>1:
            pdf_image = convert_from_path(pdf_loc, first_page=page_num+1, last_page=page_num+1, dpi=300)[0]
        else:
            img_info=images[0]
            xref = img_info[0]  # XObject reference number
            base_image = pdf_pymu.extract_image(xref)
            image_bytes = base_image["image"]
            try: 
                pdf_image = Image.open(io.BytesIO(image_bytes))
            except:
                pdf_image = convert_from_path(pdf_loc, first_page=page_num+1, last_page=page_num+1, dpi=300)[0]
            #pdf_image = images[0].to_pil()

            if pdf_image.mode == "CMYK":
                pdf_image = pdf_image.convert("RGB")
        
        image_text = pytesseract.image_to_string(pdf_image, lang='chi_tra')

        image_text.replace('\u240C', '')
        image_text = image_text.strip()
        if image_text != '' and len(image_text) > 10:
            pdf_text+= image_text
        
    pdf.close()  # 關閉PDF文件

    return pdf_text, ocred  # 返回萃取出的文本

def read_pdf_pdfplumber(pdf_loc, page_infos: list = None):
    """
    Read PDF file into text using pdfplumber.

    Args:
        pdf_loc (str): The location of the PDF file.
        page_infos (list): The range of pages to extract text from. This is not used.

    Returns:    
        str: The extracted text from the PDF file.

    """
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    pdf_text = ''

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    for page in pages:  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text  # 合并提取的文本內容

    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本