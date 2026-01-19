import base64
import json
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage


def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate(image_path):
    """Process invoice image using NVIDIA NIM vision model"""
    
    try:
        # Initialize the vision model
        llm = ChatNVIDIA(
            model="meta/llama-3.2-90b-vision-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2
        )
        
        # Convert image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Determine image format
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            image_format = "jpeg"
        elif image_path.lower().endswith('.png'):
            image_format = "png"
        else:
            raise ValueError("Unsupported image format. Use JPG, JPEG, or PNG")
        
        # Create extraction prompt
        prompt = """Extract all invoice details from this image and return as JSON with this exact structure:
{
    "vendor": {
        "name": "",
        "address": "",
        "phone": "",
        "email": ""
    },
    "order_details": {
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "po_number": ""
    },
    "items": [
        {
            "description": "",
            "quantity": 0,
            "unit_price": 0.0,
            "amount": 0.0
        }
    ],
    "payment_details": {
        "subtotal": 0.0,
        "tax": 0.0,
        "total": 0.0,
        "currency": ""
    }
}
Return only valid JSON. Extract all visible information accurately."""
        
        # Create the message with proper format according to LangChain docs
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}}
            ]
        )
        
        # Invoke the model
        print("Sending request to NVIDIA NIM...")
        response = llm.invoke([message])
        
        # Parse JSON response
        ocr_result = parse_ocr_response(response.content)
        return ocr_result
        
    except Exception as e:
        raise Exception(f"Failed to process image with NVIDIA NIM: {str(e)}")


def parse_ocr_response(response_text):
    """Parse the OCR response and extract JSON"""
    try:
        # Clean up response text
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            
            # Ensure all required keys exist
            default_structure = {
                "vendor": {"name": "", "address": "", "phone": "", "email": ""},
                "order_details": {"invoice_number": "", "invoice_date": "", "due_date": "", "po_number": ""},
                "items": [],
                "payment_details": {"subtotal": 0.0, "tax": 0.0, "total": 0.0, "currency": "USD"}
            }
            
            # Merge with defaults
            for key in default_structure:
                if key not in parsed_data:
                    parsed_data[key] = default_structure[key]
            
            return parsed_data
        else:
            # If no JSON found, return minimal structure
            return {
                "raw_response": response_text,
                "vendor": {"name": "Unknown"},
                "order_details": {},
                "items": [],
                "payment_details": {}
            }
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Response text: {response_text[:500]}")
        # Return a default structure if parsing fails
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response_text[:1000],
            "vendor": {"name": "Parse Error"},
            "order_details": {},
            "items": [],
            "payment_details": {}
        }


def save_ocr_to_db(conn, ocr_result):
    """Save OCR results to database"""
    cursor = conn.cursor()
    
    # Extract vendor info
    vendor = ocr_result.get('vendor', {})
    vendor_name = vendor.get('name', 'Unknown')
    vendor_address = vendor.get('address', '')
    vendor_phone = vendor.get('phone', '')
    vendor_email = vendor.get('email', '')
    
    # Extract order details
    order_details = ocr_result.get('order_details', {})
    invoice_number = order_details.get('invoice_number', '')
    invoice_date = order_details.get('invoice_date', '')
    due_date = order_details.get('due_date', '')
    po_number = order_details.get('po_number', '')
    
    # Extract payment details
    payment_details = ocr_result.get('payment_details', {})
    subtotal = float(payment_details.get('subtotal', 0.0))
    tax = float(payment_details.get('tax', 0.0))
    total = float(payment_details.get('total', 0.0))
    currency = payment_details.get('currency', 'USD')
    
    # Insert into documents table
    cursor.execute('''
        INSERT INTO documents (
            vendor_name, vendor_address, vendor_phone, vendor_email,
            invoice_number, invoice_date, due_date, po_number,
            subtotal, tax, total, currency, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        vendor_name, vendor_address, vendor_phone, vendor_email,
        invoice_number, invoice_date, due_date, po_number,
        subtotal, tax, total, currency, json.dumps(ocr_result)
    ))
    
    doc_id = cursor.lastrowid
    print(f"✅ Inserted document with ID: {doc_id}")
    
    # Insert items if they exist
    items = ocr_result.get('items', [])
    print(f"🔍 DEBUG: Found {len(items)} items in OCR result")
    print(f"🔍 DEBUG: Items data: {items}")
    
    items_inserted = 0
    for item in items:
        try:
            description = item.get('description', '')
            quantity = item.get('quantity', 0)
            unit_price = item.get('unit_price', 0.0)
            amount = item.get('amount', 0.0)
            
            # Skip empty items
            if not description and quantity == 0:
                print(f"⚠️ Skipping empty item")
                continue
            
            cursor.execute('''
                INSERT INTO items (
                    document_id, description, quantity, unit_price, amount
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                doc_id,
                description,
                int(quantity),
                float(unit_price),
                float(amount)
            ))
            items_inserted += 1
            print(f"✅ Inserted item: {description[:50] if description else 'Unnamed item'}")
        except Exception as e:
            print(f"❌ Failed to insert item: {e}")
            print(f"   Item data: {item}")
    
    # Commit all changes
    conn.commit()
    print(f"✅ Committed {items_inserted} items for doc_id {doc_id}")
    
    # VERIFY ITEMS SAVED
    cursor.execute("SELECT COUNT(*) FROM items WHERE document_id = ?", (doc_id,))
    saved_items_count = cursor.fetchone()
    print(f"✅ VERIFIED: {saved_items_count} items in DB for doc_id {doc_id}")
    
    if items_inserted != saved_items_count:
        print(f"⚠️ WARNING: Inserted {items_inserted} but DB shows {saved_items_count}")
    
    return doc_id
