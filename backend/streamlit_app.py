"""
Streamlit UI for NVIDIA NIM-based OCR and NL2SQL pipeline

Requirements:
- Set NVIDIA_API_KEY environment variable
- Install: pip install streamlit python-dotenv
- Run: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

st.markdown("""
<style>
  /* App background */
  .stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }

  section.main > div {
    max-width: 100%;
    padding: 1rem;
  }

  /* Base chat message box styling (common) */
  div[data-testid="stChatMessage"]{
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12);
  }

  /* DARK MODE */
  @media (prefers-color-scheme: dark) {
    div[data-testid="stChatMessage"]{
      background: #0b0b0b !important;
    }

    /* Target the actual rendered text inside chat messages */
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #f7fafc !important;
    }
  }

  /* LIGHT MODE */
  @media (prefers-color-scheme: light) {
    div[data-testid="stChatMessage"]{
      background: #ffffff !important;
    }

    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #111827 !important;
    }
  }
</style>
""", unsafe_allow_html=True)

# Import NVIDIA NIM modules
from nvidia_ocr import generate, save_ocr_to_db
from db import init_db

# Import NL2SQL class directly to avoid circular import
try:
    from nat_sql_nvidia import NL2SQLConverter
    USE_NL2SQL_CLASS = True
except ImportError:
    USE_NL2SQL_CLASS = False
    st.error("Could not import NL2SQLConverter. Please check nat_sql_nvidia.py for circular imports.")

# Page configuration
st.set_page_config(
    page_title="NVIDIA NIM OCR & NL2SQL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #76B900;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = None
if 'doc_id' not in st.session_state:
    st.session_state.doc_id = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

def check_api_key():
    """Check if NVIDIA API key is set"""
    api_key = os.getenv("NVIDIA_API_KEY")
    return api_key is not None, api_key

def get_db_connection():
    """Get a fresh database connection for the current thread"""
    return init_db()

def save_invoice_to_db(conn, ocr_result):
    """Save OCR result to database with proper error handling"""
    try:
        cursor = conn.cursor()
        
        # Extract vendor information
        vendor = ocr_result.get('vendor', {})
        vendor_name = vendor.get('name', '')
        vendor_address = vendor.get('address', '')
        vendor_phone = vendor.get('phone', '')
        vendor_email = vendor.get('email', '')
        
        # Extract order details
        order = ocr_result.get('order_details', {})
        invoice_number = order.get('invoice_number', '')
        invoice_date = order.get('invoice_date', '')
        due_date = order.get('due_date', '')
        po_number = order.get('po_number', '')
        
        # Extract payment details
        payment = ocr_result.get('payment_details', {})
        subtotal = float(payment.get('subtotal', 0))
        tax = float(payment.get('tax', 0))
        total = float(payment.get('total', 0))
        currency = payment.get('currency', 'USD')
        
        # Store raw JSON
        import json
        raw_data = json.dumps(ocr_result)
        
        # Insert document
        cursor.execute("""
            INSERT INTO documents (
                vendor_name, vendor_address, vendor_phone, vendor_email,
                invoice_number, invoice_date, due_date, po_number,
                subtotal, tax, total, currency, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vendor_name, vendor_address, vendor_phone, vendor_email,
            invoice_number, invoice_date, due_date, po_number,
            subtotal, tax, total, currency, raw_data
        ))
        
        doc_id = cursor.lastrowid
        
        # Insert items
        items = ocr_result.get('items', [])
        for item in items:
            description = item.get('description', '')
            quantity = int(item.get('quantity', 0))
            unit_price = float(item.get('unit_price', 0))
            amount = float(item.get('amount', 0))
            
            cursor.execute("""
                INSERT INTO items (
                    document_id, description, quantity, unit_price, amount
                ) VALUES (?, ?, ?, ?, ?)
            """, (doc_id, description, quantity, unit_price, amount))
        
        conn.commit()
        return doc_id
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Database save failed: {str(e)}")


def process_image(image_file):
    """Process uploaded image with OCR"""
    try:
        # Save uploaded file temporarily
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        image_path = upload_dir / image_file.name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Process with OCR
        with st.spinner("Processing invoice with NVIDIA NIM Vision Model..."):
            ocr_result = generate(str(image_path.absolute()))
        
        st.info("✅ OCR completed successfully")
        
        # Save to database with fresh connection
        st.info("💾 Saving to database...")
        conn = get_db_connection()
        
        try:
            # Try using the existing function first
            try:
                doc_id = save_ocr_to_db(conn, ocr_result)
            except:
                # If that fails, use our backup function
                doc_id = save_invoice_to_db(conn, ocr_result)
            
            st.success(f"✅ Data saved with doc_id: {doc_id}")
        except Exception as db_error:
            st.error(f"❌ Database save failed: {db_error}")
            raise db_error
        finally:
            conn.close()
        
        # Verify the save
        verify_conn = get_db_connection()
        cursor = verify_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        verify_conn.close()
        
        st.success(f"✅ Verification: {count} document(s) in database")
        
        # Store in session state
        st.session_state.ocr_result = ocr_result
        st.session_state.doc_id = doc_id
        st.session_state.db_initialized = True
        
        return True, ocr_result, doc_id
    
    except Exception as e:
        st.error(f"Full error details: {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False, str(e), None

def execute_nl_query(question):
    """Execute natural language query"""
    try:
        # Convert to SQL using NL2SQL class
        if USE_NL2SQL_CLASS:
            converter = NL2SQLConverter()
            sql = converter.convert(question)
        else:
            st.error("NL2SQL not available due to import error")
            return False, "Import error", None
        
        # Clean up the SQL - fix duplicate column names in JOINs
        if "JOIN" in sql.upper() and "SELECT *" in sql.upper():
            # Replace SELECT * with specific columns to avoid duplicates
            if "items" in sql.lower() and "documents" in sql.lower():
                sql = sql.replace(
                    "SELECT *",
                    "SELECT items.id as item_id, items.description, items.quantity, items.unit_price, items.amount, documents.vendor_name, documents.invoice_number, documents.invoice_date, documents.total"
                )
        
        # Execute query with fresh connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        
        # Add to history
        st.session_state.query_history.append({
            'question': question,
            'sql': sql,
            'results': results
        })
        
        return True, sql, results
    
    except Exception as e:
        return False, str(e), None

# Main UI
def main():
    # Header
    st.markdown('<div class="main-header">🤖 NVIDIA NIM OCR & NL2SQL Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract data from invoices and query using natural language</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key check
        has_key, api_key = check_api_key()
        if has_key:
            st.success("✅ NVIDIA API Key detected")
            st.code(f"{api_key[:10]}...", language=None)
        else:
            st.error("❌ NVIDIA API Key not found")
            st.markdown("""
                **To fix this:**
                1. Get API key from [build.nvidia.com](https://build.nvidia.com)
                2. Create `.env` file with:
                   ```
                   NVIDIA_API_KEY=nvapi-your-key
                   ```
                3. Restart the app
            """)
            st.stop()
        
        st.divider()
        
        # Database status
        st.header("💾 Database")
        if st.button("Initialize/Reset Database"):
            conn = get_db_connection()
            conn.close()
            st.session_state.db_initialized = True
            st.success("Database initialized!")
        
        if st.session_state.db_initialized:
            st.info("✅ Initialized")
        else:
            st.warning("⚠️ Not initialized yet")
        
        st.divider()
        
        # Query history
        st.header("📜 Query History")
        if st.session_state.query_history:
            st.write(f"Total queries: {len(st.session_state.query_history)}")
            if st.button("Clear History"):
                st.session_state.query_history = []
                st.rerun()
        else:
            st.write("No queries yet")
        
        st.divider()
        
        # Debug: Show database contents
        st.header("🔍 Debug")
        if st.button("Show Database Schema"):
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                st.code("\n\n".join([t[0] for t in tables if t[0]]), language="sql")
                
                # Show sample data
                cursor.execute("SELECT * FROM documents LIMIT 5")
                rows = cursor.fetchall()
                
                st.write("**Sample Data:**")
                for row in rows:
                    st.write(row)
                
                conn.close()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "💬 Query Data", "📊 Results & History"])
    
    # Tab 1: Upload and Process
    with tab1:
        st.header("Step 1: Upload Invoice Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an invoice image",
                type=['png', 'jpg', 'jpeg', 'pdf'],
                help="Upload an invoice image for OCR processing"
            )
            
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Invoice", use_container_width=True)
                
                # Check if this file has been processed
                file_processed = (st.session_state.ocr_result is not None and 
                                st.session_state.get('last_processed_file') == uploaded_file.name)
                
                if file_processed:
                    st.success("✅ This invoice has been processed and saved to database!")
                    if st.button("🔄 Reprocess Invoice", use_container_width=True):
                        success, result, doc_id = process_image(uploaded_file)
                        
                        if success:
                            st.balloons()
                            st.markdown('<div class="success-box">✅ Invoice reprocessed successfully!</div>', unsafe_allow_html=True)
                            st.session_state.last_processed_file = uploaded_file.name
                            st.rerun()
                        else:
                            st.markdown(f'<div class="error-box">❌ Error: {result}</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Click the button below to process and save to database")
                    if st.button("🚀 Process Invoice", type="primary", use_container_width=True):
                        success, result, doc_id = process_image(uploaded_file)
                        
                        if success:
                            st.balloons()
                            st.markdown('<div class="success-box">✅ Invoice processed and saved to database!</div>', unsafe_allow_html=True)
                            st.session_state.last_processed_file = uploaded_file.name
                            st.rerun()
                        else:
                            st.markdown(f'<div class="error-box">❌ Error: {result}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.ocr_result:
                st.subheader("📋 Extracted Data")
                st.success("✅ Data saved to database - Ready for querying!")
                
                ocr_data = st.session_state.ocr_result
                
                # Display key information
                if "vendor" in ocr_data:
                    st.metric("Vendor", ocr_data['vendor'].get('name', 'N/A'))
                
                if "order_details" in ocr_data:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Invoice #", ocr_data['order_details'].get('invoice_number', 'N/A'))
                    with col_b:
                        st.metric("Date", ocr_data['order_details'].get('invoice_date', 'N/A'))
                
                if "payment_details" in ocr_data:
                    st.metric("Total Amount", ocr_data['payment_details'].get('total', 'N/A'))
                
                st.info(f"💾 Saved to database with ID: {st.session_state.doc_id}")
                
                # Add quick action button
                st.divider()
                if st.button("➡️ Go to Query Data Tab", use_container_width=True, type="primary"):
                    st.session_state.active_tab = 1
                    st.info("Switch to the 'Query Data' tab above to start asking questions!")
                
                # Show full JSON
                with st.expander("🔍 View Full OCR Data"):
                    st.json(ocr_data)
            else:
                st.info("👆 Upload and process an invoice to see extracted data")
                st.markdown("""
                **Instructions:**
                1. Click "Browse files" or drag & drop an invoice image
                2. Click "🚀 Process Invoice" button to extract data
                3. Data will be saved to database automatically
                4. Go to "Query Data" tab to ask questions
                """)
    
    # Tab 2: Query Data
    with tab2:
        st.header("Step 2: Query Your Data with Natural Language")
        
        if not st.session_state.db_initialized:
            st.error("⚠️ **No data in database yet!**")
            st.markdown("""
            ### Please follow these steps:
            1. Go to **"Upload & Process"** tab (first tab)
            2. Upload an invoice image
            3. Click **"🚀 Process Invoice"** button
            4. Come back here to query the data
            
            The database is initialized but empty until you process an invoice.
            """)
            st.stop()
        
        # Show data status
        if st.session_state.ocr_result:
            st.success(f"✅ Database contains invoice data (Doc ID: {st.session_state.doc_id})")
        else:
            st.warning("⚠️ Database initialized but no invoice processed yet. Please upload and process an invoice first.")
        
        # Add debug section at the top
        with st.expander("🔍 Debug: Show Database Contents", expanded=True):
            col_debug1, col_debug2, col_debug3 = st.columns(3)
            
            with col_debug1:
                if st.button("Show Database Schema", use_container_width=True):
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        
                        st.subheader("Table Structure:")
                        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
                        tables = cursor.fetchall()
                        
                        for table in tables:
                            if table[0]:
                                st.code(table[0], language="sql")
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_debug2:
                if st.button("Show All Data", use_container_width=True):
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        
                        st.subheader("Documents Table:")
                        cursor.execute("SELECT COUNT(*) FROM documents")
                        doc_count = cursor.fetchone()[0]
                        st.write(f"**Total documents: {doc_count}**")
                        
                        if doc_count > 0:
                            cursor.execute("SELECT * FROM documents")
                            rows = cursor.fetchall()
                            
                            for i, row in enumerate(rows, 1):
                                st.write(f"**Record {i}:**")
                                st.write(f"- ID: {row[0]}")
                                st.write(f"- Vendor: {row[1]}")
                                st.write(f"- Invoice #: {row[5]}")
                                st.write(f"- Total: {row[11]}")
                                st.write(f"- Date: {row[6]}")
                                with st.expander("Full Record"):
                                    st.write(row)
                        else:
                            st.error("❌ **No data in documents table!**")
                        
                        st.subheader("Items Table:")
                        cursor.execute("SELECT COUNT(*) FROM items")
                        items_count = cursor.fetchone()[0]
                        st.write(f"**Total items: {items_count}**")
                        
                        if items_count > 0:
                            cursor.execute("SELECT * FROM items")
                            items = cursor.fetchall()
                            for item in items:
                                st.write(item)
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_debug3:
                if st.button("Manual Insert Test", use_container_width=True):
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        
                        # Insert test data
                        cursor.execute("""
                            INSERT INTO documents (
                                vendor_name, invoice_number, invoice_date, 
                                subtotal, tax, total, currency
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, ("Test Vendor", "TEST-001", "2024-01-15", 100.0, 10.0, 110.0, "USD"))
                        
                        conn.commit()
                        st.success("✅ Test data inserted!")
                        
                        # Verify
                        cursor.execute("SELECT COUNT(*) FROM documents")
                        count = cursor.fetchone()[0]
                        st.write(f"Documents in DB: {count}")
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.divider()
        
        # Example questions
        st.markdown("**Example Questions:**")
        examples = [
            "Show all invoices",
            "What is the total amount?",
            "List all items",
            "Show invoices from specific vendor"
        ]
        
        cols = st.columns(len(examples))
        for idx, (col, example) in enumerate(zip(cols, examples)):
            with col:
                if st.button(example, key=f"ex_{idx}", use_container_width=True):
                    st.session_state.current_question = example
        
        st.divider()
        
        # Query input
        question = st.text_input(
            "Ask a question about your invoice data:",
            value=st.session_state.get('current_question', ''),
            placeholder="e.g., Show all invoices with total greater than 1000",
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.button("🔍 Execute Query", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.current_question = ''
            st.rerun()
        
        if submit_button and question:
            success, sql, results = execute_nl_query(question)
            
            if success:
                st.success("✅ Query executed successfully!")
                
                st.subheader("Generated SQL")
                st.code(sql, language="sql")
                
                st.subheader("Results")
                if results:
                    # Display results in a nice format
                    for i, row in enumerate(results, 1):
                        st.write(f"**{i}.** {row}")
                else:
                    st.info("No results found")
            else:
                st.error(f"❌ Error: {sql}")
    
    # Tab 3: Results and History
    with tab3:
        st.header("📊 Query Results & History")
        
        if st.session_state.query_history:
            for idx, entry in enumerate(reversed(st.session_state.query_history), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - idx + 1}: {entry['question']}", expanded=(idx==1)):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.code(entry['sql'], language="sql")
                    
                    st.markdown("**Results:**")
                    if entry['results']:
                        for i, row in enumerate(entry['results'], 1):
                            st.write(f"{i}. {row}")
                    else:
                        st.info("No results")
                    
                    st.divider()
        else:
            st.info("No query history yet. Go to the 'Query Data' tab to start asking questions!")

if __name__ == "__main__":
    main()
