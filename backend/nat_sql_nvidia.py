import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

NL2SQL_PROMPT = """
You are an expert SQLite query generator.

Database schema:

documents(
    id INTEGER PRIMARY KEY,
    vendor_name TEXT,
    vendor_address TEXT,
    vendor_phone TEXT,
    vendor_email TEXT,
    invoice_number TEXT,
    invoice_date TEXT,
    due_date TEXT,
    po_number TEXT,
    subtotal REAL,
    tax REAL,
    total REAL,
    currency TEXT,
    raw_data TEXT,
    created_at TIMESTAMP
)

items(
    id INTEGER PRIMARY KEY,
    document_id INTEGER,  -- FOREIGN KEY to documents.id
    description TEXT,
    quantity INTEGER,
    unit_price REAL,
    amount REAL
)

IMPORTANT RELATIONSHIPS:
- items.document_id references documents.id
- To get items for an invoice, use: JOIN items ON documents.id = items.document_id
- Total amount is stored in documents.total (already calculated)
- For line items sum, use: SUM(items.amount)

COMMON QUERIES:
- "What is the total amount?" -> SELECT total FROM documents
- "Show all items" -> SELECT * FROM items JOIN documents ON items.document_id = documents.id
- "Total of all invoices" -> SELECT SUM(total) FROM documents

Rules:
- Generate ONLY one valid SQLite SELECT query
- Do NOT explain or add commentary
- Do NOT use markdown code blocks
- Use proper JOINs when accessing both tables
- For "the invoice" or "given invoice", query the most recent: ORDER BY created_at DESC LIMIT 1
- Return raw SQL query only
"""


class NL2SQLConverter:
    def __init__(self, api_key: str = "your_nvidia_api_key", model: str = "meta/llama-3.1-70b-instruct"):
        """
        Initialize NL2SQL converter with NVIDIA NIM
        
        Args:
            api_key: NVIDIA API key (if None, reads from NVIDIA_API_KEY env var)
            model: Model to use for SQL generation
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not found. "
                "Set NVIDIA_API_KEY environment variable or pass api_key parameter. "
                "Get key from: https://build.nvidia.com"
            )
        
        self.llm = ChatNVIDIA(
            model=model,
            api_key=self.api_key,
            temperature=0,
            max_tokens=200
        )
        
        print(f"✓ Initialized with model: {model}")
    
    def convert(self, question: str) -> str:
        """Convert natural language question to SQL"""
        try:
            messages = [
                SystemMessage(content=NL2SQL_PROMPT),
                HumanMessage(content=question)
            ]
            
            response = self.llm.invoke(messages)
            sql = response.content.strip()
            
            # Clean up markdown
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            # Basic validation
            if not sql.upper().startswith("SELECT"):
                raise ValueError(f"Generated query doesn't start with SELECT: {sql}")
            
            return sql
            
        except Exception as e:
            raise Exception(f"Failed to convert question to SQL: {str(e)}")


# Backward compatible function for existing code
def nl_to_sql(question: str) -> str:
    """
    Convert natural language to SQL
    Requires NVIDIA_API_KEY environment variable
    """
    converter = NL2SQLConverter()

    return converter.convert(question)
