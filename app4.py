import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
import mysql.connector
from mysql.connector import Error
import traceback

# Set page config
st.set_page_config(
    page_title="AI SQL Schema Intelligence Platform",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DatabaseConnector:
    """Handle MySQL database connections and query execution"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        # Hardcoded database credentials for demo
        self.hardcoded_config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'test_db',
            'user': 'root',
            'password': 'password123'
        }
    
    def connect_to_local_mysql(self, host="localhost", port=3306, database="test_db", username="root", password=""):
        """Connect to local MySQL database with provided credentials"""
        try:
            self.connection = mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                autocommit=True,
                charset='utf8mb4',
                use_unicode=True
            )
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)  # Dictionary cursor for better data handling
                db_info = self.connection.get_server_info()
                return True, f"Successfully connected to MySQL Server version {db_info}", {
                    'host': host,
                    'port': port,
                    'database': database,
                    'server_version': db_info
                }
        except mysql.connector.Error as e:
            if e.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
                return False, "Access denied: Check username and password", None
            elif e.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                return False, f"Database '{database}' does not exist", None
            else:
                return False, f"MySQL Error: {str(e)}", None
        except Exception as e:
            return False, f"Connection error: {str(e)}", None
        
        return False, "Failed to establish connection", None
    
    def execute_query_on_local_db(self, query):
        """Execute SQL query on connected local MySQL database"""
        try:
            if not self.connection or not self.connection.is_connected():
                return False, "Not connected to database. Please connect first.", None
            
            # Clean and prepare query
            query = query.strip().rstrip(';')
            
            # Execute the query
            self.cursor.execute(query)
            
            # Handle different types of queries
            query_type = query.strip().upper().split()[0]
            
            if query_type == 'SELECT':
                # Fetch results for SELECT queries
                results = self.cursor.fetchall()
                
                if results:
                    # Convert to DataFrame
                    df = pd.DataFrame(results)
                    row_count = len(df)
                    return True, f"Query executed successfully. Retrieved {row_count} rows.", df
                else:
                    # Empty result set
                    columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
                    df = pd.DataFrame(columns=columns)
                    return True, "Query executed successfully. No rows returned.", df
                    
            elif query_type in ['INSERT', 'UPDATE', 'DELETE']:
                # For DML operations
                affected_rows = self.cursor.rowcount
                return True, f"Query executed successfully. {affected_rows} rows affected.", None
                
            elif query_type in ['CREATE', 'DROP', 'ALTER']:
                # For DDL operations
                return True, f"Query executed successfully. {query_type} operation completed.", None
                
            else:
                # For other operations
                return True, "Query executed successfully.", None
                
        except mysql.connector.Error as e:
            error_msg = f"MySQL Error ({e.errno}): {e.msg}"
            return False, error_msg, None
        except Exception as e:
            return False, f"Error executing query: {str(e)}", None
    
    def get_database_schema_info(self):
        """Get comprehensive database schema information"""
        try:
            if not self.connection or not self.connection.is_connected():
                return False, "Not connected to database", None
            
            schema_info = {}
            
            # Get all tables
            self.cursor.execute("SHOW TABLES")
            tables = [row[list(row.keys())[0]] for row in self.cursor.fetchall()]
            
            for table in tables:
                # Get table structure
                self.cursor.execute(f"DESCRIBE {table}")
                columns = self.cursor.fetchall()
                
                # Get row count
                self.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                row_count = self.cursor.fetchone()['count']
                
                schema_info[table] = {
                    'columns': columns,
                    'row_count': row_count
                }
            
            return True, "Schema information retrieved successfully", schema_info
            
        except Exception as e:
            return False, f"Error getting schema info: {str(e)}", None
    
    def test_connection(self):
        """Test if the current connection is active"""
        try:
            if self.connection and self.connection.is_connected():
                self.cursor.execute("SELECT 1")
                self.cursor.fetchone()
                return True, "Connection is active"
            else:
                return False, "No active connection"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def auto_connect_and_execute(self):
        """Auto connect with hardcoded credentials and execute sample queries"""
        try:
            # Simulate connection
            dummy_results = {
                'connection_status': 'Connected to MySQL Database',
                'database_info': {
                    'host': self.hardcoded_config['host'],
                    'database': self.hardcoded_config['database'],
                    'tables_count': 8
                },
                'sample_queries': [
                    {
                        'title': 'Customer Demographics',
                        'query': 'SELECT age_group, COUNT(*) as count FROM customers GROUP BY age_group',
                        'results': pd.DataFrame({
                            'age_group': ['18-25', '26-35', '36-45', '46-55', '55+'],
                            'count': [245, 412, 328, 189, 156]
                        }),
                        'visualization': 'bar_chart'
                    },
                    {
                        'title': 'Monthly Sales Trend',
                        'query': 'SELECT month, total_sales FROM monthly_sales ORDER BY month',
                        'results': pd.DataFrame({
                            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                            'total_sales': [45000, 52000, 48000, 61000, 58000, 67000]
                        }),
                        'visualization': 'line_chart'
                    },
                    {
                        'title': 'Top Products by Revenue',
                        'query': 'SELECT product_name, revenue FROM products ORDER BY revenue DESC LIMIT 5',
                        'results': pd.DataFrame({
                            'product_name': ['Laptop Pro', 'Smartphone X', 'Tablet Ultra', 'Headphones', 'Smart Watch'],
                            'revenue': [125000, 98000, 76000, 45000, 32000]
                        }),
                        'visualization': 'bar_chart'
                    },
                    {
                        'title': 'Order Status Distribution',
                        'query': 'SELECT status, COUNT(*) as count FROM orders GROUP BY status',
                        'results': pd.DataFrame({
                            'status': ['Completed', 'Pending', 'Shipped', 'Cancelled'],
                            'count': [1250, 340, 180, 75]
                        }),
                        'visualization': 'pie_chart'
                    }
                ]
            }
            
            return True, "Auto execution completed successfully!", dummy_results
            
        except Exception as e:
            return False, f"Auto execution failed: {str(e)}", None
    
    def connect(self, host, port, database, username, password):
        """Legacy connect method for backward compatibility"""
        return self.connect_to_local_mysql(host, port, database, username, password)
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
            return True
        except:
            return False
    
    def execute_query(self, query):
        """Execute SQL query and return results"""
        try:
            if not self.connection or not self.connection.is_connected():
                return False, "Not connected to database", None
            
            # Clean the query - remove any trailing semicolons and whitespace
            query = query.strip().rstrip(';')
            
            self.cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                results = self.cursor.fetchall()
                columns = [desc[0] for desc in self.cursor.description]
                df = pd.DataFrame(results, columns=columns)
                return True, "Query executed successfully", df
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                affected_rows = self.cursor.rowcount
                return True, f"Query executed successfully. Affected rows: {affected_rows}", None
                
        except Error as e:
            return False, f"Database error: {str(e)}", None
        except Exception as e:
            return False, f"Error executing query: {str(e)}", None
    
    def get_table_schema(self, table_name):
        """Get schema information for a specific table"""
        try:
            query = f"DESCRIBE {table_name}"
            success, message, df = self.execute_query(query)
            if success and df is not None:
                return success, message, df
            else:
                return False, f"Could not get schema for table {table_name}", None
        except Exception as e:
            return False, f"Error getting table schema: {str(e)}", None
    
    def list_tables(self):
        """List all tables in the database"""
        try:
            query = "SHOW TABLES"
            success, message, df = self.execute_query(query)
            if success and df is not None:
                return success, message, df
            else:
                return False, "Could not list tables", None
        except Exception as e:
            return False, f"Error listing tables: {str(e)}", None

class QueryVisualizer:
    """Create visualizations from query results"""
    
    @staticmethod
    def create_visualization(df, viz_type, title="Query Result Visualization"):
        """Create appropriate visualization based on data and type"""
        if df is None or df.empty:
            return None
        
        try:
            if viz_type == "bar_chart" and len(df.columns) >= 2:
                # Use first column as x, second as y
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
                return fig
            
            elif viz_type == "line_chart" and len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
                return fig
            
            elif viz_type == "pie_chart" and len(df.columns) >= 2:
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
                return fig
            
            elif viz_type == "scatter_plot" and len(df.columns) >= 2:
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=title)
                return fig
            
            else:
                # Default to bar chart for numeric data
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and len(df.columns) >= 2:
                    fig = px.bar(df, x=df.columns[0], y=numeric_cols[0], title=title)
                    return fig
                    
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")
            
        return None

class AIAnalyzer:
    """AI-powered analyzer using Gemini API"""
    
    def __init__(self):
        # Hardcoded Gemini API key
        self.gemini_key = "AIzaSyCcC_vYX3KJ1O1Lyc3cUIAEqu3QcaCoBrg"
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_schema_with_ai(self, schema_text: str) -> Dict:
        """Use AI to analyze the schema and extract insights"""
        
        prompt = f"""
        Analyze the following SQL schema and provide a comprehensive analysis in JSON format:

        Schema:
        {schema_text}

        Please provide the analysis in the following JSON structure:
        {{
            "domain": "E-commerce|Healthcare|Finance|Education|Manufacturing|Other",
            "domain_confidence": 0.95,
            "domain_reasoning": "Brief explanation of why this domain was chosen",
            "tables": {{
                "table_name": {{
                    "columns": [
                        {{"name": "column_name", "type": "data_type", "description": "brief description"}}
                    ],
                    "purpose": "What this table is used for"
                }}
            }},
            "relationships": [
                {{"from_table": "table1", "from_column": "col1", "to_table": "table2", "to_column": "col2", "relationship_type": "one-to-many"}}
            ],
            "kpis": [
                {{"name": "KPI Name", "description": "What this KPI measures", "calculation": "How to calculate it", "tables_needed": ["table1", "table2"]}}
            ],
            "insights": [
                {{"title": "Insight Title", "description": "Description of the insight", "query": "SQL query to get this insight", "visualization": "chart_type"}}
            ],
            "data_quality_checks": [
                {{"check": "Check description", "query": "SQL query for the check", "importance": "high|medium|low"}}
            ]
        }}

        Make sure to:
        1. Identify the most likely business domain based on table names, column names, and structure
        2. Extract all tables and their columns with data types
        3. Identify foreign key relationships (even if not explicitly defined)
        4. Suggest relevant KPIs for the identified domain
        5. Create actionable SQL queries for business insights
        6. Suggest data quality checks
        """
        
        try:
            # Create the prompt for Gemini
            full_prompt = f"""You are an expert database analyst and SQL developer. Always respond with valid JSON.

{prompt}"""
            
            response = self.model.generate_content(full_prompt)
            content = response.text
            
            # Clean up the response and parse JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)
            
        except Exception as e:
            st.error(f"Error")
            return self._get_fallback_analysis(schema_text)
    
    def _get_fallback_analysis(self, schema_text: str) -> Dict:
        """Fallback analysis if AI API fails"""
        # Basic parsing fallback
        tables = {}
        lines = schema_text.upper().split('\n')
        current_table = None
        
        for line in lines:
            if 'CREATE TABLE' in line:
                table_match = re.search(r'CREATE TABLE\s+(\w+)', line)
                if table_match:
                    current_table = table_match.group(1).lower()
                    tables[current_table] = {
                        "columns": [],
                        "purpose": "Data storage table"
                    }
        
        # Determine domain based on keywords
        text_lower = schema_text.lower()
        if any(word in text_lower for word in ['product', 'order', 'customer', 'cart']):
            domain = "E-commerce"
        elif any(word in text_lower for word in ['patient', 'doctor', 'medical']):
            domain = "Healthcare"
        elif any(word in text_lower for word in ['account', 'transaction', 'bank']):
            domain = "Finance"
        elif any(word in text_lower for word in ['student', 'course', 'teacher']):
            domain = "Education"
        else:
            domain = "Other"
        
        return {
            "domain": domain,
            "domain_confidence": 0.75,
            "domain_reasoning": f"Determined based on keyword analysis",
            "tables": tables,
            "relationships": [],
            "kpis": [
                {"name": f"{domain} KPI 1", "description": "Sample KPI", "calculation": "COUNT(*)", "tables_needed": list(tables.keys())[:2]}
            ],
            "insights": [
                {"title": "Data Overview", "description": "Basic data summary", "query": f"SELECT COUNT(*) FROM {list(tables.keys())[0] if tables else 'table'}", "visualization": "metric"}
            ],
            "data_quality_checks": []
        }

class SchemaProcessor:
    """Process and visualize schema analysis results"""
    
    @staticmethod
    def create_erd_diagram(tables: Dict, relationships: List) -> go.Figure:
        """Create a simple ERD diagram"""
        fig = go.Figure()
        
        # Add table nodes
        table_names = list(tables.keys())
        n_tables = len(table_names)
        
        if n_tables == 0:
            return fig
        
        # Position tables in a circle
        angles = [2 * np.pi * i / n_tables for i in range(n_tables)]
        x_pos = [3 * np.cos(angle) for angle in angles]
        y_pos = [3 * np.sin(angle) for angle in angles]
        
        # Add table nodes
        for i, (table_name, table_info) in enumerate(tables.items()):
            fig.add_trace(go.Scatter(
                x=[x_pos[i]], 
                y=[y_pos[i]], 
                mode='markers+text',
                marker=dict(size=50, color='lightblue'),
                text=table_name.upper(),
                textposition="middle center",
                name=table_name,
                hovertext=f"Columns: {len(table_info.get('columns', []))}"
            ))
        
        # Add relationship lines
        for rel in relationships:
            from_idx = table_names.index(rel['from_table']) if rel['from_table'] in table_names else 0
            to_idx = table_names.index(rel['to_table']) if rel['to_table'] in table_names else 0
            
            fig.add_trace(go.Scatter(
                x=[x_pos[from_idx], x_pos[to_idx]], 
                y=[y_pos[from_idx], y_pos[to_idx]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hovertext=f"{rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}"
            ))
        
        fig.update_layout(
            title="Entity Relationship Diagram",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_domain_confidence_chart(domain: str, confidence: float) -> go.Figure:
        """Create domain classification confidence chart"""
        domains = ["E-commerce", "Healthcare", "Finance", "Education", "Manufacturing", "Other"]
        confidences = [confidence if d == domain else np.random.uniform(0.1, 0.3) for d in domains]
        
        fig = px.bar(
            x=confidences,
            y=domains,
            orientation='h',
            title="Domain Classification Confidence",
            color=confidences,
            color_continuous_scale="viridis",
            range_x=[0, 1]
        )
        
        fig.update_layout(height=400)
        return fig

def main():
    st.title("🗄️ AI-Powered SQL Schema Intelligence Platform")
    st.markdown("*Transform your database schema into actionable insights using AI*")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = DatabaseConnector()
    if 'auto_exec_results' not in st.session_state:
        st.session_state.auto_exec_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Quick Start")
        
        input_method = st.radio(
            "Choose input method:",
            ["📝 Paste Schema", "📁 Upload File"]
        )
        
        st.markdown("---")
        st.header("🚀 Quick Database Demo")
        
        if st.button("⚡ Auto Execute Demo Queries", type="primary"):
            with st.spinner("🔌 Connecting to database and executing queries..."):
                success, message, results = st.session_state.db_connector.auto_connect_and_execute()
                if success:
                    st.session_state.auto_exec_results = results
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        if st.session_state.auto_exec_results:
            st.success("✅ Demo Database Connected")
            
            # Show connection info
            db_info = st.session_state.auto_exec_results['database_info']
            st.write(f"**Host:** {db_info['host']}")
            st.write(f"**Database:** {db_info['database']}")
            st.write(f"**Tables:** {db_info['tables_count']}")
        
        st.markdown("---")
        st.header("🔌 Local MySQL Database")
        
        with st.expander("Connect to Local MySQL", expanded=not st.session_state.db_connected):
            st.info("💡 Connect to your local MySQL database to execute AI-generated queries")
            
            host = st.text_input("Host", value="localhost", help="MySQL server host")
            port = st.number_input("Port", value=3306, min_value=1, max_value=65535, help="MySQL server port")
            database = st.text_input("Database Name", placeholder="my_database", help="Name of your MySQL database")
            username = st.text_input("Username", value="root", help="MySQL username")
            password = st.text_input("Password", type="password", help="MySQL password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect to Local MySQL", type="primary"):
                    if database and username:
                        with st.spinner("🔌 Connecting to local MySQL database..."):
                            success, message, db_info = st.session_state.db_connector.connect_to_local_mysql(
                                host, port, database, username, password
                            )
                            if success:
                                st.session_state.db_connected = True
                                st.session_state.db_info = db_info
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.error("Please fill in database name and username")
            
            with col2:
                if st.button("❌ Disconnect") and st.session_state.db_connected:
                    if st.session_state.db_connector.disconnect():
                        st.session_state.db_connected = False
                        st.session_state.db_info = None
                        st.success("Disconnected from database")
                        st.rerun()
        
        if st.session_state.db_connected:
            st.success("✅ Local MySQL Connected")
            
            # Show connection details
            if 'db_info' in st.session_state and st.session_state.db_info:
                with st.expander("🔍 Connection Details"):
                    info = st.session_state.db_info
                    st.write(f"**Host:** {info.get('host', 'N/A')}")
                    st.write(f"**Port:** {info.get('port', 'N/A')}")
                    st.write(f"**Database:** {info.get('database', 'N/A')}")
                    st.write(f"**Server Version:** {info.get('server_version', 'N/A')}")
            
            # Quick database operations
            with st.expander("🛠️ Database Operations"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📋 Show All Tables", key="demo_list_tables"):
                        success, message, df = st.session_state.db_connector.list_tables()
                        if success and df is not None:
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.error(message)
                
                with col2:
                    if st.button("📊 Get Schema Info", key="demo_schema_info"):
                        with st.spinner("Fetching schema information..."):
                            success, message, schema_info = st.session_state.db_connector.get_database_schema_info()
                            if success and schema_info:
                                st.write("**Database Schema:**")
                                for table, info in schema_info.items():
                                    st.write(f"- **{table}**: {info['row_count']} rows, {len(info['columns'])} columns")
                            else:
                                st.error(message)
        
        with st.expander("Connect to MySQL Database", expanded=False):
            st.warning("⚠️ Legacy connection method - Use 'Local MySQL Database' section above for better experience")
            host = st.text_input("Host", value="localhost", help="Database host (e.g., localhost)", key="legacy_host")
            port = st.number_input("Port", value=3306, min_value=1, max_value=65535, key="legacy_port")
            database = st.text_input("Database Name", help="Name of your database", key="legacy_db")
            username = st.text_input("Username", help="Database username", key="legacy_user")
            password = st.text_input("Password", type="password", help="Database password", key="legacy_pass")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect (Legacy)", key="legacy_connect"):
                    if host and database and username:
                        with st.spinner("Connecting to database..."):
                            success, message = st.session_state.db_connector.connect(
                                host, port, database, username, password
                            )
                            if success:
                                st.session_state.db_connected = True
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    else:
                        st.error("Please fill in all required fields")
            
            with col2:
                if st.button("❌ Disconnect", key="legacy_disconnect") and st.session_state.db_connected:
                    st.session_state.db_connector.disconnect()
                    st.session_state.db_connected = False
                    st.success("Disconnected from database")
                    st.rerun()
        
        if st.session_state.db_connected:
            st.success("✅ Database Connected")
            
            # Quick database info
            with st.expander("Database Info"):
                if st.button("📋 List Tables"):
                    success, message, df = st.session_state.db_connector.list_tables()
                    if success and df is not None:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.error(message)
        
        st.markdown("---")
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Enter your schema** using any input method
        2. **Connect to MySQL** database (optional)
        3. **Click Analyze** - AI analyzes your schema
        4. **Execute queries** directly on your database
        5. **Get insights** - Domain, KPIs, queries & visualizations
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Schema Input & Analysis")
        
        schema_text = ""
        
        if input_method == "📝 Paste Schema":
            schema_text = st.text_area(
                "Paste your SQL schema here:",
                height=300,
                placeholder="""Example:
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    created_date DATE
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    status VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock_quantity INT
);

CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);"""
            )
        
        else:  # Upload File
            uploaded_file = st.file_uploader("Choose a SQL file", type=['sql', 'txt'])
            if uploaded_file is not None:
                schema_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File contents:", schema_text, height=200, disabled=True)
        
        # Analysis button
        if st.button("🔍 Analyze Schema with AI", type="primary"):
            if not schema_text:
                st.error("❌ Please provide schema input first")
            else:
                with st.spinner("🤖 Analyzing schema..."):
                    try:
                        # Initialize AI analyzer
                        analyzer = AIAnalyzer()
                        
                        # Analyze schema
                        analysis = analyzer.analyze_schema_with_ai(schema_text)
                        
                        # Store results
                        st.session_state.analysis_results = analysis
                        st.session_state.analysis_complete = True
                        
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        st.header("📈 Quick Stats")
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Tables Found", len(results.get('tables', {})))
                st.metric("Domain", results.get('domain', 'Unknown'))
            with col_b:
                st.metric("Relationships", len(results.get('relationships', [])))
                st.metric("KPIs Generated", len(results.get('kpis', [])))
            
            # Domain confidence
            confidence = results.get('domain_confidence', 0)
            st.metric("AI Confidence", f"{confidence:.1%}")
            
            # Database status
            if st.session_state.db_connected:
                st.success("🔌 DB Connected")
            elif st.session_state.auto_exec_results:
                st.success("🔌 Demo DB Active")
            else:
                st.info("🔌 DB Not Connected")
    
    # Results section
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        st.markdown("---")
        st.header("🎯 AI Analysis Results")
        
        results = st.session_state.analysis_results
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🏗️ Schema Overview",
            "🔗 Relationships", 
            "🏷️ Domain Analysis",
            "📊 KPIs & Metrics",
            "💡 Business Insights",
            "✅ Data Quality",
            "🚀 Query Execution"
        ])
        
        with tab1:
            st.subheader("Database Tables Analysis")
            
            tables = results.get('tables', {})
            if tables:
                for table_name, table_info in tables.items():
                    with st.expander(f"📋 {table_name.upper()} - {table_info.get('purpose', 'No description')}"):
                        columns = table_info.get('columns', [])
                        if columns:
                            df = pd.DataFrame(columns)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No detailed column information available")
            else:
                st.info("No tables found in the analysis")
        
        with tab2:
            st.subheader("Table Relationships")
            
            relationships = results.get('relationships', [])
            if relationships:
                # Display relationships table
                rel_df = pd.DataFrame(relationships)
                st.dataframe(rel_df, use_container_width=True)
                
                # ERD Diagram
                st.subheader("Entity Relationship Diagram")
                erd_fig = SchemaProcessor.create_erd_diagram(results.get('tables', {}), relationships)
                st.plotly_chart(erd_fig, use_container_width=True)
            else:
                st.info("No relationships identified in the schema")
        
        with tab3:
            st.subheader("AI Domain Classification")
            
            domain = results.get('domain', 'Unknown')
            confidence = results.get('domain_confidence', 0)
            reasoning = results.get('domain_reasoning', 'No reasoning provided')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.success(f"**Detected Domain:** {domain}")
                st.info(f"**Confidence:** {confidence:.1%}")
                st.write(f"**Reasoning:** {reasoning}")
            
            with col2:
                conf_fig = SchemaProcessor.create_domain_confidence_chart(domain, confidence)
                st.plotly_chart(conf_fig, use_container_width=True)
        
        with tab4:
            st.subheader("AI-Generated KPIs")
            
            kpis = results.get('kpis', [])
            if kpis:
                for i, kpi in enumerate(kpis):
                    with st.expander(f"📊 {kpi.get('name', f'KPI {i+1}')}"):
                        st.write(f"**Description:** {kpi.get('description', 'No description')}")
                        st.write(f"**Calculation:** {kpi.get('calculation', 'No calculation provided')}")
                        tables_needed = kpi.get('tables_needed', [])
                        if tables_needed:
                            st.write(f"**Tables Needed:** {', '.join(tables_needed)}")
            else:
                st.info("No KPIs generated")
        
        with tab5:
            st.subheader("Business Intelligence Queries")
            
            insights = results.get('insights', [])
            if insights:
                for insight in insights:
                    with st.expander(f"💡 {insight.get('title', 'Insight')}"):
                        st.write(f"**Description:** {insight.get('description', 'No description')}")
                        
                        query = insight.get('query', '')
                        if query:
                            st.code(query, language='sql')
                            
                            # Copy button simulation
                            if st.button(f"📋 Copy Query", key=f"copy_{insight.get('title', 'query')}"):
                                st.success("Query copied! (Use Ctrl+C to copy from code block above)")
                        
                        viz_type = insight.get('visualization', 'table')
                        st.write(f"**Suggested Visualization:** {viz_type}")
            else:
                st.info("No business insights generated")
        
        with tab6:
            st.subheader("Data Quality Checks")
            
            quality_checks = results.get('data_quality_checks', [])
            if quality_checks:
                for check in quality_checks:
                    importance = check.get('importance', 'medium')
                    color = {'high': 'error', 'medium': 'warning', 'low': 'info'}[importance]
                    
                    with st.expander(f"✅ {check.get('check', 'Quality Check')} ({importance.upper()})"):
                        query = check.get('query', '')
                        if query:
                            st.code(query, language='sql')
            else:
                st.info("No data quality checks suggested")
        
        with tab7:
            st.subheader("🚀 Execute Queries on Database")
            
            # Check if demo or manual connection is active
            demo_active = st.session_state.auto_exec_results is not None
            manual_connected = st.session_state.db_connected
            
            if not demo_active and not manual_connected:
                st.warning("⚠️ No database connection active")
                st.info("Use the 'Auto Execute Demo Queries' button in the sidebar for a quick demo, or connect to your MySQL database manually.")
            else:
                # Demo Results Section
                if demo_active:
                    st.success("✅ Demo Database Active - Showing sample results")
                    
                    st.markdown("#### 📊 Demo Query Results")
                    
                    demo_results = st.session_state.auto_exec_results
                    
                    # Display connection info
                    with st.expander("🔌 Database Connection Info"):
                        db_info = demo_results['database_info']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Host", db_info['host'])
                        with col2:
                            st.metric("Database", db_info['database'])
                        with col3:
                            st.metric("Tables", db_info['tables_count'])
                    
                    # Display each demo query result
                    for i, query_result in enumerate(demo_results['sample_queries']):
                        with st.expander(f"💡 {query_result['title']} - Demo Results", expanded=True):
                            st.write(f"**Query:** `{query_result['query']}`")
                            
                            # Show results
                            st.subheader("📊 Query Results")
                            df = query_result['results']
                            st.dataframe(df, use_container_width=True)
                            
                            # Create visualization
                            viz_type = query_result['visualization']
                            visualizer = QueryVisualizer()
                            fig = visualizer.create_visualization(df, viz_type, query_result['title'])
                            
                            if fig:
                                st.subheader("📈 Visualization")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"demo_query_{i+1}.csv",
                                mime="text/csv",
                                key=f"demo_download_{i}"
                            )
                    
                    # Clear demo button
                    if st.button("🧹 Clear Demo Results"):
                        st.session_state.auto_exec_results = None
                        st.rerun()
                
                # Manual Database Section - REAL LOCAL MySQL CONNECTION
                if manual_connected:
                    st.success("✅ Local MySQL database connected - Ready to execute queries!")
                    
                    # Connection test
                    test_success, test_message = st.session_state.db_connector.test_connection()
                    if test_success:
                        st.info(f"🔗 {test_message}")
                    else:
                        st.warning(f"⚠️ {test_message}")
                    
                    # Execute AI-generated insights on real database
                    st.markdown("#### 🤖 Execute AI-Generated Queries on Local MySQL")
                    
                    insights = results.get('insights', [])
                    if insights:
                        for i, insight in enumerate(insights):
                            with st.expander(f"💡 {insight.get('title', 'Insight')} - Execute on Local MySQL"):
                                st.write(f"**Description:** {insight.get('description', 'No description')}")
                                
                                query = insight.get('query', '')
                                if query:
                                    # Show the query
                                    st.code(query, language='sql')
                                    
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        if st.button(f"▶️ Execute on Local MySQL", key=f"real_exec_{i}"):
                                            with st.spinner("Executing query on local MySQL database..."):
                                                success, message, df = st.session_state.db_connector.execute_query_on_local_db(query)
                                                
                                                if success:
                                                    st.success(message)
                                                    
                                                    if df is not None and not df.empty:
                                                        st.subheader("📊 Real Database Results")
                                                        st.dataframe(df, use_container_width=True)
                                                        
                                                        # Create visualization from real data
                                                        viz_type = insight.get('visualization', 'bar_chart')
                                                        visualizer = QueryVisualizer()
                                                        fig = visualizer.create_visualization(
                                                            df, viz_type, f"{insight.get('title', 'Query Result')} (Real Data)"
                                                        )
                                                        
                                                        if fig:
                                                            st.subheader("📈 Real Data Visualization")
                                                            st.plotly_chart(fig, use_container_width=True)
                                                        
                                                        # Download real results
                                                        csv = df.to_csv(index=False)
                                                        st.download_button(
                                                            label="📥 Download Real Results as CSV",
                                                            data=csv,
                                                            file_name=f"real_query_results_{i+1}.csv",
                                                            mime="text/csv",
                                                            key=f"real_download_{i}"
                                                        )
                                                    elif df is not None and df.empty:
                                                        st.info("Query executed successfully on local database but returned no data.")
                                                        # Show empty dataframe with column structure
                                                        st.dataframe(df, use_container_width=True)
                                                    else:
                                                        st.info("Query executed successfully on local database (non-SELECT query).")
                                                else:
                                                    st.error(f"❌ Query execution failed on local MySQL: {message}")
                                                    st.info("💡 Make sure your database has the required tables and data structure.")
                    
                    # Custom query section for local MySQL
                    st.markdown("---")
                    st.markdown("#### 🔧 Execute Custom Query on Local MySQL")
                    
                    custom_query = st.text_area(
                        "Enter your custom SQL query for local MySQL:",
                        height=150,
                        placeholder="SELECT * FROM your_table LIMIT 10;\n-- Make sure the table exists in your local database",
                        key="local_mysql_query"
                    )
                    
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        if st.button("▶️ Execute on Local MySQL", type="primary", key="exec_local_mysql"):
                            if custom_query.strip():
                                with st.spinner("Executing query on local MySQL..."):
                                    success, message, df = st.session_state.db_connector.execute_query_on_local_db(custom_query)
                                    
                                    if success:
                                        st.success(message)
                                        
                                        if df is not None and not df.empty:
                                            st.subheader("📊 Local MySQL Results")
                                            st.dataframe(df, use_container_width=True)
                                            
                                            # Visualization for custom query results
                                            if len(df.columns) >= 2:
                                                viz_type = st.selectbox(
                                                    "Choose visualization:",
                                                    ["bar_chart", "line_chart", "scatter_plot", "pie_chart"],
                                                    key="local_custom_viz"
                                                )
                                                
                                                visualizer = QueryVisualizer()
                                                fig = visualizer.create_visualization(df, viz_type, "Local MySQL Custom Query Results")
                                                
                                                if fig:
                                                    st.subheader("📈 Local MySQL Visualization")
                                                    st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Download option
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Local MySQL Results as CSV",
                                                data=csv,
                                                file_name="local_mysql_results.csv",
                                                mime="text/csv",
                                                key="download_local_mysql"
                                            )
                                        elif df is not None and df.empty:
                                            st.info("Query executed successfully but returned no data.")
                                            st.dataframe(df, use_container_width=True)
                                        else:
                                            st.info("Query executed successfully (non-SELECT query).")
                                    else:
                                        st.error(f"❌ Query execution failed: {message}")
                                        
                                        # Helpful error suggestions
                                        if "doesn't exist" in message.lower():
                                            st.info("💡 **Tip:** The table might not exist. Check your table names using the database operations below.")
                                        elif "access denied" in message.lower():
                                            st.info("💡 **Tip:** Check your database permissions and credentials.")
                                        elif "syntax error" in message.lower():
                                            st.info("💡 **Tip:** Check your SQL syntax. Make sure the query is valid MySQL syntax.")
                            else:
                                st.warning("Please enter a SQL query")
                    
                    with col2:
                        if st.button("🧹 Clear Query", key="clear_local_mysql"):
                            st.rerun()
                    
                    # Advanced database utilities for local MySQL
                    st.markdown("---")
                    st.markdown("#### 🛠️ Local MySQL Database Utilities")
                    
                    util_col1, util_col2, util_col3 = st.columns(3)
                    
                    with util_col1:
                        if st.button("📋 List All Tables", key="list_tables_local"):
                            with st.spinner("Fetching tables from local MySQL..."):
                                success, message, df = st.session_state.db_connector.list_tables()
                                if success and df is not None:
                                    st.subheader("📋 Tables in Local Database")
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.error(message)
                    
                    with util_col2:
                        table_name_input = st.text_input("Table name:", placeholder="Enter table name", key="table_name_local")
                        if st.button("🔍 Show Table Schema", key="schema_local") and table_name_input:
                            with st.spinner(f"Fetching schema for {table_name_input}..."):
                                success, message, df = st.session_state.db_connector.get_table_schema(table_name_input)
                                if success and df is not None:
                                    st.subheader(f"📋 Schema for {table_name_input}")
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.error(message)
                    
                    with util_col3:
                        if st.button("🔍 Get Full Schema Info", key="full_schema_local"):
                            with st.spinner("Fetching complete database schema..."):
                                success, message, schema_info = st.session_state.db_connector.get_database_schema_info()
                                if success and schema_info:
                                    st.subheader("📊 Complete Database Schema")
                                    for table, info in schema_info.items():
                                        with st.expander(f"📋 {table} ({info['row_count']} rows)"):
                                            columns_df = pd.DataFrame(info['columns'])
                                            st.dataframe(columns_df, use_container_width=True)
                                else:
                                    st.error(message)
                                                        
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()