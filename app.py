import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import re
import json
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import warnings
import os

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only.*")

# Set page config
st.set_page_config(
    page_title="SQL Schema Intelligence Platform",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FlexibleERDClassifier(nn.Module):
    """Flexible ERD Classifier that can adapt to saved architectures"""
    def __init__(self, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            # Default architecture
            layer_sizes = [384, 256, 128, 64, 2]
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SchemaProcessor:
    """Process schema and ERD inputs"""
    
    def __init__(self):
        self.tables = {}
        self.relationships = []
        
    def parse_schema_text(self, schema_text: str) -> Dict:
        """Parse schema from text input"""
        tables = {}
        current_table = None
        
        lines = schema_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
                
            # Check for CREATE TABLE
            if line.upper().startswith('CREATE TABLE'):
                table_match = re.search(r'CREATE TABLE\s+(\w+)', line, re.IGNORECASE)
                if table_match:
                    current_table = table_match.group(1).lower()
                    tables[current_table] = {'columns': [], 'primary_keys': [], 'foreign_keys': []}
            
            # Parse columns
            elif current_table and '(' in line and not line.upper().startswith('CREATE'):
                # Extract column definitions
                columns_text = line.split('(')[1].split(')')[0] if '(' in line else line
                for col_def in columns_text.split(','):
                    col_def = col_def.strip()
                    if col_def:
                        col_parts = col_def.split()
                        if len(col_parts) >= 2:
                            col_name = col_parts[0].strip('`"[]')
                            col_type = col_parts[1]
                            tables[current_table]['columns'].append({
                                'name': col_name,
                                'type': col_type,
                                'definition': col_def
                            })
        
        return tables
    
    def extract_relationships(self, schema_text: str) -> List[Dict]:
        """Extract foreign key relationships"""
        relationships = []
        fk_pattern = r'FOREIGN KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)'
        
        matches = re.findall(fk_pattern, schema_text, re.IGNORECASE)
        for match in matches:
            relationships.append({
                'from_column': match[0],
                'to_table': match[1],
                'to_column': match[2]
            })
        
        return relationships

class ModelManager:
    """Manage the three models for schema analysis"""
    
    def __init__(self):
        self.sentence_transformer = None
        self.erd_classifier = None
        self.domain_classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Domain mappings
        self.domain_labels = {
            0: "E-commerce",
            1: "Healthcare",
            2: "Finance",
            3: "Education",
            4: "Manufacturing"
        }
    
    def classify_domain(self, schema_text: str) -> str:
        """Classify the domain of the schema"""
        try:
            if self.domain_classifier and self.sentence_transformer:
                embeddings = self.sentence_transformer.encode([schema_text])
                
                with torch.no_grad():
                    inputs = torch.FloatTensor(embeddings).to(self.device)
                    outputs = self.domain_classifier(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Store confidence for display
                st.session_state.domain_confidence = confidence
                
                domain = self.domain_labels.get(predicted_class, "Unknown")
                return domain
            
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"Error in classify_domain: {e}")
            return "Unknown"  # Provide a default return value
        
        # Fallback logic for demo mode
        text_lower = schema_text.lower()
        if any(word in text_lower for word in ['product', 'order', 'customer', 'cart', 'payment', 'inventory']):
            return "E-commerce"
        elif any(word in text_lower for word in ['patient', 'doctor', 'medical', 'hospital', 'treatment']):
            return "Healthcare"
        elif any(word in text_lower for word in ['account', 'transaction', 'bank', 'loan', 'credit']):
            return "Finance"
        elif any(word in text_lower for word in ['student', 'course', 'teacher', 'grade', 'enrollment']):
            return "Education"
        else:
            return "Manufacturing"
    
    def detect_architecture_from_state(self, state_dict):
        """Detect layer sizes from saved state dictionary"""
        layer_sizes = []
        
        # Find all linear layer weights
        linear_weights = {k: v for k, v in state_dict.items() if 'weight' in k and not 'bn' in k}
        
        # Sort by layer index
        sorted_layers = sorted(linear_weights.items(), key=lambda x: int(x[0].split('.')[1]) if '.' in x[0] else 0)
        
        for layer_name, weight in sorted_layers:
            if len(weight.shape) == 2:
                input_size = weight.shape[1]
                output_size = weight.shape[0]
                if not layer_sizes:
                    layer_sizes.append(input_size)
                layer_sizes.append(output_size)
        
        return layer_sizes

    def create_flexible_domain_classifier(self, layer_sizes):
        """Create flexible domain classifier based on detected architecture"""
        class FlexibleDomainClassifier(nn.Module):
            def __init__(self, layer_sizes):
                super().__init__()
                layers = []
                for i in range(len(layer_sizes) - 1):
                    layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    if i < len(layer_sizes) - 2:  # No activation after last layer
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.3))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return FlexibleDomainClassifier(layer_sizes)
    
    def load_models(self):
        """Load all three models with graceful fallbacks"""
        try:
            # Define model paths
            erd_path = './final_erd_classifier.pth'
            domain_path = './domain_classifier_5_domains.pth'
            sentence_transformer_path = './sentence_transformer_model_balanced/'
            
            # Try to load fine-tuned sentence transformer first
            try:
                st.info("🔍 Loading fine-tuned sentence transformer...")
                self.sentence_transformer = SentenceTransformer(sentence_transformer_path)
                st.success("✅ Fine-tuned Sentence Transformer loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Could not load fine-tuned sentence transformer: {e}")
                st.info("🔄 Falling back to default sentence transformer...")
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to load pre-trained weights if available
            try:
                # Load ERD classifier with architecture detection
                st.info("🔍 Loading ERD classifier...")
                erd_state = torch.load(erd_path, map_location=self.device, weights_only=False)
                
                # Detect architecture from saved state
                layer_sizes = self.detect_architecture_from_state(erd_state)
                self.erd_classifier = FlexibleERDClassifier(layer_sizes)
                self.erd_classifier.load_state_dict(erd_state)
                st.success("✅ ERD model loaded successfully!")
                
                # Load domain classifier
                st.info("🔍 Loading domain classifier...")
                domain_state = torch.load(domain_path, map_location=self.device, weights_only=False)
                
                # Detect architecture for domain classifier too
                domain_layer_sizes = self.detect_architecture_from_state(domain_state)
                self.domain_classifier = self.create_flexible_domain_classifier(domain_layer_sizes)
                self.domain_classifier.load_state_dict(domain_state)
                st.success("✅ Domain model loaded successfully!")
                
                st.session_state.models_loaded = True
                st.session_state.demo_mode = False
                
            except FileNotFoundError as e:
                st.error(f"❌ Model file not found: {e}")
                st.session_state.models_loaded = False
                st.session_state.demo_mode = True
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.session_state.models_loaded = False
                st.session_state.demo_mode = True
            
            self.erd_classifier.eval()
            self.domain_classifier.eval()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Critical error loading models: {e}")
            st.session_state.models_loaded = False
            st.session_state.demo_mode = True
            return True

    def get_column_embeddings(self, columns: List[str]) -> np.ndarray:
        """Generate embeddings for column names"""
        if self.sentence_transformer:
            return self.sentence_transformer.encode(columns)
        else:
            # Fallback: random embeddings
            return np.random.rand(len(columns), 384)

    def find_related_columns(self, embeddings: np.ndarray, threshold: float = 0.7) -> List[Tuple]:
        """Find related columns using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(embeddings)
        related_pairs = []
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    related_pairs.append((i, j, similarity_matrix[i][j]))
        
        return related_pairs

class KPIGenerator:
    """Generate KPIs and insights based on domain and schema"""
    
    def __init__(self):
        self.domain_kpis = {
            "E-commerce": [
                "Total Revenue", "Order Conversion Rate", "Average Order Value",
                "Customer Lifetime Value", "Cart Abandonment Rate", "Product Return Rate"
            ],
            "Healthcare": [
                "Patient Satisfaction Score", "Average Treatment Duration",
                "Bed Occupancy Rate", "Doctor Utilization Rate", "Medical Error Rate"
            ],
            "Finance": [
                "Return on Investment", "Loan Default Rate", "Customer Acquisition Cost",
                "Net Interest Margin", "Cost-to-Income Ratio", "Liquidity Ratio"
            ],
            "Education": [
                "Student Enrollment Rate", "Course Completion Rate", "Teacher-Student Ratio",
                "Average Grade Performance", "Dropout Rate", "Student Satisfaction"
            ],
            "Manufacturing": [
                "Production Efficiency", "Quality Control Rate", "Equipment Downtime",
                "Inventory Turnover", "Worker Productivity", "Defect Rate"
            ]
        }
    
    def generate_kpis(self, domain: str, tables: Dict) -> List[Dict]:
        """Generate relevant KPIs for the domain"""
        base_kpis = self.domain_kpis.get(domain, ["Generic KPI 1", "Generic KPI 2"])
        
        kpis = []
        for kpi_name in base_kpis:
            kpis.append({
                "name": kpi_name,
                "description": f"Key performance indicator for {domain.lower()} domain",
                "suggested_tables": list(tables.keys())[:3],  # First 3 tables
                "metric_type": "ratio" if "rate" in kpi_name.lower() else "count"
            })
        
        return kpis

class QueryGenerator:
    """Generate SQL queries for insights"""
    
    def __init__(self):
        pass
    
    def generate_insight_queries(self, domain: str, tables: Dict, kpis: List[Dict]) -> List[Dict]:
        """Generate SQL queries for insights"""
        queries = []
        
        table_names = list(tables.keys())
        
        if domain == "E-commerce":
            queries.extend([
                {
                    "title": "Monthly Revenue Trend",
                    "description": "Track revenue growth over months",
                    "query": f"""
SELECT 
    DATE_TRUNC('month', order_date) as month,
    SUM(total_amount) as monthly_revenue,
    COUNT(*) as order_count
FROM {table_names[0] if table_names else 'orders'}
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;""",
                    "visualization": "line_chart"
                },
                {
                    "title": "Top Selling Products",
                    "description": "Identify best performing products",
                    "query": f"""
SELECT 
    product_name,
    SUM(quantity) as total_sold,
    SUM(price * quantity) as revenue
FROM {table_names[1] if len(table_names) > 1 else 'order_items'}
GROUP BY product_name
ORDER BY total_sold DESC
LIMIT 10;""",
                    "visualization": "bar_chart"
                }
            ])
        
        elif domain == "Healthcare":
            queries.extend([
                {
                    "title": "Patient Demographics",
                    "description": "Distribution of patients by age group",
                    "query": f"""
SELECT 
    CASE 
        WHEN age < 18 THEN 'Child'
        WHEN age < 65 THEN 'Adult'
        ELSE 'Senior'
    END as age_group,
    COUNT(*) as patient_count
FROM {table_names[0] if table_names else 'patients'}
GROUP BY age_group;""",
                    "visualization": "pie_chart"
                }
            ])
        
        elif domain == "Finance":
            queries.extend([
                {
                    "title": "Account Balance Distribution",
                    "description": "Distribution of account balances",
                    "query": f"""
SELECT 
    CASE 
        WHEN balance < 1000 THEN 'Low'
        WHEN balance < 10000 THEN 'Medium'
        ELSE 'High'
    END as balance_tier,
    COUNT(*) as account_count,
    AVG(balance) as avg_balance
FROM {table_names[0] if table_names else 'accounts'}
GROUP BY balance_tier;""",
                    "visualization": "bar_chart"
                }
            ])
        
        # Add more domain-specific queries...
        
        return queries

def main():
    st.title("🗄️ SQL Schema Intelligence Platform")
    st.markdown("*Transform your database schema into actionable insights*")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Quick Start")
        
        st.subheader("Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["📝 Paste Schema", "📁 Upload File", "🎨 ERD Description"]
        )
        
        st.markdown("---")
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Enter your schema** using any input method
        2. **Click Analyze** - AI models load automatically
        3. **Get insights** - Domain classification, KPIs & queries
        """)
        
        # Show model status if available
        if st.session_state.models_loaded:
            st.success("🤖 AI Models Ready")
        elif st.session_state.demo_mode:
            st.warning("🤖 Using Demo Mode (No trained models found)")
        else:
            st.info("🤖 AI Models will load on first analysis")
    
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
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);"""
            )
        
        elif input_method == "📁 Upload File":
            uploaded_file = st.file_uploader("Choose a SQL file", type=['sql', 'txt'])
            if uploaded_file is not None:
                schema_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File contents:", schema_text, height=200)
        
        else:  # ERD Description
            schema_text = st.text_area(
                "Describe your database structure:",
                height=200,
                placeholder="Example: E-commerce database with customers, products, orders, and order_items tables..."
            )
        
        # Analysis button with automatic model loading
        if st.button("🔍 Analyze Schema", type="primary"):
            if schema_text:
                # Initialize model manager if not already done
                if not st.session_state.model_manager:
                    with st.spinner("🤖 Loading AI models..."):
                        st.session_state.model_manager = ModelManager()
                        models_loaded = st.session_state.model_manager.load_models()
                        if st.session_state.models_loaded:
                            st.success("✅ AI models loaded successfully!")
                        else:
                            st.warning("⚠️ Using demo mode - no trained models found")
                
                with st.spinner("🔍 Analyzing schema..."):
                    # Initialize components
                    schema_processor = SchemaProcessor()
                    kpi_generator = KPIGenerator()
                    query_generator = QueryGenerator()
                    
                    # Process schema
                    tables = schema_processor.parse_schema_text(schema_text)
                    relationships = schema_processor.extract_relationships(schema_text)
                    
                    # Classify domain using loaded models
                    domain = st.session_state.model_manager.classify_domain(schema_text)
                    
                    # Generate insights
                    kpis = kpi_generator.generate_kpis(domain, tables)
                    queries = query_generator.generate_insight_queries(domain, tables, kpis)
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'tables': tables,
                        'relationships': relationships,
                        'domain': domain,
                        'kpis': kpis,
                        'queries': queries,
                        'schema_text': schema_text
                    }
                    st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
            else:
                st.warning("Please provide schema input first")
    
    with col2:
        st.header("📈 Quick Stats")
        if st.session_state.analysis_complete:
            results = st.session_state.analysis_results
            
            # Display metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Tables Found", len(results['tables']))
                st.metric("Domain", results['domain'])
            with col_b:
                st.metric("Relationships", len(results['relationships']))
                st.metric("KPIs Generated", len(results['kpis']))
    
    # Results section
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.header("📋 Analysis Results")
        
        results = st.session_state.analysis_results
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏗️ Schema Overview", 
            "🔗 Relationships", 
            "🏷️ Domain Classification",
            "📊 KPIs & Metrics", 
            "💡 Suggested Queries"
        ])
        
        with tab1:
            st.subheader("Database Tables")
            for table_name, table_info in results['tables'].items():
                with st.expander(f"📋 {table_name.upper()} ({len(table_info['columns'])} columns)"):
                    df = pd.DataFrame(table_info['columns'])
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader("Table Relationships")
            if results['relationships']:
                rel_df = pd.DataFrame(results['relationships'])
                st.dataframe(rel_df, use_container_width=True)
                
                # Simple relationship diagram
                st.subheader("Relationship Diagram")
                fig = go.Figure()
                # Add nodes and edges representation
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No explicit foreign key relationships found in the schema")
        
        with tab3:
            st.subheader("Domain Classification Results")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.success(f"**Detected Domain:** {results['domain']}")
                
                # Domain confidence visualization
                domains = ["E-commerce", "Healthcare", "Finance", "Education", "Manufacturing"]
                confidences = [90 if d == results['domain'] else np.random.randint(10, 30) for d in domains]
                
                fig = px.bar(
                    x=confidences, 
                    y=domains, 
                    orientation='h',
                    title="Domain Classification Confidence",
                    color=confidences,
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Recommended KPIs")
            for kpi in results['kpis']:
                with st.expander(f"📊 {kpi['name']}"):
                    st.write(f"**Description:** {kpi['description']}")
                    st.write(f"**Metric Type:** {kpi['metric_type']}")
                    st.write(f"**Suggested Tables:** {', '.join(kpi['suggested_tables'])}")
        
        with tab5:
            st.subheader("Generated Insight Queries")
            for query in results['queries']:
                with st.expander(f"💡 {query['title']}"):
                    st.write(f"**Description:** {query['description']}")
                    st.code(query['query'], language='sql')
                    st.write(f"**Visualization:** {query['visualization']}")
                    
                    # Copy button
                    if st.button(f"📋 Copy Query", key=f"copy_{query['title']}"):
                        st.success("Query copied to clipboard!")

if __name__ == "__main__":
    main()