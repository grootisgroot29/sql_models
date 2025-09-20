import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from utils.column_association import ColumnAssociationModel
from utils.domain_classifier import DomainClassifier
from utils.kpi_mapper import KPIMapper
from utils.query_generator import QueryGenerator

# Page configuration
st.set_page_config(
    page_title="Business Schema Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'schema_data' not in st.session_state:
    st.session_state.schema_data = None
if 'erd_data' not in st.session_state:
    st.session_state.erd_data = None

# Initialize models
@st.cache_resource
def load_models():
    return {
        'column_model': ColumnAssociationModel(),
        'domain_model': DomainClassifier(),
        'kpi_mapper': KPIMapper()
    }

models = load_models()

# Sidebar
st.sidebar.title("📊 Business Schema Insights")
st.sidebar.markdown("Upload your database schema and ERD to generate business insights and queries.")

# File upload
uploaded_schema = st.sidebar.file_uploader("Upload Schema (CSV)", type=['csv'])
uploaded_erd = st.sidebar.file_uploader("Upload ERD (JSON - optional)", type=['json'])

# Example data
if st.sidebar.button("Load Example E-commerce Schema"):
    example_schema = pd.read_csv('examples/ecommerce_schema.csv')
    st.session_state.schema_data = example_schema
    st.session_state.erd_data = json.load(open('examples/sample_erd.json'))
    st.session_state.processed = False

# Main content
st.title("Business Intelligence from Database Schema")
st.markdown("Transform your database schema into actionable business insights and queries.")

if uploaded_schema:
    st.session_state.schema_data = pd.read_csv(uploaded_schema)
if uploaded_erd:
    st.session_state.erd_data = json.load(uploaded_erd)

if st.session_state.schema_data is not None:
    # Display uploaded data
    st.subheader("Uploaded Schema Data")
    st.dataframe(st.session_state.schema_data)
    
    if st.session_state.erd_data:
        st.subheader("Uploaded ERD Data")
        st.json(st.session_state.erd_data)

    if st.button("Process Schema") or st.session_state.processed:
        st.session_state.processed = True
        
        # Step 1: Column Association and ERD
        with st.spinner("Analyzing column semantics and building ERD graph..."):
            schema_with_semantics = st.session_state.schema_data.copy()
            semantic_labels = models['column_model'].predict_semantics(schema_with_semantics)
            schema_with_semantics['semantic_label'] = semantic_labels
            
            erd_graph = models['column_model'].build_erd_graph(
                schema_with_semantics, 
                st.session_state.erd_data
            )
            graph_viz = models['column_model'].visualize_graph(erd_graph)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Column Semantics")
            st.dataframe(schema_with_semantics[['table_name', 'column_name', 'semantic_label']])
        
        with col2:
            st.subheader("ERD Visualization")
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=graph_viz['edge_x'], y=graph_viz['edge_y'],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'))
            
            # Add nodes
            node_trace = go.Scatter(
                x=graph_viz['node_x'], y=graph_viz['node_y'],
                mode='markers+text',
                hoverinfo='text',
                text=graph_viz['node_text'],
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    color=[1 if t == 'table' else 2 for t in graph_viz['node_types']],
                    colorbar=dict(
                        thickness=15,
                        title='Node Type',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )
            
            fig.add_trace(node_trace)
            fig.update_layout(showlegend=False, hovermode='closest')
            st.plotly_chart(fig, use_container_width=True)
        
        # Step 2: Domain Classification
        with st.spinner("Identifying business domain..."):
            domain, confidence, schema_text = models['domain_model'].predict_domain(
                schema_with_semantics, erd_graph
            )
        
        st.subheader("Domain Classification")
        st.success(f"**Identified Domain:** {domain.upper()} (Confidence: {confidence:.2%})")
        st.text_area("Schema Text Representation", schema_text, height=100)
        
        # Step 3: KPI Mapping
        with st.spinner("Mapping business KPIs..."):
            kpis = models['kpi_mapper'].map_kpis(domain, schema_with_semantics)
        
        st.subheader("Identified Business KPIs")
        for kpi in kpis:
            with st.expander(f"📈 {kpi['kpi_name']} (Confidence: {kpi['confidence']})"):
                st.write(f"**Description:** {kpi['description']}")
                st.write(f"**Formula:** `{kpi['formula']}`")
                st.write("**Components:**")
                for role, col in kpi['component_map'].items():
                    st.write(f"  - {role}: `{col}`")
        
        # Step 4: Query Generation
        st.subheader("Query Generation")
        query_input = st.text_input(
            "Ask a business question:",
            placeholder="e.g., 'Show top 5 products by revenue in 2023'"
        )
        
        if query_input:
            query_gen = QueryGenerator(models['kpi_mapper'], schema_with_semantics)
            generated_query = query_gen.generate_query(query_input, domain)
            
            st.code(generated_query, language='sql')
            
            if st.button("Execute Query (Simulation)"):
                st.success("Query executed successfully! (This is a simulation)")
                # In a real implementation, you would connect to the database and execute the query
                st.dataframe(pd.DataFrame({
                    'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
                    'revenue': [50000, 45000, 40000, 35000, 30000]
                }))

else:
    st.info("👈 Please upload a schema file to get started, or load the example schema.")
    st.markdown("""
    ### Expected Schema CSV Format:
    - `table_name`: Name of the table
    - `column_name`: Name of the column
    - `data_type`: Data type of the column (optional)
    
    ### Example:
    ```csv
    table_name,column_name,data_type
    customers,customer_id,int
    customers,customer_name,varchar
    sales,sale_id,int
    sales,amount,decimal
    sales,sale_date,date
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Business Intelligence Pipeline")