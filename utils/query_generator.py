import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

class DomainClassifier:
    def __init__(self):
        # Load your pre-trained domain classification model
        self.tokenizer = AutoTokenizer.from_pretrained('models/domain_model')
        self.model = AutoModelForSequenceClassification.from_pretrained('models/domain_model')
        self.domain_labels = ['ecommerce', 'hr', 'finance', 'healthcare', 'education']
    
    def predict_domain(self, schema_data, erd_graph):
        """Predict the business domain from schema and ERD"""
        # Create a text representation of the schema for classification
        schema_text = self._create_schema_text(schema_data, erd_graph)
        
        # Mock prediction - replace with your actual model inference
        tokens = self.tokenizer(schema_text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = np.argmax(predictions.numpy())
        confidence = predictions[0][predicted_class_idx].item()
        
        return self.domain_labels[predicted_class_idx], confidence, schema_text
    
    def _create_schema_text(self, schema_data, erd_graph):
        """Create a textual representation of the schema for classification"""
        text_parts = []
        
        # Add table and column information
        for table in schema_data['table_name'].unique():
            table_cols = schema_data[schema_data['table_name'] == table]
            col_info = ', '.join([f"{row['column_name']}({row.get('semantic_label', '')})" 
                                for _, row in table_cols.iterrows()])
            text_parts.append(f"Table {table} has columns: {col_info}")
        
        # Add relationship information from ERD
        relationships = []
        for edge in erd_graph.edges(data=True):
            if edge[2].get('type') == 'relationship':
                relationships.append(f"{edge[0]} -> {edge[1]} ({edge[2].get('relationship', '')})")
        
        if relationships:
            text_parts.append("Relationships: " + "; ".join(relationships))
        
        return ". ".join(text_parts)