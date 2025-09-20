import json
import pandas as pd
import streamlit as st

class KPIMapper:
    def __init__(self, ontology_path='data/kpi_ontology.json'):
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)
    
    def map_kpis(self, domain, schema_data):
        """Map schema columns to domain-specific KPIs"""
        domain_kpis = next((d for d in self.ontology['domains'] if d['name'] == domain), None)
        if not domain_kpis:
            return []
        
        identified_kpis = []
        for kpi in domain_kpis['kpis']:
            mapping = self._map_single_kpi(kpi, schema_data)
            if mapping['is_valid']:
                identified_kpis.append(mapping)
        
        return identified_kpis
    
    def _map_single_kpi(self, kpi_definition, schema_data):
        component_map = {}
        confidence = 1.0
        
        for component in kpi_definition['required_components']:
            candidate_columns = []
            
            for _, row in schema_data.iterrows():
                if row.get('semantic_label') == component['semantic_type']:
                    score = self._score_column_relevance(row['column_name'], component['description_keywords'])
                    candidate_columns.append((row['column_name'], row['table_name'], score))
            
            if candidate_columns:
                candidate_columns.sort(key=lambda x: x[2], reverse=True)
                best_col, best_table, best_score = candidate_columns[0]
                component_map[component['role']] = f"{best_table}.{best_col}"
                confidence *= best_score
            else:
                return {"is_valid": False}
        
        # Instantiate the formula
        formula = kpi_definition['formula_template']
        for role, col_name in component_map.items():
            formula = formula.replace(f"{{{role}}}", col_name)
        
        return {
            "is_valid": True,
            "kpi_name": kpi_definition['name'],
            "description": kpi_definition['description'],
            "formula": formula,
            "confidence": round(confidence, 2),
            "component_map": component_map
        }
    
    def _score_column_relevance(self, column_name, keywords):
        column_name_lower = column_name.lower()
        score = 0
        for keyword in keywords:
            if keyword in column_name_lower:
                score += 1
        return score / len(keywords)