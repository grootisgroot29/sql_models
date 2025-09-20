import re
from datetime import datetime

class QueryGenerator:
    def __init__(self, kpi_mapper, schema_data):
        self.kpi_mapper = kpi_mapper
        self.schema_data = schema_data
        self.available_kpis = {}
    
    def generate_query(self, natural_language_query, domain):
        """Generate SQL query from natural language"""
        parsed = self._parse_query(natural_language_query)
        kpi_info = self._get_kpi_info(parsed.get('measure'))
        
        if not kpi_info:
            return "Sorry, I couldn't identify a relevant KPI for your query."
        
        template = self._select_template(parsed['intent'])
        query = self._fill_template(template, parsed, kpi_info)
        
        return query
    
    def _parse_query(self, query):
        """Simple NLU parser"""
        query_lower = query.lower()
        intent = "aggregation"
        entities = {}
        
        # Detect intent
        if any(word in query_lower for word in ['top', 'most', 'highest', 'max', 'best']):
            intent = "ranking"
            if re.search(r'\b(\d+)\b', query_lower):
                entities['limit'] = int(re.search(r'\b(\d+)\b', query_lower).group(1))
        
        # Detect measures
        if any(word in query_lower for word in ['revenue', 'sales', 'amount']):
            entities['measure'] = 'Revenue'
        elif any(word in query_lower for word in ['customer', 'user', 'client']):
            entities['measure'] = 'Customer Count'
        
        # Detect dimensions
        if any(word in query_lower for word in ['product', 'item']):
            entities['dimension'] = 'product'
        elif any(word in query_lower for word in ['customer', 'user']):
            entities['dimension'] = 'customer'
        
        # Detect time filters
        current_year = datetime.now().year
        if '2023' in query_lower or 'last year' in query_lower:
            entities['time_filter'] = {'year': 2023}
        elif 'this year' in query_lower:
            entities['time_filter'] = {'year': current_year}
        
        return {'intent': intent, 'entities': entities}
    
    def _get_kpi_info(self, measure_name):
        """Get KPI information from mapper"""
        if not measure_name:
            return None
        
        kpis = self.kpi_mapper.map_kpis('ecommerce', self.schema_data)  # Assuming ecommerce for demo
        for kpi in kpis:
            if kpi['kpi_name'].lower() == measure_name.lower():
                return kpi
        return None
    
    def _select_template(self, intent):
        templates = {
            'aggregation': """
            SELECT {dimension}, {measure_formula} as value
            FROM {main_table}
            {join_clause}
            {where_clause}
            GROUP BY {dimension}
            ORDER BY value DESC
            """,
            'ranking': """
            SELECT {dimension}, {measure_formula} as value
            FROM {main_table}
            {join_clause}
            {where_clause}
            GROUP BY {dimension}
            ORDER BY value DESC
            LIMIT {limit}
            """
        }
        return templates.get(intent, templates['aggregation'])
    
    def _fill_template(self, template, parsed, kpi_info):
        """Fill template with actual values"""
        replacements = {
            'measure_formula': kpi_info['formula'],
            'dimension': 'products.product_name' if parsed['entities'].get('dimension') == 'product' else 'customers.customer_name',
            'main_table': 'sales',
            'join_clause': 'LEFT JOIN products ON sales.product_id = products.product_id',
            'limit': parsed['entities'].get('limit', 10),
            'where_clause': 'WHERE YEAR(sale_date) = 2023' if parsed['entities'].get('time_filter') else ''
        }
        
        for key, value in replacements.items():
            template = template.replace(f'{{{key}}}', str(value))
        
        return template