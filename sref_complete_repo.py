# =================================================================
#  DATA LOADING & KNOWLEDGE GRAPH
# =================================================================

# src/sref/utils/data_loader.py
"""
Data loading utilities for SREF framework
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def load_promise_dataset(data_path: str) -> Dict:
    """Load PROMISE repository dataset."""
    data_path = Path(data_path)
    
    # Load NFR dataset
    nfr_file = data_path / 'nfr_dataset.csv'
    if nfr_file.exists():
        df = pd.read_csv(nfr_file)
        return {
            'requirements': df['requirement_text'].tolist(),
            'labels': df['category_id'].tolist(),
            'categories': df['category'].tolist()
        }
    else:
        # Return sample data if file doesn't exist
        logger.warning(f"PROMISE dataset not found at {nfr_file}. Using sample data.")
        return _get_sample_promise_data()

def load_industrial_dataset(data_path: str, domain: str) -> Dict:
    """Load industrial dataset for specific domain."""
    data_path = Path(data_path) / domain
    
    # Load requirements
    req_file = data_path / 'requirements.csv'
    if req_file.exists():
        df = pd.read_csv(req_file)
        return {
            'requirements': df['text'].tolist(),
            'labels': df['label'].tolist() if 'label' in df.columns else [],
            'domain': domain,
            'metadata': df.to_dict('records')
        }
    else:
        logger.warning(f"Industrial dataset not found for {domain}. Using sample data.")
        return _get_sample_industrial_data(domain)

def load_evaluation_datasets(data_dir: str, domains: List[str]) -> Dict:
    """Load evaluation datasets for multiple domains."""
    datasets = {}
    
    for domain in domains:
        if domain == 'promise':
            datasets[domain] = load_promise_dataset(data_dir)
        else:
            datasets[domain] = load_industrial_dataset(data_dir, domain)
            
    return datasets

def _get_sample_promise_data() -> Dict:
    """Generate sample PROMISE dataset."""
    return {
        'requirements': [
            "The system shall authenticate users using multi-factor authentication.",
            "The application must respond to user queries within 2 seconds.",
            "Users should be able to easily navigate through the interface.",
            "The system shall maintain 99.9% uptime availability.",
            "The software must be portable across different operating systems.",
            "The system shall encrypt all sensitive data using AES-256.",
            "The interface should be intuitive for users with disabilities."
        ],
        'labels': [1, 2, 3, 4, 6, 1, 3],  # Security, Performance, Usability, Reliability, Portability, Security, Usability
        'categories': ['Security', 'Performance', 'Usability', 'Reliability', 'Portability', 'Security', 'Usability']
    }

def _get_sample_industrial_data(domain: str) -> Dict:
    """Generate sample industrial dataset."""
    domain_requirements = {
        'healthcare': [
            "The EHR system shall comply with HIPAA privacy regulations.",
            "Patient data must be encrypted both at rest and in transit.",
            "The system shall provide real-time clinical decision support.",
            "Medical records must be accessible within 3 seconds.",
            "The interface shall support multiple languages for diverse populations."
        ],
        'automotive': [
            "The ADAS system shall detect obstacles within 100 meters.",
            "Vehicle controls must respond within 50 milliseconds.",
            "The infotainment system shall support Android Auto and Apple CarPlay.",
            "Engine diagnostics must be available through OBD-II interface.",
            "The system shall maintain functionality in temperatures from -40°C to 85°C."
        ],
        'financial': [
            "All transactions must be PCI DSS compliant.",
            "The system shall detect fraudulent activities in real-time.",
            "User authentication must support biometric verification.",
            "Payment processing must complete within 5 seconds.",
            "The application shall maintain audit logs for 7 years."
        ]
    }
    
    reqs = domain_requirements.get(domain, ["Sample requirement for " + domain])
    return {
        'requirements': reqs,
        'labels': list(range(len(reqs))),
        'domain': domain,
        'metadata': [{'id': i, 'domain': domain} for i in range(len(reqs))]
    }

# =================================================================
# KNOWLEDGE GRAPH CONSTRUCTOR
# =================================================================

# src/sref/knowledge_graph/constructor.py
"""
Knowledge Graph Constructor for Requirements
"""

import spacy
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class KnowledgeGraphConstructor:
    """
    Constructs knowledge graphs from requirements text using NLP techniques.
    """
    
    def __init__(self, entity_model: str = "en_core_web_sm", 
                 relation_model: str = "en_core_web_sm"):
        """
        Initialize knowledge graph constructor.
        
        Args:
            entity_model: SpaCy model for entity extraction
            relation_model: SpaCy model for relation extraction
        """
        try:
            self.nlp = spacy.load(entity_model)
        except OSError:
            logger.warning(f"SpaCy model {entity_model} not found. Installing...")
            spacy.cli.download(entity_model)
            self.nlp = spacy.load(entity_model)
            
        self.graph = nx.DiGraph()
        self.entity_types = {
            'SYSTEM': 'System Component',
            'USER': 'User Role',
            'PROCESS': 'Business Process',
            'DATA': 'Data Entity',
            'INTERFACE': 'Interface Component',
            'SECURITY': 'Security Measure',
            'PERFORMANCE': 'Performance Metric'
        }
        
        logger.info("Knowledge graph constructor initialized")
        
    def build_graph(self, requirements: List[str]) -> Dict:
        """
        Build knowledge graph from requirements.
        
        Args:
            requirements: List of requirement texts
            
        Returns:
            Dictionary containing graph structure and statistics
        """
        logger.info(f"Building knowledge graph from {len(requirements)} requirements...")
        
        # Clear previous graph
        self.graph.clear()
        
        # Extract entities and relations
        all_entities = []
        all_relations = []
        
        for i, req_text in enumerate(requirements):
            req_entities = self._extract_entities(req_text, req_id=i)
            req_relations = self._extract_relations(req_text, req_entities, req_id=i)
            
            all_entities.extend(req_entities)
            all_relations.extend(req_relations)
            
        # Build graph structure
        self._build_graph_structure(all_entities, all_relations)
        
        # Generate triples
        triples = self._generate_triples()
        
        # Calculate statistics
        stats = self._calculate_graph_statistics()
        
        return {
            'entities': all_entities,
            'relations': all_relations,
            'triples': triples,
            'graph_structure': {
                'nodes': list(self.graph.nodes(data=True)),
                'edges': list(self.graph.edges(data=True))
            },
            'statistics': stats
        }
        
    def _extract_entities(self, text: str, req_id: int) -> List[Dict]:
        """Extract entities from requirement text."""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity_type = self._classify_entity_type(ent.text, ent.label_)
            entities.append({
                'text': ent.text,
                'type': entity_type,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_,
                'requirement_id': req_id
            })
            
        # Extract domain-specific entities
        domain_entities = self._extract_domain_entities(text, req_id)
        entities.extend(domain_entities)
        
        return entities
        
    def _classify_entity_type(self, text: str, spacy_label: str) -> str:
        """Classify entity into domain-specific types."""
        text_lower = text.lower()
        
        # System components
        if any(word in text_lower for word in ['system', 'application', 'software', 'platform']):
            return 'SYSTEM'
        
        # User roles
        elif any(word in text_lower for word in ['user', 'admin', 'operator', 'customer']):
            return 'USER'
            
        # Data entities
        elif any(word in text_lower for word in ['data', 'database', 'record', 'information']):
            return 'DATA'
            
        # Interface components
        elif any(word in text_lower for word in ['interface', 'api', 'gui', 'ui']):
            return 'INTERFACE'
            
        # Security measures
        elif any(word in text_lower for word in ['authentication', 'authorization', 'encryption', 'security']):
            return 'SECURITY'
            
        # Performance metrics
        elif any(word in text_lower for word in ['performance', 'speed', 'response', 'time']):
            return 'PERFORMANCE'
            
        # Process entities
        elif spacy_label in ['EVENT', 'PRODUCT'] or any(word in text_lower for word in ['process', 'procedure']):
            return 'PROCESS'
            
        # Default classification
        else:
            return 'ENTITY'
            
    def _extract_domain_entities(self, text: str, req_id: int) -> List[Dict]:
        """Extract domain-specific entities using patterns."""
        domain_entities = []
        
        # Technical specifications
        tech_patterns = [
            r'(\d+(?:\.\d+)?)\s*(seconds?|ms|minutes?|hours?)',  # Time specifications
            r'(\d+(?:\.\d+)?)\s*(%|percent)',  # Percentage values
            r'(AES-\d+|RSA-\d+|SHA-\d+)',  # Encryption standards
            r'(HTTP[S]?|FTP[S]?|SMTP|TCP|UDP)',  # Protocols
            r'(API|REST|SOAP|GraphQL)',  # API types
        ]
        
        import re
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                domain_entities.append({
                    'text': match.group(0),
                    'type': 'TECHNICAL_SPEC',
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'TECHNICAL',
                    'requirement_id': req_id
                })
                
        return domain_entities
        
    def _extract_relations(self, text: str, entities: List[Dict], req_id: int) -> List[Dict]:
        """Extract relations between entities."""
        doc = self.nlp(text)
        relations = []
        
        # Extract dependency-based relations
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                head_text = token.head.text
                dep_text = token.text
                
                # Find matching entities
                head_entity = self._find_entity_by_text(head_text, entities)
                dep_entity = self._find_entity_by_text(dep_text, entities)
                
                if head_entity and dep_entity:
                    relation_type = self._classify_relation(token.dep_, head_text, dep_text)
                    relations.append({
                        'head': head_entity['text'],
                        'relation': relation_type,
                        'tail': dep_entity['text'],
                        'confidence': 0.8,
                        'requirement_id': req_id
                    })
                    
        # Extract pattern-based relations
        pattern_relations = self._extract_pattern_relations(text, entities, req_id)
        relations.extend(pattern_relations)
        
        return relations
        
    def _find_entity_by_text(self, text: str, entities: List[Dict]) -> Optional[Dict]:
        """Find entity by text match."""
        for entity in entities:
            if entity['text'].lower() == text.lower():
                return entity
        return None
        
    def _classify_relation(self, dependency: str, head: str, tail: str) -> str:
        """Classify relation based on dependency and context."""
        head_lower = head.lower()
        tail_lower = tail.lower()
        
        # System relations
        if 'system' in head_lower or 'system' in tail_lower:
            if dependency == 'nsubj':
                return 'PERFORMS'
            elif dependency == 'dobj':
                return 'PROCESSES'
                
        # User relations
        elif 'user' in head_lower or 'user' in tail_lower:
            if dependency == 'nsubj':
                return 'USES'
            elif dependency == 'dobj':
                return 'INTERACTS_WITH'
                
        # Data relations
        elif 'data' in head_lower or 'data' in tail_lower:
            return 'CONTAINS'
            
        # Default relations
        else:
            relation_map = {
                'nsubj': 'SUBJECT_OF',
                'dobj': 'OBJECT_OF',
                'pobj': 'RELATED_TO'
            }
            return relation_map.get(dependency, 'RELATED_TO')
            
    def _extract_pattern_relations(self, text: str, entities: List[Dict], req_id: int) -> List[Dict]:
        """Extract relations using predefined patterns."""
        relations = []
        
        # Modal verb patterns
        modal_patterns = [
            (r'(\w+)\s+shall\s+(\w+)', 'MUST'),
            (r'(\w+)\s+should\s+(\w+)', 'SHOULD'),
            (r'(\w+)\s+must\s+(\w+)', 'REQUIRED'),
            (r'(\w+)\s+can\s+(\w+)', 'CAN'),
            (r'(\w+)\s+will\s+(\w+)', 'WILL')
        ]
        
        import re
        for pattern, relation_type in modal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1)
                object_word = match.group(2)
                
                # Find matching entities
                subj_entity = self._find_entity_by_text(subject, entities)
                obj_entity = self._find_entity_by_text(object_word, entities)
                
                if subj_entity and obj_entity:
                    relations.append({
                        'head': subj_entity['text'],
                        'relation': relation_type,
                        'tail': obj_entity['text'],
                        'confidence': 0.9,
                        'requirement_id': req_id
                    })
                    
        return relations
        
    def _build_graph_structure(self, entities: List[Dict], relations: List[Dict]):
        """Build NetworkX graph structure."""
        # Add entity nodes
        for entity in entities:
            self.graph.add_node(
                entity['text'],
                type=entity['type'],
                requirement_id=entity['requirement_id'],
                entity_data=entity
            )
            
        # Add relation edges
        for relation in relations:
            self.graph.add_edge(
                relation['head'],
                relation['tail'],
                relation=relation['relation'],
                confidence=relation['confidence'],
                requirement_id=relation['requirement_id']
            )
            
    def _generate_triples(self) -> List[Dict]:
        """Generate RDF-style triples from graph."""
        triples = []
        
        for edge in self.graph.edges(data=True):
            head, tail, data = edge
            triples.append({
                'subject': head,
                'predicate': data.get('relation', 'RELATED_TO'),
                'object': tail,
                'confidence': data.get('confidence', 0.8),
                'requirement_id': data.get('requirement_id', -1)
            })
            
        return triples
        
    def _calculate_graph_statistics(self) -> Dict:
        """Calculate graph statistics."""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'num_connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'entity_type_distribution': self._get_entity_type_distribution(),
            'relation_type_distribution': self._get_relation_type_distribution()
        }
        
        return stats
        
    def _get_entity_type_distribution(self) -> Dict:
        """Get distribution of entity types."""
        distribution = defaultdict(int)
        
        for _, node_data in self.graph.nodes(data=True):
            entity_type = node_data.get('type', 'UNKNOWN')
            distribution[entity_type] += 1
            
        return dict(distribution)
        
    def _get_relation_type_distribution(self) -> Dict:
        """Get distribution of relation types."""
        distribution = defaultdict(int)
        
        for _, _, edge_data in self.graph.edges(data=True):
            relation_type = edge_data.get('relation', 'UNKNOWN')
            distribution[relation_type] += 1
            
        return dict(distribution)

# =================================================================
# PREPROCESSING UTILITIES
# =================================================================

# src/sref/utils/preprocessing.py
"""
Text preprocessing utilities for requirements
"""

import re
import string
from typing import List, Dict, Optional
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class RequirementsPreprocessor:
    """Preprocessor for requirements text."""
    
    def __init__(self):
        """Initialize preprocessor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Domain-specific stop words
        self.domain_stop_words = {
            'system', 'application', 'software', 'program', 'shall', 'must', 'should', 'will', 'can', 'may'
        }
        
    def preprocess_requirements(self, requirements: List[str], 
                              options: Optional[Dict] = None) -> List[str]:
        """
        Preprocess a list of requirements.
        
        Args:
            requirements: List of requirement texts
            options: Preprocessing options
            
        Returns:
            List of preprocessed requirement texts
        """
        if options is None:
            options = {
                'normalize_text': True,
                'remove_stopwords': False,
                'lemmatize': False,
                'remove_punctuation': False,
                'standardize_modal_verbs': True
            }
            
        processed = []
        
        for req in requirements:
            processed_req = self._preprocess_single(req, options)
            processed.append(processed_req)
            
        return processed
        
    def _preprocess_single(self, text: str, options: Dict) -> str:
        """Preprocess a single requirement."""
        # Basic normalization
        if options.get('normalize_text', True):
            text = self._normalize_text(text)
            
        # Standardize modal verbs
        if options.get('standardize_modal_verbs', True):
            text = self._standardize_modal_verbs(text)
            
        # Remove punctuation
        if options.get('remove_punctuation', False):
            text = self._remove_punctuation(text)
            
        # Tokenization and linguistic processing
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip stop words
            if options.get('remove_stopwords', False) and token.is_stop:
                continue
                
            # Skip domain stop words
            if token.text.lower() in self.domain_stop_words:
                continue
                
            # Lemmatization
            if options.get('lemmatize', False):
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
                
        return ' '.join(tokens)
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text format."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Normalize hyphens
        text = re.sub(r'[-–—]', '-', text)
        
        return text.strip()
        
    def _standardize_modal_verbs(self, text: str) -> str:
        """Standardize modal verbs in requirements."""
        # Replace modal verb variations
        modal_replacements = {
            r'\bshall\b': 'must',
            r'\bwill\b': 'must',
            r'\bshould\b': 'should',
            r'\bmay\b': 'may',
            r'\bcan\b': 'can'
        }
        
        for pattern, replacement in modal_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text
        
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        # Keep some punctuation that might be important
        keep_chars = '.-_%'
        punct_to_remove = ''.join(c for c in string.punctuation if c not in keep_chars)
        
        translator = str.maketrans('', '', punct_to_remove)
        return text.translate(translator)

def preprocess_requirements(requirements: List[str], **kwargs) -> List[str]:
    """Convenience function for preprocessing requirements."""
    preprocessor = RequirementsPreprocessor()
    return preprocessor.preprocess_requirements(requirements, kwargs)

# =================================================================
# API SERVER IMPLEMENTATION
# =================================================================

# src/sref/api/server.py
"""
FastAPI server for SREF framework
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
from pathlib import Path

from ..core.framework import SREFFramework
from ..utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SREF Framework API",
    description="Semantic Requirements Engineering Framework API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SREF framework
sref_framework = None

# Request/Response models
class RequirementsRequest(BaseModel):
    requirements: List[str]
    options: Optional[Dict] = None

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict] = None
    processing_time: float
    statistics: Dict

class ClassificationRequest(BaseModel):
    requirements: List[str]

class ValidationRequest(BaseModel):
    requirements: List[str]
    classifications: Optional[List[Dict]] = None

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize SREF framework on startup."""
    global sref_framework
    
    try:
        logger.info("Initializing SREF framework...")
        config = Config()
        sref_framework = SREFFramework()
        sref_framework.initialize_components()
        logger.info("SREF framework initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SREF framework: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SREF Framework API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "framework_initialized": sref_framework is not None,
        "version": "1.0.0"
    }

@app.post("/process", response_model=ProcessingResponse)
async def process_requirements(request: RequirementsRequest):
    """Process requirements through complete SREF pipeline."""
    if not sref_framework:
        raise HTTPException(status_code=500, detail="SREF framework not initialized")
    
    try:
        logger.info(f"Processing {len(request.requirements)} requirements")
        
        # Process requirements
        results = sref_framework.process_requirements(request.requirements)
        
        return ProcessingResponse(
            success=True,
            message="Requirements processed successfully",
            results=results,
            processing_time=results.get('processing_time', 0.0),
            statistics=results.get('statistics', {})
        )
        
    except Exception as e:
        logger.error(f"Error processing requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify_requirements(request: ClassificationRequest):
    """Classify requirements using BERT classifier."""
    if not sref_framework or not sref_framework.bert_classifier:
        raise HTTPException(status_code=500, detail="BERT classifier not initialized")
    
    try:
        logger.info(f"Classifying {len(request.requirements)} requirements")
        
        # Classify requirements
        classifications = sref_framework.bert_classifier.classify_batch(request.requirements)
        
        return {
            "success": True,
            "message": "Requirements classified successfully",
            "classifications": classifications
        }
        
    except Exception as e:
        logger.error(f"Error classifying requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate_requirements(request: ValidationRequest):
    """Validate requirements using ontology validator."""
    if not sref_framework or not sref_framework.ontology_validator:
        raise HTTPException(status_code=500, detail="Ontology validator not initialized")
    
    try:
        logger.info(f"Validating {len(request.requirements)} requirements")
        
        # Validate requirements
        validation_results = sref_framework.ontology_validator.validate_requirements(
            request.requirements, request.classifications
        )
        
        return {
            "success": True,
            "message": "Requirements validated successfully",
            "validation_results": validation_results
        }
        
    except Exception as e:
        logger.error(f"Error validating requirements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge_graph")
async def build_knowledge_graph(request: RequirementsRequest):
    """Build knowledge graph from requirements."""
    if not sref_framework or not sref_framework.kg_constructor:
        raise HTTPException(status_code=500, detail="Knowledge graph constructor not initialized")
    
    try:
        logger.info(f"Building knowledge graph from {len(request.requirements)} requirements")
        
        # Build knowledge graph
        kg_results = sref_framework.kg_constructor.build_graph(request.requirements)
        
        return {
            "success": True,
            "message": "Knowledge graph built successfully",
            "knowledge_graph": kg_results
        }
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get framework statistics."""
    if not sref_framework:
        raise HTTPException(status_code=500, detail="SREF framework not initialized")
    
    try:
        stats = sref_framework.get_framework_statistics()
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main function to run the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

# =================================================================
# COMMAND LINE INTERFACE
# =================================================================

# src/sref/cli/train.py
"""
Command line interface for training SREF models
"""

import click
import logging
from pathlib import Path

from ..models.bert_classifier import BERTRequirementsClassifier
from ..utils.data_loader import load_promise_dataset, load_industrial_dataset
from ..utils.preprocessing import preprocess_requirements

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--data-path', required=True, help='Path to training data')
@click.option('--output-dir', default='./trained_models', help='Output directory')
@click.option('--model-name', default='bert-base-uncased', help='Base model name')
@click.option('--domain', default='promise', help='Domain for training')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Training batch size')
@click.option('--learning-rate', default=2e-5, help='Learning rate')
def train(data_path, output_dir, model_name, domain, epochs, batch_size, learning_rate):
    """Train BERT classifier for requirements classification."""
    
    logger.info(f"Starting training for {domain} domain...")
    
    # Load data
    if domain == 'promise':
        data = load_promise_dataset(data_path)
    else:
        data = load_industrial_dataset(data_path, domain)
    
    # Preprocess requirements
    texts = preprocess_requirements(data['requirements'])
    labels = data['labels']
    
    logger.info(f"Loaded {len(texts)} requirements")
    
    # Initialize classifier
    classifier = BERTRequirementsClassifier(
        model_name=model_name,
        num_labels=len(set(labels)),
        max_length=512
    )
    
    # Train model
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Save model
    output_path = Path(output_dir) / f"bert_classifier_{domain}"
    output_path.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(output_path))
    
    logger.info(f"Training completed. Model saved to {output_path}")

if __name__ == "__main__":
    train()

# src/sref/cli/evaluate.py
"""
Command line interface for evaluating SREF framework
"""

import click
import json
import logging
from pathlib import Path

from ..core.framework import SREFFramework
from ..utils.data_loader import load_evaluation_datasets
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--config', default='config/evaluation_config.yaml', help='Configuration file')
@click.option('--data-dir', required=True, help='Directory containing evaluation datasets')
@click.option('--model-dir', required=True, help='Directory containing trained models')
@click.option('--output-dir', default='./evaluation_results', help='Output directory')
@click.option('--domains', multiple=True, default=['promise'], help='Domains to evaluate')
@click.option('--visualize', is_flag=True, help='Generate visualizations')
def evaluate(config, data_dir, model_dir, output_dir, domains, visualize):
    """Evaluate SREF framework on specified domains."""
    
    logger.info(f"Starting evaluation for domains: {domains}")
    
    # Initialize framework
    framework = SREFFramework(config_path=config)
    framework.initialize_components()
    
    # Load evaluation datasets
    datasets = load_evaluation_datasets(data_dir, list(domains))
    
    # Evaluate each domain
    all_results = {}
    
    for domain in domains:
        if domain in datasets:
            logger.info(f"Evaluating {domain} domain...")
            
            # Load domain-specific model if available
            model_path = Path(model_dir) / f"bert_classifier_{domain}"
            if model_path.exists():
                framework.bert_classifier.load_model(str(model_path))
            
            # Process requirements
            requirements = datasets[domain]['requirements']
            results = framework.process_requirements(requirements)
            all_results[domain] = results
            
            logger.info(f"Completed evaluation for {domain}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation completed. Results saved to {output_path}")

if __name__ == "__main__":
    evaluate()

# =================================================================
# CONFIGURATION FILES
# =================================================================

# config/default_config.yaml
"""
# SREF Framework Configuration

# BERT Classifier Configuration
bert_model_name: "bert-base-uncased"
num_requirement_categories: 7
max_sequence_length: 512
device: "auto"  # auto, cpu, cuda

# Ontology Configuration
ontology_path: "ontology/requirements_ontology.owl"
swrl_rules_path: "ontology/swrl_rules.swrl"

# Knowledge Graph Configuration
entity_model: "en_core_web_sm"
relation_model: "en_core_web_sm"
kg_confidence_threshold: 0.7

# Processing Configuration
batch_size: 32
log_level: "INFO"
enable_caching: true
cache_dir: "./cache"

# Evaluation Configuration
evaluation_metrics:
  - accuracy
  - precision
  - recall
  - f1_score
  - confusion_matrix

# API Configuration
api_host: "0.0.0.0"
api_port: 8000
api_workers: 4

# Domain-specific Configuration
domains:
  promise:
    data_path: "data/promise"
    categories: ["Functional", "Security", "Performance", "Usability", "Reliability", "Maintainability", "Portability"]
  
  healthcare:
    data_path: "data/healthcare"
    categories: ["Clinical", "Administrative", "Security", "Interoperability", "Performance"]
  
  automotive:
    data_path: "data/automotive"
    categories: ["Safety", "Performance", "Functional", "Diagnostic", "Communication"]
  
  financial:
    data_path: "data/financial"
    categories: ["Transaction", "Security", "Compliance", "Performance", "Audit"]
"""

# config/training_config.yaml
"""
# Training Configuration for SREF Models

# BERT Training Parameters
bert_training:
  model_name: "bert-base-uncased"
  num_epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_length: 512
  gradient_accumulation_steps: 1
  fp16: false
  
# Early Stopping
early_stopping:
  patience: 3
  min_delta: 0.001
  monitor: "eval_f1"
  mode: "max"

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  stratify: true
  random_seed: 42

# Preprocessing
preprocessing:
  normalize_text: true
  standardize_modal_verbs: true
  remove_stopwords: false
  lemmatize: false
  remove_punctuation: false

# Model Saving
model_saving:
  save_best_only: true
  save_strategy: "steps"
  save_steps: 1000
  output_dir: "./trained_models"
  
# Logging and Monitoring
logging:
  log_level: "INFO"
  log_steps: 100
  eval_steps: 500
  wandb_project: "sref-framework"
  wandb_enabled: false
"""

# =================================================================
# TESTING FRAMEWORK
# =================================================================

# tests/test_framework.py
"""
Unit tests for SREF framework
"""

import pytest
from unittest.mock import Mock, patch
import torch

from sref.core.framework import SREFFramework
from sref.models.bert_classifier import BERTRequirementsClassifier
from sref.ontology.validator import OntologyValidator
from sref.knowledge_graph.constructor import KnowledgeGraphConstructor

class TestSREFFramework:
    """Test suite for SREF framework."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_requirements = [
            "The system shall authenticate users using multi-factor authentication.",
            "The application must respond to user queries within 2 seconds.",
            "Users should be able to easily navigate through the interface."
        ]
        
    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = SREFFramework()
        assert framework is not None
        assert framework.config is not None
        
    @patch('sref.core.framework.BERTRequirementsClassifier')
    @patch('sref.core.framework.OntologyValidator')
    @patch('sref.core.framework.KnowledgeGraphConstructor')
    def test_component_initialization(self, mock_kg, mock_validator, mock_bert):
        """Test component initialization."""
        framework = SREFFramework()
        framework.initialize_components()
        
        assert framework.bert_classifier is not None
        assert framework.ontology_validator is not None
        assert framework.kg_constructor is not None
        
    def test_process_requirements_structure(self):
        """Test requirements processing structure."""
        framework = SREFFramework()
        
        # Mock components
        framework.bert_classifier = Mock()
        framework.ontology_validator = Mock()
        framework.kg_constructor = Mock()
        
        # Configure mocks
        framework.bert_classifier.classify_batch.return_value = [
            {'predicted_category': 'Security', 'confidence': 0.9}
        ]
        framework.ontology_validator.validate_requirements.return_value = [
            {'inconsistencies': [], 'validation_score': 0.8}
        ]
        framework.kg_constructor.build_graph.return_value = {
            'entities': [], 'relations': [], 'triples': []
        }
        
        # Test processing
        results = framework.process_requirements(self.sample_requirements[:1])
        
        assert 'classifications' in results
        assert 'validation_results' in results
        assert 'knowledge_graph' in results
        assert 'processing_time' in results
        assert 'statistics' in results

class TestBERTClassifier:
    """Test suite for BERT classifier."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_texts = [
            "The system shall authenticate users",
            "The application must respond within 2 seconds",
            "Users should navigate easily"
        ]
        self.sample_labels = [1, 2, 3]  # Security, Performance, Usability
        
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = BERTRequirementsClassifier(
            model_name="bert-base-uncased",
            num_labels=7
        )
        
        assert classifier.model is not None
        assert classifier.tokenizer is not None
        assert classifier.num_labels == 7
        
    def test_classification_output_structure(self):
        """Test classification output structure."""
        classifier = BERTRequirementsClassifier(num_labels=7)
        
        # Mock the model output
        with patch.object(classifier.model, 'eval'), \
             patch.object(classifier.model, '__call__') as mock_call:
            
            # Mock model output
            mock_logits = torch.tensor([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0]])
            mock_output = Mock()
            mock_output.logits = mock_logits
            mock_call.return_value = mock_output
            
            results = classifier.classify_batch(self.sample_texts[:1])
            
            assert len(results) == 1
            assert 'predicted_label' in results[0]
            assert 'predicted_category' in results[0]
            assert 'confidence' in results[0]
            assert 'all_probabilities' in results[0]

class TestOntologyValidator:
    """Test suite for ontology validator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_requirements = [
            "The system shall authenticate users",
            "The application must respond within 2 seconds"
        ]
        
    def test_validator_initialization(self):
        """Test validator initialization."""
        with patch('pathlib.Path.exists', return_value=False):
            validator = OntologyValidator(
                ontology_path="test_ontology.owl",
                rules_path="test_rules.swrl"
            )
            
            assert validator.graph is not None
            assert validator.SREF is not None
            
    def test_validation_output_structure(self):
        """Test validation output structure."""
        with patch('pathlib.Path.exists', return_value=False):
            validator = OntologyValidator(
                ontology_path="test_ontology.owl", 
                rules_path="test_rules.swrl"
            )
            
            results = validator.validate_requirements(self.sample_requirements)
            
            assert len(results) == len(self.sample_requirements)
            for result in results:
                assert 'requirement_id' in result
                assert 'inconsistencies' in result
                assert 'quality_issues' in result
                assert 'recommendations' in result
                assert 'validation_score' in result

class TestKnowledgeGraphConstructor:
    """Test suite for knowledge graph constructor."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_requirements = [
            "The system shall authenticate users",
            "The database must store user information"
        ]
        
    @patch('spacy.load')
    def test_kg_constructor_initialization(self, mock_spacy):
        """Test KG constructor initialization."""
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        
        constructor = KnowledgeGraphConstructor()
        
        assert constructor.nlp is not None
        assert constructor.graph is not None
        
    @patch('spacy.load')
    def test_kg_build_output_structure(self, mock_spacy):
        """Test KG build output structure."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        constructor = KnowledgeGraphConstructor()
        
        results = constructor.build_graph(self.sample_requirements)
        
        assert 'entities' in results
        assert 'relations' in results
        assert 'triples' in results
        assert 'graph_structure' in results
        assert 'statistics' in results

# tests/test_api.py
"""
API tests for SREF framework
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from sref.api.server import app

class TestAPI:
    """Test suite for API endpoints."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "SREF Framework API" in response.json()["message"]
        
    @patch('sref.api.server.sref_framework')
    def test_classification_endpoint(self, mock_framework):
        """Test classification endpoint."""
        # Mock framework
        mock_classifier = Mock()
        mock_classifier.classify_batch.return_value = [
            {'predicted_category': 'Security', 'confidence': 0.9}
        ]
        mock_framework.bert_classifier = mock_classifier
        
        response = self.client.post("/classify", json={
            "requirements": ["The system shall authenticate users"]
        })
        
        assert response.status_code == 200
        assert response.json()["success"] == True
        assert "classifications" in response.json()

# tests/conftest.py
"""
Test configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_requirements():
    """Sample requirements for testing."""
    return [
        "The system shall authenticate users using multi-factor authentication.",
        "The application must respond to user queries within 2 seconds.",
        "Users should be able to easily navigate through the interface.",
        "The system shall maintain 99.9% uptime availability.",
        "The software must be portable across different operating systems."
    ]

@pytest.fixture
def sample_classifications():
    """Sample classifications for testing."""
    return [
        {'predicted_category': 'Security', 'confidence': 0.9, 'predicted_label': 1},
        {'predicted_category': 'Performance', 'confidence': 0.8, 'predicted_label': 2},
        {'predicted_category': 'Usability', 'confidence': 0.7, 'predicted_label': 3},
        {'predicted_category': 'Reliability', 'confidence': 0.85, 'predicted_label': 4},
        {'predicted_category': 'Portability', 'confidence': 0.75, 'predicted_label': 6}
    ]

# =================================================================
# DOCUMENTATION
# =================================================================

# docs/installation.md
"""
# Installation Guide

## System Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional, for faster training)
- Java 11 or higher (for OWL reasoning)

## Installation Methods

### 1. Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/cornelius-okechukwu/sref-framework.git
cd sref-framework

# Build and run with Docker Compose
docker-compose up --build

# Access the API at http://localhost:8000
# Access Jupyter notebooks at http://localhost:8888
```

### 2. Local Installation

```bash
# Clone the repository
git clone https://github.com/cornelius-okechukwu/sref-framework.git
cd sref-framework

# Create virtual environment
python -m venv sref-env
source sref-env/bin/activate  # On Windows: sref-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SREF package
pip install -e .

# Download SpaCy models
python -m spacy download en_core_web_sm

# Run tests to verify installation
pytest tests/
```

### 3. Development Installation

```bash
# Clone the repository
git clone https://github.com/cornelius-okechukwu/sref-framework.git
cd sref-framework

# Install in development mode with all extras
pip install -e ".[dev,notebooks,gpu]"

# Install pre-commit hooks
pre-commit install

# Run development server
python src/sref/api/server.py
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Model Configuration
BERT_MODEL_NAME=bert-base-uncased
MAX_SEQUENCE_LENGTH=512
DEVICE=auto

# Paths
DATA_PATH=./data
MODEL_PATH=./trained_models
ONTOLOGY_PATH=./ontology

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=INFO
```

### Configuration Files

Modify `config/default_config.yaml` to customize:

- Model parameters
- Ontology paths
- Processing options
- API settings

## Verification

Test your installation:

```bash
# Run unit tests
pytest tests/

# Test API endpoints
curl http://localhost:8000/health

# Test CLI commands
sref-train --help
sref-evaluate --help
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**: Install PyTorch with CUDA support
2. **SpaCy model missing**: Run `python -m spacy download en_core_web_sm`
3. **Java not found**: Install Java 11+ for OWL reasoning
4. **Memory issues**: Reduce batch size in configuration

### Getting Help

- Check the [FAQ](docs/faq.md)
- Open an issue on GitHub
- Contact: okechukwu@utb.cz
"""

# docs/quickstart.md
"""
# Quick Start Guide

## 1. Basic Usage

### Using the Python API

```python
from sref import SREFFramework

# Initialize framework
framework = SREFFramework()
framework.initialize_components()

# Process requirements
requirements = [
    "The system shall authenticate users using multi-factor authentication.",
    "The application must respond within 2 seconds.",
    "Users should navigate easily through the interface."
]

results = framework.process_requirements(requirements)

# Access results
for i, classification in enumerate(results['classifications']):
    print(f"Requirement {i+1}: {classification['predicted_category']} "
          f"({classification['confidence']:.1%})")
```

### Using the REST API

```bash
# Start the API server
docker-compose up sref-framework

# Classify requirements
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": [
      "The system shall authenticate users using multi-factor authentication.",
      "The application must respond within 2 seconds."
    ]
  }'
```

### Using the Command Line

```bash
# Train a model
sref-train --data-path ./data/promise --domain promise --epochs 5

# Evaluate framework
sref-evaluate --data-dir ./data --model-dir ./trained_models --domains promise healthcare
```

## 2. Training Custom Models

### Prepare Your Data

Create a CSV file with requirements and labels:

```csv
requirement_text,category,label
"The system shall authenticate users",Security,1
"Response time must be under 2 seconds",Performance,2
"Interface should be user-friendly",Usability,3
```

### Train BERT Classifier

```python
from sref.models.bert_classifier import BERTRequirementsClassifier
from sref.utils.data_loader import load_promise_dataset

# Load data
data = load_promise_dataset("./data/promise")

# Initialize classifier
classifier = BERTRequirementsClassifier(
    model_name="bert-base-uncased",
    num_labels=7
)

# Train model
classifier.train(
    train_texts=data['requirements'],
    train_labels=data['labels'],
    epochs=10,
    batch_size=32
)

# Save model
classifier.save_model("./my_trained_model")
```

## 3. Ontology Validation

### Custom Ontology

Create your own OWL ontology:

```xml
<?xml version="1.0"?>
<rdf:RDF xmlns="http://mycompany.com/ontology#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    
    <owl:Class rdf:about="#MyRequirement">
        <rdfs:subClassOf rdf:resource="#Requirement"/>
    </owl:Class>
    
</rdf:RDF>
```

### Custom SWRL Rules

Add validation rules:

```swrl
MyRequirement(?r) ∧ hasPriority(?r, ?p) ∧ 
swrlb:greaterThan(?p, 5) ∧ 
swrlb:booleanNot(hasTestCase(?r, ?tc))
→ hasQualityIssue(?r, "HighPriorityWithoutTest")
```

## 4. Knowledge Graph Construction

### Extract Entities and Relations

```python
from sref.knowledge_graph.constructor import KnowledgeGraphConstructor

# Initialize constructor
kg_constructor = KnowledgeGraphConstructor()

# Build knowledge graph
requirements = ["The system shall authenticate users via API"]
kg_results = kg_constructor.build_graph(requirements)

# Access entities
for entity in kg_results['entities']:
    print(f"Entity: {entity['text']} (Type: {entity['type']})")

# Access relations  
for relation in kg_results['relations']:
    print(f"Relation: {relation['head']} --{relation['relation']}--> {relation['tail']}")
```

## 5. Evaluation and Metrics

### Evaluate Framework Performance

```python
from sref.evaluation.metrics_calculator import MetricsCalculator

# Initialize metrics calculator
metrics_calc = MetricsCalculator()

# Calculate classification metrics
classification_metrics = metrics_calc.calculate_classification_metrics(
    y_true=true_labels,
    y_pred=predicted_labels,
    class_names=['Functional', 'Security', 'Performance', 'Usability']
)

print(f"Accuracy: {classification_metrics['accuracy']:.1%}")
print(f"F1-Score: {classification_metrics['f1_weighted']:.3f}")
```

### Generate Visualizations

```python
from sref.evaluation.visualization import SREFVisualizer

# Create visualizer
visualizer = SREFVisualizer()

# Plot results
visualizer.plot_classification_performance(
    metrics=classification_metrics,
    save_path="./results/classification_plot.png"
)
```

## 6. Integration Examples

### Web Application Integration

```python
from fastapi import FastAPI
from sref import SREFFramework

app = FastAPI()
sref = SREFFramework()
sref.initialize_components()

@app.post("/analyze_requirements")
async def analyze_requirements(requirements: List[str]):
    results = sref.process_requirements(requirements)
    return {
        "classifications": results['classifications'],
        "validation_issues": [
            r for r in results['validation_results'] 
            if r['inconsistencies']
        ],
        "processing_time": results['processing_time']
    }
```

### Jupyter Notebook Integration

```python
# Install in notebook environment
!pip install sref-framework

# Import and use
from sref import SREFFramework
import pandas as pd

# Load and analyze requirements
df = pd.read_csv('requirements.csv')
framework = SREFFramework()
framework.initialize_components()

results = framework.process_requirements(df['requirement_text'].tolist())

# Display results
pd.DataFrame(results['classifications'])
```

## Next Steps

- Read the [API Documentation](docs/api.md)
- Explore [Examples](examples/)
- Check [Configuration Options](docs/configuration.md)
- Learn about [Custom Domains](docs/domains.md)
"""

# =================================================================
# EXAMPLE USAGE FILES
# =================================================================

i+1}: {cls['predicted_category']} "
              f"({cls['confidence']:.1%})")
        print(f"   Text: {req[:60]}...")
        print()
    
    # Display validation results
    print("4. Validation Results:")
    print("-" * 30)
    for i, validation in enumerate(results['validation_results']):
        print(f"Requirement {i+1}:")
        print(f"   Validation Score: {validation['validation_score']:.2f}")
        
        if validation['inconsistencies']:
            print(f"   ⚠ Inconsistencies: {', '.join(validation['inconsistencies'])}")
        
        if validation['quality_issues']:
            print(f"   ⚠ Quality Issues: {', '.join(validation['quality_issues'])}")
            
        if validation['recommendations']:
            print(f"   💡 Recommendations: {', '.join(validation['recommendations'])}")
        print()
    
    # Display knowledge graph results
    print("5. Knowledge Graph Results:")
    print("-" * 30)
    kg = results['knowledge_graph']
    print(f"   Entities extracted: {len(kg.get('entities', []))}")
    print(f"   Relations identified: {len(kg.get('relations', []))}")
    print(f"   Triples generated: {len(kg.get('triples', []))}")
    
    # Show sample entities
    if kg.get('entities'):
        print("\n   Sample entities:")
        for entity in kg['entities'][:5]:
            print(f"     - {entity.get('text', 'N/A')} ({entity.get('type', 'unknown')})")
    
    # Display overall statistics
    print("\n6. Processing Statistics:")
    print("-" * 30)
    stats = results['statistics']
    print(f"   Average confidence: {stats.get('avg_classification_confidence', 0):.1%}")
    print(f"   Validation issues: {stats.get('validation_issues_found', 0)}")
    print(f"   Processing rate: {stats.get('processing_rate', 0):.1f} req/sec")
    
    # Save results
    print("\n7. Saving Results...")
    framework.save_results(results, "example_results.json")
    print("   ✓ Results saved to example_results.json")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
