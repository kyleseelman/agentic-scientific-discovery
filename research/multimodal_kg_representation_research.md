# Multimodal Biological Data in Knowledge Graphs & Agent Memory Systems

## Research Survey (2022–2026)

---

## 1. Landscape: What Leading Open-Source Biomedical KGs Actually Use

### 1.1 PrimeKG (Harvard / Zitnik Lab, 2023)

| Attribute | Value |
|---|---|
| **Data model** | Heterogeneous property graph (CSV/NetworkX) |
| **Scale** | 129,375 nodes, 4,050,249 edges |
| **Node types** | 10: Drug, Disease, Gene/Protein, Pathway, Biological Process, Molecular Function, Cellular Component, Anatomy, Phenotype, Exposure |
| **Edge types** | ~30 undirected relation types |
| **Sources** | 20 resources (DrugBank, DisGeNET, MONDO, Reactome, UBERON, GO, etc.) |
| **Multimodal** | Text descriptions for drug/disease nodes (clinical guidelines from Mayo Clinic, Orphanet, DrugBank) |
| **ID scheme** | Node identifiers from source ontologies (MONDO, UBERON, GO, DrugBank, etc.) |
| **Distribution** | CSV on Harvard Dataverse; NetworkX construction via Jupyter |

**Key design decisions:**
- Flat CSV with `(source, relation, target, source_type, target_type)` — no embedded numeric features in the graph itself.
- Text features are stored as separate columns/files, not as node properties in a database.
- PrimeKG++ (BioMedKG, 2025) extends this with amino acid sequences (from NCBI/Gene), SMILES strings (PubChem/DrugBank), and pathway text (Reactome), embedding them via domain-specific language models (ESM-2 for proteins, MolBERT for drugs, BioBERT for text).

### 1.2 SPOKE (UCSF / Baranzini Lab, 2023)

| Attribute | Value |
|---|---|
| **Data model** | **Neo4j property graph** |
| **Scale** | 27M+ nodes, 53M+ edges |
| **Node types** | 21 (Gene, Protein, Compound, Disease, Symptom, Side Effect, Anatomy, Pathway, Biological Process, Molecular Function, Cellular Component, Pharmacologic Class, Food, Nutrient, etc.) |
| **Edge types** | 55 |
| **Sources** | 41 databases |
| **Ontology framework** | 11 ontologies structure the graph (DO, GO, UBERON, MESH, ChEBI, etc.) |
| **Updates** | Weekly automated rebuild via Python scripts |
| **Access** | REST API, Neo4j Browser, KG-RAG integration for LLMs |

**Key design decisions:**
- Pure property graph in Neo4j — no RDF, no formal OWL reasoning.
- Ontologies used as structural scaffolding for node type hierarchies and ID mapping, not for inference.
- Experimentally measured data only (no literature-mined text triples), giving high factual reliability.
- KG-RAG (2024) wraps SPOKE for LLM integration: extracts "prompt-aware context" from graph traversals and feeds it to GPT-4/Llama as structured evidence.

### 1.3 Hetionet (Himmelstein Lab, 2017 — static)

| Attribute | Value |
|---|---|
| **Data model** | Heterogeneous network ("hetnet"), property graph (Neo4j + JSON) |
| **Scale** | 47,031 nodes, 2,250,197 edges |
| **Node types** | 11 (Compound, Disease, Gene, Anatomy, Pathway, Biological Process, Molecular Function, Cellular Component, Pharmacologic Class, Side Effect, Symptom) |
| **Edge types** | 24 |
| **Sources** | 29 databases |

**Key design decisions:**
- Metagraph (schema of types) explicitly defined; JSON format includes `metanode_kinds`, `metaedge_tuples`, and abbreviation mappings.
- Edge properties include provenance (`source`), numeric scores (`z_score`), method metadata (`method: "measured"`), and quality flags (`unbiased: true`).
- Ontology slims for ID harmonization (DO slim for diseases, MESH for symptoms, GO for functions).
- Static resource — no updates since v1.0, but foundational design pattern for many successors.

### 1.4 OptimusKG (Harvard / Zitnik Lab, 2025)

| Attribute | Value |
|---|---|
| **Data model** | **Labeled Property Graph (LPG)** in Apache Parquet |
| **Scale** | 190,531 nodes, 21.8M edges, 67.2M property instances, 110.3M property values |
| **Node types** | 10 |
| **Edge types** | 26 |
| **Properties** | 150 distinct property keys |
| **Sources** | 65 resources, 18 ontologies/controlled vocabularies |
| **Framework** | BioCypher + Biolink Model |

**Key design decisions — the current state of the art:**
- **Top-level schema enforcement**: every node has `id` (CURIE format, e.g., `ENSG00000141510`), `label` (type abbreviation), `properties` (typed struct), `xrefs` (cross-references), `sources` (provenance).
- **Type-specific property structs**: Gene nodes have `transcription_start_site`, `transcript_ids`, `function_descriptions`; Drug nodes have SMILES, mechanism of action, etc.
- **Cross-references as structured data**: `xrefs: [{id: "P04637", source: "UniProt"}, {id: "7157", source: "Entrez"}]` — enabling ID bridging without transitive inference.
- **Provenance tracking**: `sources: {direct: ["DisGeNET"], indirect: ["GWAS Catalog"]}` on both nodes and edges.
- **Validation**: PaperQA3 agent verified 70% of sampled edges have literature support; 83.4% of false edges had no support.
- **Distribution**: Parquet files → loadable as Polars DataFrames or NetworkX MultiDiGraph.

### 1.5 Bioteque (IRB Barcelona, 2022)

| Attribute | Value |
|---|---|
| **Data model** | Heterogeneous KG → **pre-calculated embeddings** |
| **Scale** | 450K+ nodes, 30M+ edges |
| **Entity types** | 12 (Gene, Disease, Compound, Cell Line, Tissue, Pathway, Molecular Function, Biological Process, Cellular Component, Pharmacologic Class, Side Effect, Anatomy) |
| **Relation types** | 67 |
| **Sources** | 150+ databases |
| **Output format** | Fixed-dimension vector embeddings per (entity, metapath) pair |

**Key design decisions:**
- Embedding-first philosophy: the graph exists to produce embeddings, not to be queried directly.
- ~1,000 metapath-based descriptors per entity capture distinct functional contexts (e.g., "gene—interacts—gene—associated—disease" gives a different embedding than "gene—expressed_in—tissue").
- Embeddings are 128-dimensional vectors, pre-computed and downloadable.
- Designed for off-the-shelf ML: plug embeddings directly into classifiers, clustering, drug repurposing pipelines.

### 1.6 BioMedGraphica (Washington U / Li Lab, 2024)

| Attribute | Value |
|---|---|
| **Data model** | **Text-Attributed Knowledge Graph (TAKG)** with Textual-Numeric Graph (TNG) export |
| **Scale** | 2.3M+ entities, 27M+ relations (full); 3.1M entities, 56.8M relations (v2) |
| **Entity types** | 11 (Promoter, Gene, Transcript, Protein, Pathway, Metabolite, Microbiota, Phenotype, Disease, Drug, + Clinical) |
| **Relation types** | 30 |
| **Sources** | 43 databases |

**Key design decisions:**
- **TNG format**: textual attributes (names, function descriptions, biological annotations) + numeric attributes (expression vectors, mutation profiles, multi-omic measurements) co-exist on each node.
- Soft entity matching via BioBERT embeddings for cross-database harmonization.
- GUI for generating user-specific subgraphs from multi-omic input data.
- Output: `.npy` files for direct GNN/foundation model training.

---

## 2. Comparison of Representation Approaches

### 2.1 Property Graph (NetworkX / Neo4j Style)

**Used by:** SPOKE, Hetionet, OptimusKG, PrimeKG (via NetworkX)

| Aspect | Assessment |
|---|---|
| **Pros** | Intuitive typed nodes/edges; arbitrary key-value properties on any element; efficient traversals (Cypher, Gremlin); native support for numeric features as node properties; rich ecosystem (Neo4j, Neptune, Memgraph); direct LLM/agent integration via Cypher |
| **Cons** | No built-in ontological reasoning (no subsumption, no transitive closure); schema enforcement is application-level; property types not strongly constrained without additional tooling; federation across instances requires custom logic |
| **Numeric data** | Store as node properties (`expression_vector: [0.1, 0.3, ...]`) or as connected feature nodes. Neo4j 5.x supports native vector indexes for similarity search. |
| **Provenance** | Edge properties: `{source: "DisGeNET", confidence: 0.87, pmid: "12345678"}` |
| **Tooling maturity** | Excellent. Neo4j GraphRAG, LangChain, LlamaIndex all have first-class property graph support. |

**Verdict:** Best general-purpose choice for agent systems. Strong query language, rich ecosystem, handles heterogeneous data well.

### 2.2 RDF / OWL with Ontology Alignment

**Used by:** Bio2RDF, Wikidata (biomedical subset), Open PHACTS (historical), some FAIR-compliant resources

| Aspect | Assessment |
|---|---|
| **Pros** | Standards-based (W3C); formal semantics enable reasoning (subsumption, transitivity, disjointness); SPARQL is expressive; excellent for ontology alignment and federated queries across endpoints; strong provenance via named graphs/reification |
| **Cons** | Triple-based model is verbose for property-rich entities (reification overhead); SPARQL endpoints often slow for large-scale traversals; poor support for numeric vectors (no native array types); steep learning curve; limited LLM/agent tooling |
| **Numeric data** | Must be serialized as literals or linked via blank nodes — awkward for high-dimensional vectors |
| **Provenance** | Named graphs or RDF-star for statement-level metadata — powerful but complex |
| **Tooling maturity** | Mature for traditional bioinformatics; poor for modern ML/agent workflows. |

**Verdict:** Valuable for cross-institutional data federation and ontology-heavy tasks, but too heavyweight and inflexible for an agent memory system that needs fast traversals and dense features.

### 2.3 Hypergraph for N-ary Relationships

**Used by:** HyperADRs (drug-gene-ADR triads), HIT (disease-gene classification), HyperRAG (RAG systems)

| Aspect | Assessment |
|---|---|
| **Pros** | Naturally captures n-ary interactions (drug + gene + ADR + context); reduces semantic fragmentation vs. decomposing into binary triples; more compact representation for complex events; HyperRAG shows reduced path explosion vs. binary KG-RAG |
| **Cons** | Very limited tooling (no production-grade hypergraph databases); most implementations require custom data structures; no standard query language; hard to visualize; incompatible with existing KG embedding methods without conversion; small research community |
| **Numeric data** | Custom — typically through hypergraph neural networks (HGCN) that learn node embeddings over hyperedges |
| **Provenance** | No standard patterns; must be custom-designed per hyperedge |
| **Tooling maturity** | Research-stage only. PyTorch Geometric has basic hypergraph support. |

**Verdict:** Theoretically superior for multi-entity interactions (e.g., "drug X inhibits gene Y in tissue Z causing ADR W"), but impractical as a primary storage/query layer. Best used as a computation layer on top of a property graph.

### 2.4 Embedding-First (TransE, RotatE, Learned Representations)

**Used by:** Bioteque (metapath embeddings), BioMedKG (contrastive learning), various drug discovery pipelines

| Aspect | Assessment |
|---|---|
| **Pros** | Unified vector space enables similarity search across entity types; handles missing data gracefully (link prediction); RotatE captures symmetry/antisymmetry/composition patterns; direct input to ML pipelines; compresses complex graph topology into fixed-size vectors |
| **Cons** | Opaque — embeddings are not human-interpretable; no structured query capability ("what drugs target gene X?" requires decode step); information loss from compression; need retraining when graph updates; RotatE struggles with >1M nodes without careful engineering; no provenance or confidence per-relationship |
| **Numeric data** | This IS the numeric data — but original structured features are lost |
| **Provenance** | Not representable in embedding space |
| **Tooling maturity** | Good for ML (PyKEEN, DGL-KE, TorchKGE); poor for agent queries. |

**RotatE vs TransE for biomedical KGs:**
- TransE: simple h + r ≈ t translation; fast; struggles with 1-to-many and symmetric relations (common in biology).
- RotatE: rotation in complex space; captures all relation patterns; consistently outperforms TransE on Hetionet and BioKG benchmarks.
- Both are most useful as a downstream task (link prediction, drug repurposing scoring) rather than as a primary representation.

**Verdict:** Essential complement to structured graphs, but cannot serve as the primary representation for an agent that needs to answer structured queries, explain reasoning, and track provenance.

### 2.5 Hybrid: Structured Graph + Vector Store

**Used by:** AlzKB (Cedars-Sinai), biomedical GraphRAG systems, Neo4j + Qdrant pipelines, KG-RAG (SPOKE)

| Aspect | Assessment |
|---|---|
| **Pros** | Best of both worlds: structured traversals for relational queries + vector similarity for semantic/dense features; supports both "what genes does drug X target?" (Cypher) and "find similar expression profiles" (vector search); agent-native architecture with tool-calling orchestration; incremental updates to both layers independently |
| **Cons** | Operational complexity (two systems to maintain); consistency between graph and vector store must be managed; query routing logic needed (which questions go where); more infrastructure |
| **Numeric data** | Dense features (embeddings, expression vectors) in vector store; structured features (molecular weight, chromosome) as graph properties |
| **Provenance** | In the graph layer, same as property graph |
| **Tooling maturity** | Rapidly maturing. Neo4j 5.x has native vector indexes. LangChain/LlamaIndex support hybrid retrieval. |

**Verdict:** The recommended approach for an agent system. Provides the query flexibility, provenance tracking, and dense feature support needed for the use case described.

---

## 3. Cross-Cutting Concerns

### 3.1 Cross-Referencing IDs Across Databases

This is one of the hardest practical problems in biomedical KG construction. Key lessons from the literature:

**The problem:** A single human gene can be identified by:
- HGNC symbol: `TP53`
- Entrez Gene ID: `7157`
- Ensembl Gene ID: `ENSG00000141510`
- UniProt accession: `P04637`
- OMIM: `191170`
- RefSeq: `NM_000546.6`

**Best practices from leading KGs:**

1. **Use CURIE-format canonical IDs** (OptimusKG approach): `ENSG00000141510`, `UniProtKB:P04637`. Prefer Ensembl for genes (versioned, stable, genomic-coordinate-linked).

2. **Store cross-references as structured xref lists**, not as identity assertions:
   ```json
   {
     "id": "ENSG00000141510",
     "xrefs": [
       {"source": "UniProt", "id": "P04637"},
       {"source": "Entrez", "id": "7157"},
       {"source": "HGNC", "id": "HGNC:11998"}
     ]
   }
   ```

3. **Distinguish relationship types** (BED approach):
   - `corresponds_to`: same biological entity, different database (Gene → Protein from same gene)
   - `is_associated_to`: related but different scope (Gene ID → Protein ID from same locus)
   - `is_replaced_by`: deprecated ID mapping

4. **Use UniProt idmapping as the gold standard bridge** between protein/gene ID systems. The `idmapping_selected.tab` file provides direct mappings: UniProtKB-AC ↔ GeneID (Entrez) ↔ RefSeq ↔ Ensembl ↔ PDB ↔ GO.

5. **Avoid transitive inference** unless validated. Mapping A→B and B→C does not guarantee A→C is biologically meaningful. BED (Biological Entity Dictionary) uses a Neo4j graph of ID relationships to find shortest paths while respecting scope boundaries.

6. **Version your IDs**: Protein sequences and gene annotations change over time. Always store `(id, version, timestamp)` tuples. UniParc IDs provide permanent sequence-level identity.

### 3.2 Attaching Numeric Data to Graph Nodes

Three patterns observed in the literature:

**Pattern A: Node properties (OptimusKG, Hetionet)**
```
Gene node → properties.expression_summary = [0.1, 0.3, 0.7, ...]
Edge → properties.z_score = -4.156
```
Best for: low-dimensional, structured numeric features (scores, counts, coordinates).

**Pattern B: Separate embedding store (Bioteque, BioMedKG)**
```
Gene node ID → vector store lookup → 128-dim embedding
```
Best for: high-dimensional learned representations, similarity search.

**Pattern C: Textual-Numeric Graph / TNG (BioMedGraphica)**
```
Gene node → text_attrs: {name, function, GO_terms}
           → numeric_attrs: {expression_vector, mutation_profile, methylation}
```
Best for: combined ML pipelines where both modalities feed into GNNs or foundation models.

**Recommendation:** Use Pattern A for structured features and Pattern B for dense embeddings, in a hybrid architecture.

### 3.3 Provenance and Confidence Scoring

The field has converged on a multi-signal approach:

**Evidence tiering (HEG-TKG, 2025):**
- **GOLD** (confidence ≥ 0.95): Cross-source confirmed (e.g., guideline + independent literature)
- **SILVER** (confidence ≥ 0.85): Multi-model or multi-document consensus
- **BRONZE** (confidence ≥ 0.70): Single source, single extraction

**Composite scoring (EvidenceNet, 2025):**
```
S(e) = w_design × StudyDesign + w_impact × ImpactFactor + 
       w_stats × StatisticalSignificance + w_sample × SampleSize + 
       w_llm × ExtractionConfidence
```
Mapped to grades: A (≥0.8), B (0.6-0.8), C (0.4-0.6), D (<0.4).

**Practical provenance schema (recommended):**
```json
{
  "source_id": "PMID:12345678",
  "source_type": "experimental",        // experimental | computational | curated | text-mined
  "database": "DisGeNET",
  "evidence_score": 0.87,
  "evidence_tier": "silver",
  "extraction_method": "curated",        // curated | nlp | llm_consensus
  "study_design": "case_control",        // rct | cohort | case_control | case_report | in_vitro | computational
  "timestamp": "2024-03-15",
  "llm_consensus_fraction": null         // 0.0-1.0 if LLM-extracted
}
```

---

## 4. Concrete Recommendation for Your Use Case

### Use Case Summary
An agentic scientific discovery system that needs to:
- Query structured relations (gene→pathway, drug→target, disease→gene)
- Search dense features (expression profiles, paper embeddings, molecular fingerprints)
- Maintain provenance and confidence across heterogeneous evidence
- Support cross-modal queries ("find drugs targeting genes with similar expression profiles to my DE genes")

### Recommended Architecture: Hybrid Property Graph + Vector Store

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATOR                        │
│   (routes queries to appropriate retrieval backend)         │
├──────────────────────┬──────────────────────────────────────┤
│                      │                                      │
│  ┌───────────────────▼──────────────┐  ┌───────────────────▼──┐
│  │     PROPERTY GRAPH (Primary)     │  │   VECTOR STORE       │
│  │                                  │  │                      │
│  │  Typed nodes with properties:    │  │  Collections:        │
│  │  • Gene, Drug, Disease, Pathway  │  │  • gene_embeddings   │
│  │  • Paper, Protein, Phenotype     │  │  • paper_embeddings  │
│  │  • Experiment, Expression        │  │  • drug_fingerprints │
│  │                                  │  │  • expression_sigs   │
│  │  Typed edges with provenance:    │  │                      │
│  │  • targets, associated_with      │  │  Each vector linked  │
│  │  • participates_in, expressed_in │  │  to graph node by ID │
│  │  • treats, contraindicates       │  │                      │
│  │  • findings, hypotheses          │  │                      │
│  └──────────────────────────────────┘  └──────────────────────┘
│                                                              │
│  Implementation options:                                     │
│  • NetworkX (in-memory, <1M nodes) — current project scale   │
│  • Neo4j (persistent, >1M nodes) — production scale          │
│  • Vector: numpy/FAISS (in-memory) or Qdrant (persistent)    │
└──────────────────────────────────────────────────────────────┘
```

### 5. Schema Example for Recommended Approach

#### 5.1 Node Types and Properties

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class NodeType(Enum):
    GENE = "gene"
    PROTEIN = "protein"
    DRUG = "drug"
    DISEASE = "disease"
    PATHWAY = "pathway"
    PHENOTYPE = "phenotype"
    PAPER = "paper"
    EXPERIMENT = "experiment"
    EXPRESSION_DATASET = "expression_dataset"
    BIOLOGICAL_PROCESS = "biological_process"
    MOLECULAR_FUNCTION = "molecular_function"
    ANATOMY = "anatomy"

@dataclass
class CrossReference:
    source: str           # "UniProt", "Entrez", "Ensembl", "HGNC", "DrugBank"
    id: str               # External identifier
    id_version: Optional[str] = None  # For versioned IDs

@dataclass
class Provenance:
    source_id: Optional[str] = None     # "PMID:12345678"
    source_type: str = "curated"        # experimental | computational | curated | text_mined
    database: Optional[str] = None      # "DisGeNET", "DrugBank", etc.
    evidence_score: float = 0.5
    evidence_tier: str = "bronze"       # gold | silver | bronze
    extraction_method: str = "curated"  # curated | nlp | llm_consensus
    timestamp: Optional[str] = None

@dataclass
class KGNode:
    """Base node in the knowledge graph."""
    id: str               # CURIE format: "ENSG00000141510", "DrugBank:DB00945"
    node_type: NodeType
    name: str             # Human-readable: "TP53", "Aspirin"
    description: str = ""
    xrefs: list[CrossReference] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    properties: dict = field(default_factory=dict)  # Type-specific


# Type-specific property examples:

gene_properties = {
    "symbol": "TP53",
    "chromosome": "17",
    "start_position": 7661779,
    "end_position": 7687550,
    "strand": "-",
    "biotype": "protein_coding",
    "function_descriptions": ["Tumor suppressor", "Transcription factor"],
    "go_terms": ["GO:0006915", "GO:0042981"],  # Apoptosis, regulation of apoptosis
}

drug_properties = {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "molecular_weight": 180.16,
    "drug_groups": ["approved"],
    "mechanism_of_action": "Irreversible COX inhibitor",
    "atc_codes": ["B01AC06", "N02BA01"],
}

paper_properties = {
    "pmid": "12345678",
    "doi": "10.1038/s41586-024-00001-0",
    "title": "...",
    "abstract": "...",
    "year": 2024,
    "journal": "Nature",
    "mesh_terms": ["Genes, Tumor Suppressor", "Drug Therapy"],
}

expression_dataset_properties = {
    "dataset_id": "GSE12345",
    "platform": "RNA-seq",
    "organism": "Homo sapiens",
    "tissue": "liver",
    "condition": "hepatocellular_carcinoma",
    "n_samples": 120,
    "n_genes": 20000,
    # Dense expression matrix stored in vector store, not here
}
```

#### 5.2 Edge Types and Properties

```python
class EdgeType(Enum):
    # Gene/Protein relations
    TARGETS = "targets"                     # Drug → Gene/Protein
    INTERACTS_WITH = "interacts_with"       # Protein ↔ Protein
    REGULATES = "regulates"                 # Gene → Gene (directed)
    EXPRESSED_IN = "expressed_in"           # Gene → Anatomy

    # Disease relations
    ASSOCIATED_WITH = "associated_with"     # Gene ↔ Disease
    TREATS = "treats"                       # Drug → Disease
    CONTRAINDICATES = "contraindicates"     # Drug → Disease
    PRESENTS_WITH = "presents_with"         # Disease → Phenotype

    # Pathway/Function relations
    PARTICIPATES_IN = "participates_in"     # Gene → Pathway
    HAS_FUNCTION = "has_function"           # Gene → Molecular Function
    INVOLVED_IN = "involved_in"            # Gene → Biological Process

    # Literature relations
    MENTIONS = "mentions"                   # Paper → Gene/Drug/Disease
    SUPPORTS = "supports"                   # Paper → Finding
    CITES = "cites"                         # Paper → Paper

    # Agent memory relations
    FINDING_ABOUT = "finding_about"         # Finding → Gene/Disease/Drug
    HYPOTHESIS_INVOLVES = "hypothesis_involves"
    EXPERIMENT_TESTS = "experiment_tests"   # Experiment → Hypothesis
    DIFFERENTIALLY_EXPRESSED = "differentially_expressed"  # Gene → ExpressionDataset

@dataclass
class KGEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    directed: bool = True
    provenance: list[Provenance] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

# Edge property examples:

drug_targets_gene = {
    "action": "inhibitor",          # inhibitor | agonist | antagonist | ...
    "known_action": True,
    "pharmacological_action": True,
}

gene_disease_association = {
    "score": 0.87,                  # Association score (e.g., DisGeNET GDA score)
    "n_publications": 42,
    "n_snps": 5,
    "source_databases": ["DisGeNET", "ClinVar", "GWAS Catalog"],
}

differential_expression = {
    "log2_fold_change": 2.5,
    "p_value": 1.2e-8,
    "adjusted_p_value": 3.4e-6,
    "direction": "up",
    "comparison": "tumor_vs_normal",
}
```

#### 5.3 Vector Store Collections

```python
vector_collections = {
    "gene_embeddings": {
        "source": "ESM-2 protein embeddings or BioBERT function descriptions",
        "dimension": 768,
        "metadata": ["gene_id", "symbol", "node_type"],
        "use_case": "Find functionally similar genes",
    },
    "paper_embeddings": {
        "source": "PubMedBERT or Specter2 from title+abstract",
        "dimension": 768,
        "metadata": ["pmid", "year", "mesh_terms"],
        "use_case": "Semantic literature search",
    },
    "drug_fingerprints": {
        "source": "Morgan fingerprint or Uni-Mol embeddings from SMILES",
        "dimension": 1024,
        "metadata": ["drug_id", "name", "drug_groups"],
        "use_case": "Find structurally similar compounds",
    },
    "expression_signatures": {
        "source": "PCA-reduced expression profiles per gene-per-dataset",
        "dimension": 128,
        "metadata": ["gene_id", "dataset_id", "condition"],
        "use_case": "Find genes with similar expression patterns",
    },
    "finding_embeddings": {
        "source": "Agent finding statements embedded via sentence-transformers",
        "dimension": 384,
        "metadata": ["finding_id", "confidence", "timestamp"],
        "use_case": "Retrieve relevant past findings for hypothesis generation",
    },
}
```

#### 5.4 Agent Query Routing

```python
def route_query(query: str, query_type: str):
    """
    Route agent queries to the appropriate backend.
    
    Structured queries → Property graph (Cypher / NetworkX traversal)
    Similarity queries → Vector store
    Hybrid queries → Both, then merge
    """
    routing = {
        # Pure graph queries
        "what_targets": "graph",       # "What does drug X target?"
        "pathway_genes": "graph",      # "What genes are in pathway Y?"
        "drug_indications": "graph",   # "What diseases does drug X treat?"
        "gene_diseases": "graph",      # "What diseases is gene X associated with?"
        "multi_hop": "graph",          # "Find drugs that target genes in pathway X"
        
        # Pure vector queries
        "similar_genes": "vector",     # "Find genes with similar function to X"
        "similar_papers": "vector",    # "Find papers about topic X"
        "similar_drugs": "vector",     # "Find drugs structurally similar to X"
        "similar_expression": "vector", # "Find genes with similar expression"
        "past_findings": "vector",     # "What have we found about topic X?"
        
        # Hybrid queries
        "drugs_for_similar_targets": "hybrid",  
            # Vector: find genes similar to query gene
            # Graph: find drugs targeting those genes
        "literature_context": "hybrid",
            # Vector: find relevant papers
            # Graph: extract structured relations from those papers' mentions
        "expression_pathway": "hybrid",
            # Vector: find genes with matching expression signature
            # Graph: what pathways are those genes in?
    }
    return routing.get(query_type, "hybrid")
```

---

## 6. Summary Comparison Table

| Approach | Query Power | Dense Features | Provenance | Tooling | Agent-Ready | Used By |
|---|---|---|---|---|---|---|
| **Property Graph** | Excellent (Cypher) | Good (node props) | Good (edge props) | Excellent | Yes | SPOKE, Hetionet, OptimusKG |
| **RDF/OWL** | Good (SPARQL) | Poor | Excellent (named graphs) | Moderate | Poor | Bio2RDF, Open PHACTS |
| **Hypergraph** | Poor (no std lang) | Custom only | Poor | Research-only | No | HyperADRs, HIT |
| **Embedding-First** | Poor (no queries) | Excellent | None | Good (ML) | Partial | Bioteque |
| **Hybrid Graph+Vector** | Excellent | Excellent | Good | Rapidly maturing | Yes | AlzKB, KG-RAG, GraphRAG |

---

## 7. Key References

| Resource | Year | Type | Key Contribution |
|---|---|---|---|
| **PrimeKG** | 2023 | KG | 10-type disease-centric multimodal KG, text-augmented nodes |
| **PrimeKG++ / BioMedKG** | 2025 | KG + ML | Sequence/SMILES enrichment + contrastive graph learning |
| **SPOKE** | 2023 | KG | 21-type Neo4j graph, 41 sources, weekly updates, KG-RAG |
| **OptimusKG** | 2025 | KG | State-of-the-art LPG with 150 property keys, CURIE IDs, PaperQA3 validation |
| **Hetionet** | 2017 | KG | Foundational hetnet design, metagraph schema pattern |
| **Bioteque** | 2022 | Embeddings | 1000+ metapath embeddings from 150 sources |
| **BioMedGraphica** | 2024 | KG + Platform | TNG format, text+numeric on nodes, 43 databases |
| **ChronoMedKG** | 2025 | KG | Temporal metadata, 6-signal evidence grading, PMID provenance |
| **HEG-TKG** | 2025 | KG | Gold/Silver/Bronze evidence tiering for clinical reasoning |
| **EvidenceNet** | 2025 | KG | Composite confidence scoring from study design + statistics |
| **MedKGent** | 2025 | Agent | LLM agent framework for temporally-evolving medical KG |
| **BED** | 2018 | ID Mapping | Graph-based biological ID resolution in Neo4j |
| **HyperADRs** | 2025 | Hypergraph | Drug-gene-ADR triad prediction via hypergraph convolution |
| **HyperRAG** | 2025 | RAG | N-ary fact retrieval over hypergraphs for LLM augmentation |
| **BiomedGPT** | 2024 | Foundation Model | Unified vision-language model, discrete token vocabulary |
| **KG-RAG** | 2024 | RAG | SPOKE + LLM integration with prompt-aware context extraction |
